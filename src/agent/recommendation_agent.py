"""RecommendationAgent — central orchestration of the recommendation pipeline.

Loads data, builds the FAISS vector index, and answers queries for a researcher
by running: profile → embedding → FAISS retrieval (top-K) → LLM reranking (top-N).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from src.agent.models import RecommendationItem, RecommendationResult
from src.builders.opportunity.structured_opportunity_builder import StructuredOpportunityBuilder
from src.builders.profil.structured_profil_builder import StructuredProfileBuilder
from src.collectors.funding_tenders_collector import FundingTendersCollector
from src.config import PipelineConfig
from src.embeddings.embedder import Embedder
from src.embeddings.opportunity_index import build_opportunity_index
from src.llm.client import LLMClient
from src.llm.errors import LLMError
from src.loaders import DataLoader
from src.reranking.llm_reranker import LLMReranker
from src.search.profile_terms import extract_profile_terms
from src.vector_store.faiss_vector_index import FAISSVectorIndex

logger = logging.getLogger(__name__)


class RecommendationAgent:
    """End-to-end recommendation agent orchestrating the whole pipeline.

    Usage::

        agent = RecommendationAgent()
        agent.load_data()
        agent.build_index()                       # builds + keeps in memory
        agent.build_index(persist_dir="out/idx")  # builds + saves to disk
        result = agent.query(1)

        # Restore from disk:
        agent.load_index("out/idx")
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

        self.profile_builder = StructuredProfileBuilder()
        self.opportunity_builder = StructuredOpportunityBuilder()
        self.embedder = Embedder(self.config.embedding_model)

        self.llm_rerankers: list[tuple[str, LLMReranker]] = [
            (name, LLMReranker(LLMClient(provider=name)))
            for name in self.config.llm_providers
        ]

        # Loaded state
        self.researchers: list[Any] = []
        self.opportunities: list[Any] = []
        self.researchers_by_id: dict = {}
        self.opportunities_by_id: dict = {}
        self.index_records: list[dict[str, Any]] = []
        self.vector_index: FAISSVectorIndex | None = None
        self._index_built = False

    # ------------------------------------------------------------------
    # Lifecycle: load data
    # ------------------------------------------------------------------

    def load_data(
        self,
        researchers_path: Optional[str | Path] = None,
        opportunities_path: Optional[str | Path] = None,
    ) -> "RecommendationAgent":
        """Load researchers and opportunities from disk."""
        researchers_file = Path(researchers_path) if researchers_path else self.config.researchers_file
        opportunities_file = Path(opportunities_path) if opportunities_path else self.config.opportunities_file

        self.researchers = DataLoader.load_researchers(researchers_file)
        self.opportunities = DataLoader.load_opportunities(opportunities_file)

        self.researchers_by_id = {r.id: r for r in self.researchers}
        self.opportunities_by_id = {o.id: o for o in self.opportunities}

        logger.info(
            "Loaded %d researchers and %d opportunities",
            len(self.researchers),
            len(self.opportunities),
        )

        if not self.researchers:
            raise ValueError("No researchers found in the loaded data.")
        if not self.opportunities:
            raise ValueError("No opportunities found in the loaded data.")

        return self

    # ------------------------------------------------------------------
    # Lifecycle: build / load index
    # ------------------------------------------------------------------

    def build_index(
        self,
        show_progress: bool = False,
        persist_dir: Optional[str | Path] = None,
    ) -> "RecommendationAgent":
        """Embed all opportunities and build the FAISS index.

        Parameters
        ----------
        show_progress:
            Display a tqdm progress bar during batch encoding.
        persist_dir:
            When set the index is saved to this directory (e.g.
            ``"vector_store/index"``).
        """
        if not self.opportunities:
            raise RuntimeError("No opportunities loaded — call load_data() first.")

        logger.info("Building opportunity index (%d records)...", len(self.opportunities))
        t0 = time.perf_counter()

        self.index_records = build_opportunity_index(
            self.opportunities,
            self.opportunity_builder,
            self.embedder,
            show_progress=show_progress,
        )

        self.vector_index = FAISSVectorIndex(dimension=self.embedder.embedding_dim)
        self.vector_index.build(self.index_records)
        self._index_built = True

        elapsed = time.perf_counter() - t0
        logger.info(
            "FAISS index built: %d vectors, dim=%d, in %.1fs",
            self.vector_index.size,
            self.embedder.embedding_dim,
            elapsed,
        )

        if persist_dir is not None:
            self.vector_index.save(persist_dir)
            logger.info("Index saved to %s", persist_dir)

        return self

    def load_index(self, directory: str | Path) -> "RecommendationAgent":
        """Restore a previously-saved FAISS index from disk.

        After calling this, ``load_data()`` must still be called before
        ``query()`` to have the researcher and opportunity objects available.
        """
        self.vector_index = FAISSVectorIndex.load(directory)
        self._index_built = True
        logger.info(
            "Loaded FAISS index: %d vectors, dim=%d",
            self.vector_index.size,
            self.vector_index.dimension,
        )
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query(self, researcher_id: int | str) -> RecommendationResult:
        """Return recommendations for a researcher by ID."""
        researcher = self.researchers_by_id.get(researcher_id)
        if researcher is None:
            raise KeyError(f"Researcher with id {researcher_id!r} not found.")
        return self._recommend(researcher)

    def query_from_profile(self, researcher: Any) -> RecommendationResult:
        """Return recommendations for an ad-hoc researcher object/profile."""
        return self._recommend(researcher)

    def refresh_for_profile(
        self,
        researcher: Any,
        max_terms: int = 8,
        max_workers: int = 4,
    ) -> dict[str, Any]:
        """Fetch profile-specific F&T results and refresh the local index."""
        collector = FundingTendersCollector()
        fetched = collector.search_for_profile(
            researcher,
            max_terms=max_terms,
            max_workers=max_workers,
        )

        existing_by_id = {
            str(opportunity.id).strip(): opportunity
            for opportunity in self.opportunities
        }
        changed = False
        added = 0
        updated = 0

        for opportunity in fetched:
            opportunity_id = str(opportunity.id).strip()
            existing = existing_by_id.get(opportunity_id)
            if existing is None:
                self.opportunities.append(opportunity)
                existing_by_id[opportunity_id] = opportunity
                added += 1
                changed = True
            elif existing != opportunity:
                index = self.opportunities.index(existing)
                self.opportunities[index] = opportunity
                existing_by_id[opportunity_id] = opportunity
                updated += 1
                changed = True

        if changed:
            self.opportunities_by_id = {
                opportunity.id: opportunity for opportunity in self.opportunities
            }
            self.build_index(
                show_progress=False,
                persist_dir=self.config.index_persist_path,
            )

        return {
            "terms": extract_profile_terms(researcher, max_terms=max_terms),
            "fetched": len(fetched),
            "added": added,
            "updated": updated,
            "index_refreshed": changed,
        }

    def _recommend(self, researcher: Any) -> RecommendationResult:
        if not self._index_built or self.vector_index is None:
            raise RuntimeError("Index not built — call build_index() or load_index() first.")

        t_start = time.perf_counter()

        # 1. Embed the researcher profile.
        researcher_text = self.profile_builder.build(researcher)
        researcher_vector = self.embedder.encode(researcher_text)

        # 2. Retrieve top-K candidates via FAISS.
        retrieval_results = self.vector_index.search(
            researcher_vector,
            top_k=self.config.top_k_retrieval,
        )
        top_candidates = retrieval_results[: self.config.top_k_retrieval]
        candidate_opps = []
        for item in top_candidates:
            opportunity = self._find_opportunity(item["opportunity_id"])
            if opportunity is not None:
                candidate_opps.append(opportunity)

        retrieval_top = self._to_items(top_candidates[: self.config.top_k_final])

        # 3. Rerank with each LLM provider.
        provider_results: dict[str, list[RecommendationItem]] = {}
        provider_timing: dict[str, float] = {}
        provider_errors: dict[str, str | None] = {}

        for provider_name, reranker in self.llm_rerankers:
            provider_errors[provider_name] = None
            provider_timing[provider_name] = 0.0

            try:
                t0 = time.perf_counter()
                reranked = reranker.rerank(researcher, candidate_opps)
                elapsed = time.perf_counter() - t0
                provider_timing[provider_name] = elapsed

                recommendations = sorted(
                    reranked.get("recommendations", []),
                    key=lambda r: r.get("score", 0),
                    reverse=True,
                )[: self.config.top_k_final]

                provider_results[provider_name] = self._to_items(recommendations)
            except LLMError as exc:
                provider_errors[provider_name] = str(exc)
                logger.warning("%s rerank failed: %s", provider_name, exc)
                provider_results[provider_name] = self._cosine_fallback(
                    top_candidates, self.config.top_k_final, provider_name
                )
            except Exception as exc:
                logger.exception("%s rerank crashed", provider_name)
                provider_errors[provider_name] = str(exc)
                provider_results[provider_name] = self._cosine_fallback(
                    top_candidates, self.config.top_k_final, provider_name
                )

        # 4. Pick the final top-N from the first provider (primary source).
        primary_with_items = self._pick_primary(provider_results)

        elapsed_total = time.perf_counter() - t_start

        return RecommendationResult(
            researcher_id=researcher.id,
            researcher_name=researcher.fullname,
            institution=researcher.institution,
            recommendations=primary_with_items,
            retrieval_top=retrieval_top,
            provider_results=provider_results,
            provider_timing=provider_timing,
            provider_errors=provider_errors,
            elapsed_total=elapsed_total,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_primary(
        self,
        provider_results: dict[str, list[RecommendationItem]],
    ) -> list[RecommendationItem]:
        names = list(provider_results.keys())
        if not names:
            return []
        return provider_results[names[0]]

    def _cosine_fallback(
        self,
        top_candidates: list[dict[str, Any]],
        top_k: int,
        provider_name: str,
    ) -> list[RecommendationItem]:
        items: list[RecommendationItem] = []
        for item in top_candidates[:top_k]:
            opp = self._find_opportunity(item["opportunity_id"])
            items.append(
                RecommendationItem(
                    opportunity_id=item["opportunity_id"],
                    title=opp.title if opp else "",
                    organization=opp.organization if opp else "",
                    score=round(item["score"] * 100, 1),
                    reason=f"{provider_name} unavailable; cosine fallback.",
                    matching_areas=[],
                    source="retrieval",
                )
            )
        return items

    def _to_items(
        self,
        results: list[dict[str, Any]],
        source: str = "reranked",
    ) -> list[RecommendationItem]:
        items: list[RecommendationItem] = []
        for r in results:
            opp_id = r.get("opportunity_id")
            opp = self._find_opportunity(opp_id)
            items.append(
                RecommendationItem(
                    opportunity_id=opp_id,
                    title=opp.title if opp else r.get("title", ""),
                    organization=opp.organization if opp else r.get("organization", ""),
                    score=r.get("score", 0.0),
                    reason=r.get("reason", ""),
                    matching_areas=r.get("matching_areas", []),
                    source=source,
                    deadline=opp.deadline if opp else r.get("deadline", ""),
                    url=opp.url if opp else r.get("url", ""),
                    opportunity_type=opp.type if opp else r.get("type", ""),
                )
            )
        return items

    def _find_opportunity(self, opportunity_id: Any) -> Any | None:
        """Resolve IDs from LLM/index output without losing opportunity metadata."""
        opportunity = self.opportunities_by_id.get(opportunity_id)
        if opportunity is not None:
            return opportunity

        normalized_id = str(opportunity_id).strip()
        for candidate in self.opportunities:
            if str(candidate.id).strip() == normalized_id:
                return candidate
        return None
