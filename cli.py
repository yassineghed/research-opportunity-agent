"""CLI entry point for the recommendation agent.

Usage:
    python cli.py query <researcher_id> [--json] [--index-dir DIR]
    python cli.py list-researchers
    python cli.py list-opportunities
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agent import RecommendationAgent
from src.config import PipelineConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def build_agent(config: PipelineConfig, index_dir: str | None = None) -> RecommendationAgent:
    agent = RecommendationAgent(config)
    agent.load_data()
    if index_dir and Path(index_dir).exists():
        agent.load_index(index_dir)
    else:
        agent.build_index()
    return agent


def cmd_query(agent, args) -> None:
    result = agent.query(args.researcher_id)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"\nResearcher: {result.researcher_name} - {result.institution}")
        print(f"Total time: {result.elapsed_total:.1f}s")
        print(f"\nTop {len(result.recommendations)} recommendations:")
        for i, item in enumerate(result.recommendations, start=1):
            print(f"{i}. {item.title} - {item.organization}")
            print(f"   Score: {item.score}")
            if item.matching_areas:
                print(f"   Matching areas: {', '.join(item.matching_areas)}")
            if item.reason:
                print(f"   Reason: {item.reason}")
            print()
        for provider, items in result.provider_results.items():
            print(f"{provider}: {len(items)} results, {result.provider_timing[provider]:.1f}s")
            print(f"  Erreurs: {result.provider_errors[provider] or 'aucune'}")


def cmd_list_researchers(agent, args) -> None:
    for r in agent.researchers:
        print(f"{r.id}\t{r.fullname}\t{r.institution}")


def cmd_list_opportunities(agent, args) -> None:
    print(f"Total: {len(agent.opportunities)}")
    for o in agent.opportunities[:20]:
        print(f"{o.id}\t{o.title[:60]}...")


def main() -> None:
    parser = argparse.ArgumentParser(prog="research-opportunity-agent")
    parser.add_argument("--index-dir", type=str, metavar="DIR", default=None, help="Load a pre-built FAISS index from DIR")
    sub = parser.add_subparsers(dest="command", required=True)

    p_query = sub.add_parser("query", help="Get recommendations for a researcher")
    p_query.add_argument("researcher_id", type=int)
    p_query.add_argument("--json", action="store_true", help="Output as JSON")
    p_query.set_defaults(func=cmd_query)

    p_list = sub.add_parser("list-researchers", help="List available researchers")
    p_list.set_defaults(func=cmd_list_researchers)

    p_listo = sub.add_parser("list-opportunities", help="List opportunities")
    p_listo.set_defaults(func=cmd_list_opportunities)

    args = parser.parse_args()

    config = PipelineConfig()
    agent = build_agent(config, index_dir=args.index_dir)
    args.func(agent, args)


if __name__ == "__main__":
    main()
