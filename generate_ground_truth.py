import sys
from pathlib import Path
import json
import time

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.builders.opportunity.structured_opportunity_builder import StructuredOpportunityBuilder
from src.builders.profil.structured_profil_builder import StructuredProfileBuilder
from src.embeddings.embedder import Embedder
from src.embeddings.opportunity_index import build_opportunity_index
from src.llm.client import LLMClient
from src.loaders import DataLoader
from src.matching.ranker import OpportunityRanker
from src.evaluation.benchmark import EvalQuery, EvalDataset

# Config
NUM_RESEARCHERS_TO_PROCESS = 5 
CANDIDATES_PER_RESEARCHER = 20 # How many ops to show the LLM to judge


def build_judge_prompt(researcher, opportunity) -> str:
    return f"""
You are an expert scientific grant reviewer. 

Your task is to determine if the following research opportunity is a HIGHLY RELEVANT match for the researcher.
A match is "HIGHLY RELEVANT" only if the researcher's domains, skills, and interests strongly align with the opportunity's topics, keywords, and description.

RESEARCHER:
Name: {researcher.fullname}
Institution: {researcher.institution}
Domains: {", ".join(researcher.research_domains)}
Skills: {", ".join(researcher.skills)}
Keywords: {", ".join(researcher.keywords)}

OPPORTUNITY:
Title: {opportunity.title}
Organization: {opportunity.organization}
Description: {opportunity.description[:500]}...
Topics: {", ".join(opportunity.topics)}
Keywords: {", ".join(opportunity.keywords)}

Is this a highly relevant match? 
Answer EXACTLY with the word "YES" or "NO". Do not include any other text or explanation.
"""

def main():
    print("Loading datasets...", flush=True)
    researchers = DataLoader.load_researchers(PROJECT_ROOT / "data" / "mock" / "researchers.json")
    opportunities = DataLoader.load_opportunities(PROJECT_ROOT / "data" / "processed" / "opportunities.json")
    
    opportunities_by_id = {opp.id: opp for opp in opportunities}

    # Initialize components
    profile_builder = StructuredProfileBuilder()
    opportunity_builder = StructuredOpportunityBuilder()
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    ranker = OpportunityRanker()
    llm_judge = LLMClient(provider="gemini") # Using Gemini as the judge

    print(f"Building index to find candidate pools...", flush=True)
    index_records = build_opportunity_index(
        opportunities,
        opportunity_builder,
        embedder,
        show_progress=True,
    )

    generated_queries = []

    # Only process a subset to avoid API limits and long wait times
    target_researchers = researchers[:NUM_RESEARCHERS_TO_PROCESS]

    for i, researcher in enumerate(target_researchers, 1):
        print(f"\n[{i}/{len(target_researchers)}] Generating ground truth for: {researcher.fullname}")
        
        # 1. Coarse retrieval to find candidates to judge
        # We fetch top N to give the LLM a pool of likely candidates to filter.
        researcher_text = profile_builder.build(researcher)
        researcher_vector = embedder.encode(researcher_text)
        
        retrieval_results = ranker.rank(
            researcher_vector,
            index_records,
            top_k=CANDIDATES_PER_RESEARCHER,
        )
        
        relevant_ids = []
        irrelevant_ids = []

        # 2. LLM Judge evaluates each candidate
        for rank, item in enumerate(retrieval_results, 1):
            opp_id = item["opportunity_id"]
            opp = opportunities_by_id[opp_id]
            
            prompt = build_judge_prompt(researcher, opp)
            try:
                response = llm_judge.generate(prompt).strip().upper()
                
                # Check for strictly YES or NO
                if "YES" in response:
                    relevant_ids.append(opp_id)
                    print(f"  [{rank}/{CANDIDATES_PER_RESEARCHER}] {opp_id} -> RELEVANT")
                else:
                    irrelevant_ids.append(opp_id)
                    print(f"  [{rank}/{CANDIDATES_PER_RESEARCHER}] {opp_id} -> IRRELEVANT")
                    
            except Exception as e:
                print(f"  [{rank}/{CANDIDATES_PER_RESEARCHER}] LLM Error on {opp_id}: {e}")
                
            time.sleep(1) # Rate limit protection

        # Save the query
        query = EvalQuery(
            query_id=f"auto-eval-{researcher.id}",
            researcher_id=researcher.id,
            relevant_opportunity_ids=relevant_ids,
            irrelevant_opportunity_ids=irrelevant_ids,
            notes=f"Auto-generated using LLM-as-a-judge from top {CANDIDATES_PER_RESEARCHER} coarse candidates."
        )
        generated_queries.append(query)

    # Save to disk
    dataset = EvalDataset(generated_queries)
    out_path = PROJECT_ROOT / "data" / "eval" / "benchmark_queries.json"
    dataset.save_to_json(out_path)
    print(f"\nDone! Generated ground-truth for {len(generated_queries)} researchers and saved to {out_path}.")

if __name__ == "__main__":
    main()
