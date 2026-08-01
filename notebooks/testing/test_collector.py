from src.collectors.cordis_collector import CordisCollector


collector = CordisCollector()

opportunities = collector.collect()

for opportunity in opportunities:
    print(opportunity.title)