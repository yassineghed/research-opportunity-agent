import json
import sys
sys.path.insert(0, ".")

from src.reranking.llm_reranker import LLMReranker

r = LLMReranker.__new__(LLMReranker)

def test_parse(text):
    extracted = r._extract_json(text)
    parsed = json.loads(extracted)
    return r._validate_schema(parsed)

# Test 1: plain JSON
json1 = '{"recommendations": [{"opportunity_id": "FT-100", "score": 90, "reason": "test", "matching_areas": ["Artificial Intelligence", "Other"]}]}'
result1 = test_parse(json1)
assert len(result1["recommendations"]) == 1
assert result1["recommendations"][0]["opportunity_id"] == "FT-100"
print("Test 1 PASS:", result1)

# Test 2: code fence
json2 = '```json\n{"recommendations": [{"opportunity_id": "FT-200", "score": 80, "reason": "test2", "matching_areas": ["Robotics"]}]}\n```'
result2 = test_parse(json2)
assert result2["recommendations"][0]["opportunity_id"] == "FT-200"
print("Test 2 PASS:", result2)

# Test 3: noise around JSON
json3 = 'some noise {"recommendations": [{"opportunity_id": "FT-300", "score": 70, "reason": "test3", "matching_areas": []}]} trailing noise'
result3 = test_parse(json3)
assert result3["recommendations"][0]["opportunity_id"] == "FT-300"
print("Test 3 PASS:", result3)

# Test 4: invalid matching areas filtered out
json4 = '{"recommendations": [{"opportunity_id": "FT-400", "score": 60, "reason": "test4", "matching_areas": ["Invalid Area", "Robotics"]}]}'
result4 = test_parse(json4)
assert "Invalid Area" not in result4["recommendations"][0]["matching_areas"]
assert "Robotics" in result4["recommendations"][0]["matching_areas"]
print("Test 4 PASS:", result4)

# Test 5: sorting by score desc
json5 = '{"recommendations": [{"opportunity_id": "A", "score": 30, "reason": "low", "matching_areas": []}, {"opportunity_id": "B", "score": 90, "reason": "high", "matching_areas": []}]}'
result5 = test_parse(json5)
assert result5["recommendations"][0]["score"] == 90
assert result5["recommendations"][1]["score"] == 30
print("Test 5 PASS:", result5)

print("\nAll tests passed!")
