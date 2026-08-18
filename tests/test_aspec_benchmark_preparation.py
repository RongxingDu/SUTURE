import json

from experiments.scripts.prepare_aspec_benchmarks import _minimal_gpqa


def test_minimal_gpqa_drops_explanations_and_validator_metadata():
    row = {
        "Record ID": "id",
        "Question": "question",
        "Correct Answer": "correct",
        "Incorrect Answer 1": "wrong 1",
        "Incorrect Answer 2": "wrong 2",
        "Incorrect Answer 3": "wrong 3",
        "High-level domain": "Physics",
        "Subdomain": "Mechanics",
        "Explanation": "private rationale",
        "Canary String": "canary",
        "Writer's Email": "private@example.com",
    }
    minimal = _minimal_gpqa(row, "train", 3)
    assert minimal["Question"] == "question"
    assert minimal["_source_split"] == "train"
    serialized = json.dumps(minimal)
    assert "private rationale" not in serialized
    assert "canary" not in serialized
    assert "private@example.com" not in serialized
