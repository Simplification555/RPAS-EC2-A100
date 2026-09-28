from aime_quality_gate import check_ordinal_scores


def test_aime_quality_gate_accepts_uniform_ordinal_audit():
    metrics = {
        "rows": [
            {"correct": True, "parser_valid": True},
            {"correct": False, "parser_valid": True},
            {"correct": False, "parser_valid": False},
        ],
        "score_0_1_2": [2, 1, 0],
        "mean_score_0_1_2": 1.0,
    }
    assert check_ordinal_scores(metrics, "fixture", expected_count=3) == []


def test_aime_quality_gate_rejects_inconsistent_ordinal_audit():
    metrics = {
        "rows": [
            {"correct": True, "parser_valid": True},
            {"correct": False, "parser_valid": False},
        ],
        "score_0_1_2": [1, 0],
        "mean_score_0_1_2": 1.0,
    }
    errors = check_ordinal_scores(metrics, "fixture", expected_count=2)
    assert errors == ["fixture per-example scores violate the shared 0/1/2 mapping"]
