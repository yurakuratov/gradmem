from analyze_gradmem_mix_failures import error_category, make_shared_first_value_row, parse_pairs


def test_error_category_is_exhaustive():
    assert error_category(True, True) == "exact"
    assert error_category(True, False) == "first_correct_second_wrong"
    assert error_category(False, True) == "first_wrong_second_correct"
    assert error_category(False, False) == "both_wrong"


def test_shared_first_values_preserve_forward_keys_and_target():
    row = {
        "context": "F!aa:x1!!bb:y2!!cc:z3!|",
        "query": "?!bb:",
        "target": "y2!|",
    }
    transformed = make_shared_first_value_row(row)
    direction, pairs = parse_pairs(transformed)
    assert direction == "F"
    assert [key for key, _ in pairs] == ["aa", "bb", "cc"]
    assert dict(pairs)["bb"] == "y2"
    assert {value[0] for _, value in pairs} == {"y"}
    assert len({value[1] for _, value in pairs}) == 3
    assert transformed["query"] == row["query"]
    assert transformed["target"] == row["target"]


def test_shared_first_values_preserve_backward_keys_and_target():
    row = {
        "context": "B!x1:aa!!y2:bb!!z3:cc!|",
        "query": "?!cc:",
        "target": "z3!|",
    }
    transformed = make_shared_first_value_row(row)
    direction, pairs = parse_pairs(transformed)
    assert direction == "B"
    assert [key for key, _ in pairs] == ["aa", "bb", "cc"]
    assert dict(pairs)["cc"] == "z3"
    assert {value[0] for _, value in pairs} == {"z"}
    assert len({value[1] for _, value in pairs}) == 3
