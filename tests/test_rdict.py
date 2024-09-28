from stormvogel.rdict import rget, rset, merge_dict


def test_rget():
    # Empty
    d = {}
    assert rget(d, []) == {}
    # Simple
    d = {"a": 1}
    assert rget(d, ["a"]) == 1
    # Nested
    d = {"a": {"b": {"c": 3, "b": 5}, "c": 2}, "c": 1}
    assert rget(d, ["a", "b", "c"]) == 3


def test_rset():
    # Empty path
    d = {}
    assert rset(d, [], 1) == {}
    # Simple
    d = {}
    assert rset(d, ["a"], 1) == {"a": 1}
    # Existing value
    d = {"a": 0}
    assert rset(d, ["a"], 1) == {"a": 1}
    # Nested
    d = {"a": {"b": 8}}
    assert rset(d, ["a", "b"], {"c": 3}) == {"a": {"b": {"c": 3}}}


def test_merge_dict():
    # Test priority for second dict.
    d1 = {"a": 1, "b": 1, "c": 1}
    d2 = {"b": 2}
    assert {"a": 1, "b": 2, "c": 1} == merge_dict(d1, d2)
    # Test conservation of elements in both dicts
    d1 = {"a": 1, "b": 1}
    d2 = {"c": 2, "d": 2}
    assert {"a": 1, "b": 1, "c": 2, "d": 2} == merge_dict(d1, d2)
    # Test nested
    d1 = {"a": {"b": {"c": 1, "d": 1}}, "e": 1}
    d2 = {"a": {"b": {"c": 2}}}
    assert {"a": {"b": {"c": 2, "d": 1}}, "e": 1} == merge_dict(d1, d2)
