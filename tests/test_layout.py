from stormvogel.layout import Layout
import os
import json


def test_layout_merge_dict():
    # Test priority for second dict.
    d1 = {"a": 1, "b": 1, "c": 1}
    d2 = {"b": 2}
    assert {"a": 1, "b": 2, "c": 1} == Layout.merge_dict(d1, d2)
    # Test conservation of elements in both dicts
    d1 = {"a": 1, "b": 1}
    d2 = {"c": 2, "d": 2}
    assert {"a": 1, "b": 1, "c": 2, "d": 2} == Layout.merge_dict(d1, d2)
    # Test nested
    d1 = {"a": {"b": {"c": 1, "d": 1}}, "e": 1}
    d2 = {"a": {"b": {"c": 2}}}
    assert {"a": {"b": {"c": 2, "d": 1}}, "e": 1} == Layout.merge_dict(d1, d2)


def test_layout_loading():
    """Tests if str(Layout) returns the correctly loaded json string."""
    with open(os.path.join(os.getcwd(), "stormvogel/layouts/default.json")) as f:
        default_str = f.read()
    with open(os.path.join(os.getcwd(), "tests/test_layout.json")) as f:
        test_str = f.read()
    default_dict = json.loads(default_str)
    test_dict = json.loads(test_str)
    expected = json.dumps(
        Layout.merge_dict(default_dict, test_dict)
    )  # We can use Layout.merge_dict since we have already tested it.
    actual = str(Layout("tests/test_layout.json"))

    assert expected == actual


def test_layout_saving():
    """Tests if the saved layout from Layout.save() is equal to str(Layout)."""
    layout = Layout("tests/test_layout.json")
    try:
        os.remove(
            os.path.join(os.getcwd(), "tests/saved_test_layout.json")
        )  # First remove the existing file.
    except FileNotFoundError:
        pass  # The file did not exist yet.
    layout.save("tests/saved_test_layout.json")
    with open(os.path.join(os.getcwd(), "tests/saved_test_layout.json")) as f:
        saved_string = f.read()
    assert saved_string == str(layout)


def test_layout_rget():
    layout = Layout("tests/test_layout.json")
    # Overwritten key
    assert layout.rget("init", "color") == "TEST_COLOR"
    # Default key and number
    assert layout.rget("init", "borderWidth") == 3
    # None/null
    assert layout.rget("states", "color") is None
    # bool
    assert layout.rget("rounding", "fractions")
