"""Two useful functions for handling dicts recursively using paths."""

import copy
from functools import reduce
from typing import Any


def rget(d: dict, path: list) -> Any:
    """Recursively get dict value. Throws KeyError if (nested) key not present"""
    return reduce(
        lambda c, k: c.__getitem__(k), path, d
    )  # Throws KeyError if key not present.


def rset(d: dict, path: list, value: Any) -> None:
    """Recursively set dict value."""
    if len(path) == 0:
        return

    def __rset(d: dict, path: list, value: Any):
        first = path.pop(0)
        if len(path) == 0:
            d[first] = value
        else:
            __rset(d[first], path, value)

    __rset(d, copy.deepcopy(path), value)


def merge_dict(dict1: dict, dict2: dict) -> dict:
    """Merge two nested dictionaries recursively. Note that dict1 is modified by reference and also returned.

    Args:
        dict1 (dict):
        dict2 (dict):

    If dict2 has a value that dict1 does not have, then the value in dict2 is chosen.
    If dict1 has a DICTIONARY and dict2 has a VALUE with the same key, then dict1 gets priority.

    Taken from StackOverflow user Anatoliy R on July 2 2024.
    https://stackoverflow.com/questions/43797333/how-to-merge-two-nested-dict-in-python"""
    for key, val in dict1.items():
        if isinstance(val, dict):
            if key in dict2 and type(dict2[key] == dict):
                merge_dict(dict1[key], dict2[key])
        else:
            if key in dict2:
                dict1[key] = dict2[key]

    for key, val in dict2.items():
        if key not in dict1:
            dict1[key] = val

    return dict1
