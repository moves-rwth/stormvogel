"""Contains the code responsible for saving/loading layouts and modifying them interactively."""

from functools import reduce
from typing import Any
from pyvis.network import Network
import os
import json

PACKAGE_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


class Layout:
    layout: dict[str, str]
    # TODO Write unit test

    def __init__(self, path: str | None = None, path_relative: bool = True) -> None:
        """Load a new Layout from a json file.
        Whenever keys are not present in the provided json file, their default values are used instead
        as specified in DEFAULTS (=layouts/default.json).

        Args:
            path (str): Path to your custom layout file.
                Leave to None for an empty/default layout. Defaults to None.
            path_relative (bool, optional): If set to true, then stormvogel will look for a custom layout
                file relative to the current working directory. Defaults to True.
        """
        with open(os.path.join(PACKAGE_ROOT_DIR, "layouts/default.json")) as f:
            default_str = f.read()
        default_dict = json.loads(default_str)

        if path is None:
            self.layout = default_dict
        else:
            if path_relative:
                complete_path = os.path.join(os.getcwd(), path)
            else:
                complete_path = path
            with open(complete_path) as f:
                parsed_str = f.read()
            parsed_dict = json.loads(parsed_str)
            # Combine the parsed dict with default to fill missing keys as default values.
            self.layout = Layout.merge_dict(default_dict, parsed_dict)

    def set_nt_layout(self, nt: Network) -> None:
        """Set the layout of the network passed as the arugment."""
        options = "var options = " + str(self)
        nt.set_options(options)

    def save(self, path: str, path_relative: bool = True) -> None:
        """Save this layout as a json file.

        Args:
            path (str): Path to your layout file.
            path_relative (bool, optional): If set to true, then stormvogel will create a custom layout
                file relative to the current working directory. Defaults to True.
        """
        if path_relative:
            complete_path = os.path.join(os.getcwd(), path)
        else:
            complete_path = path
        with open(complete_path, "w") as f:
            json.dump(self.layout, f)

    def show_buttons(self) -> None:
        raise NotImplementedError()

    @staticmethod
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
                    Layout.merge_dict(dict1[key], dict2[key])
            else:
                if key in dict2:
                    dict1[key] = dict2[key]

        for key, val in dict2.items():
            if key not in dict1:
                dict1[key] = val

        return dict1

    def rget(self, *keys) -> Any:
        """Recursively get an entry from the layout.
        If a key is not present, KeyError will be thrown.
        This should never happen to users because the default values will be used in the case of missing entries."""
        return reduce(
            lambda c, k: c.__getitem__(k), list(keys), self.layout
        )  # Throws KeyError if key not present.

    def __str__(self) -> str:
        return json.dumps(self.layout)


# Define template layouts.
DEFAULT = Layout(
    os.path.join(PACKAGE_ROOT_DIR, "layouts/default.json"), path_relative=False
)
