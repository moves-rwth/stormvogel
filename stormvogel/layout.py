"""Contains the code responsible for saving/loading layouts and modifying them interactively."""

import copy
from pyvis.network import Network
import os
import json
from stormvogel.buttons import (
    ApplyButton,
    SaveButton,
)
from stormvogel.dict_editor import Editor
from stormvogel.rdict import merge_dict, rget

PACKAGE_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


class Layout:
    """Responsible for loading/saving layout jsons."""

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
            self.layout = merge_dict(default_dict, parsed_dict)
        self.vis = None

    def __update_dynamic_labels(self, labels: list[str]):
        """Add the following labels to the layout."""
        for label in labels:
            states_group_copy = copy.deepcopy(self.layout["groups"]["states"])
            if label not in self.layout["groups"]:
                states_group_copy["color"]["background"] = "white"
                states_group_copy["enabled"] = False
            else:
                states_group_copy["color"]["background"] = self.layout["groups"][label][
                    "color"
                ]["background"]
                states_group_copy["enabled"] = self.layout["groups"][label]["enabled"]
            self.layout["groups"][label] = states_group_copy

    def __add_labels_to_schema(self, schema: dict, labels: list[str]):
        """Dynamically add options to change color for each label to gui."""
        if len(labels) > 0:
            schema["groups"]["__label_title"] = {"__html": "<h3>Labels</h3>"}
        for label in labels:
            if label not in schema["groups"]:
                schema["groups"][label] = {
                    "__html": f"<h4>{label}</h4>",
                    "__use_macro": "__label_color_macro",
                }

    def show_editor(self, vis=None):
        """Display an interactive layout editor, according to the schema."""
        with open(os.path.join(PACKAGE_ROOT_DIR, "layouts/schema.json")) as f:
            schema_str = f.read()
        schema = json.loads(schema_str)

        def try_update():
            """Try to update unless impossible (happens if show was not called yet)."""
            if hasattr(vis, "update"):
                self.__update_dynamic_labels(vis.get_labels())  # type: ignore
                vis.update()  # type: ignore

        def maybe_update():
            """Try to update if autoApply is enabled."""
            if rget(self.layout, ["autoApply"]):
                try_update()

        if hasattr(vis, "get_labels"):
            self.__add_labels_to_schema(schema, vis.get_labels())  # type: ignore

        Editor(schema=schema, update_dict=self.layout, on_update=maybe_update)
        SaveButton(self)
        ApplyButton(self, try_update)

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
            json.dump(self.layout, f, indent=2)

    def __str__(self) -> str:
        return json.dumps(self.layout, indent=2)


# Define template layouts.
def DEFAULT():
    return Layout(
        os.path.join(PACKAGE_ROOT_DIR, "layouts/default.json"), path_relative=False
    )


def RAINBOW():
    return Layout(
        os.path.join(PACKAGE_ROOT_DIR, "layouts/rainbow.json"), path_relative=False
    )
