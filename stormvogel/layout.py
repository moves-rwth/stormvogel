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

        # Load in schema for the dict_editor.
        with open(os.path.join(PACKAGE_ROOT_DIR, "layouts/schema.json")) as f:
            schema_str = f.read()
        self.schema = json.loads(schema_str)

    def set_groups(self, groups: list[str]):
        """Add the specified groups to the layout and schema.
        They will use the specified __group_macro.
        Note that the changes to schema won't be saved to schema.json.
        Remove groups that aren't used."""
        for g in groups:
            if g not in self.layout["groups"]:
                layout_group_macro = copy.deepcopy(
                    self.layout["__fake_macros"]["__group_macro"]
                )
                self.layout["groups"][g] = layout_group_macro
            if g not in self.schema["groups"]:
                self.schema["groups"][
                    g
                ] = {  # dict_editor already handles macros, so there is no need to do it manually here.
                    "__use_macro": "__group_macro"
                }

        # Remove unused groups, so that the Misc values can be used.
        # for existing_group in self.layout["groups"].keys():
        #     if existing_group not in groups + ["states", "actions"]:
        #         self.layout["groups"].pop(existing_group)

    def show_editor(self, vis=None):
        """Display an interactive layout editor, according to the schema."""
        done_loading = False

        def try_update():
            """Try to update the visualization unless impossible (happens if show was not called yet),
            or the editor menu is not done loading yet."""
            if hasattr(vis, "update") and done_loading:
                vis.update()  # type: ignore

        def maybe_update():
            """Try to update if autoApply is enabled."""
            if rget(self.layout, ["autoApply"]):
                try_update()

        Editor(schema=self.schema, update_dict=self.layout, on_update=maybe_update)
        SaveButton(self)
        ApplyButton(self, try_update)
        done_loading = True

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
