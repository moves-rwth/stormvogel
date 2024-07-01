"""Contains the code responsible for saving/loading layouts and modifying them interactively."""

from pyvis.network import Network
import os
import json

DEFAULT = "layouts/default.json"


class Layout:
    layout: dict

    def __init__(
        self,
        custom: bool,
        path: str | None = None,
        path_relative: bool = True,
        template_path: str = DEFAULT,
    ) -> None:
        """Load a new Layout from a json file. Use either a custom or a template file.

        Args:
            custom (bool, optional): If set to true, stormvogel will look for your custom layout.json file. Otherwise a template will be used.
            path (str, optional): Relavant if custom is true. Path to your custom layout file, relative to the current working directory. Defaults to None.
            path_relative (bool): Relavant if custom is true. If set to true, then stormvogel will look for a custom layout file relative to the current working directory.
            template_path (str, optional): Relavant if custom is false. Path to a template layout files.
                These are stored in the folder layouts. For simplicity, we recommed using the constants DEFAULT, etc.
                Defaults to DEFAULT (="layouts/default.json").
        """
        if custom:
            if path is None:
                raise Exception(
                    "If custom is set to true, then the path needs to be set."
                )
            cwd = os.getcwd()
            if path_relative:
                complete_path = os.path.join(cwd, path)
            else:
                complete_path = path
            with open(complete_path) as f:
                json_string = f.read()
                self.layout = json.loads(json_string)
        else:
            package_root_dir = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(package_root_dir, template_path)) as f:
                json_string = f.read()
                self.layout = json.loads(json_string)

    def set_nt_layout(self, nt: Network) -> None:
        """Set the layout of the network passed as the arugment."""
        # We here use <> instead of {} because the f-string formatting already uses them.
        option_string = f"""
var options = <
    "nodes": <
        "color": <
            "background": "{self.layout["color"]}",
            "border": "black"
        >
    >
>""".replace("<", "{").replace(">", "}")
        print(option_string)
        nt.set_options(option_string)

    def save(self) -> None:
        raise NotImplementedError()

    def show_buttons(self) -> None:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()
