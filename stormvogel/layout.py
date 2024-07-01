"""Contains the code responsible for saving/loading layouts and modifying them interactively."""

from pyvis.network import Network
import os
import json


class Layout:
    layout: dict[str, str]

    def __init__(self, path: str, path_relative: bool = True) -> None:
        """Load a new Layout from a json file. Use either a custom or a template file.

        Args:
            path (str, optional): Relavant if custom is true. Path to your custom layout file, relative to the current working directory. Defaults to None.
            path_relative (bool): Relavant if custom is true. If set to true, then stormvogel will look for a custom layout file relative to the current working directory.
        """
        if path_relative:
            complete_path = os.path.join(os.getcwd(), path)
        else:
            complete_path = path
        with open(complete_path) as f:
            json_string = f.read()
            self.layout = json.loads(json_string)

    def set_nt_layout(self, nt: Network) -> None:
        """Set the layout of the network passed as the arugment."""
        # We here use <> instead of {} because the f-string formatting already uses them.
        #         option_string = f"""
        # var options = <
        #     "nodes": <
        #         "color": <
        #             "background": "{self.layout["color"]}",
        #             "border": "black"
        #         >
        #     >
        # >""".replace("<", "{").replace(">", "}")
        options = "var options = " + str(self.layout).replace("'", '"')
        print(options)
        nt.set_options(options)

    def save(self) -> None:
        raise NotImplementedError()

    def show_buttons(self) -> None:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    def __getitem__(self, key: str) -> str | None:
        try:
            return self.layout[key]
        except KeyError:
            return None


package_root_dir = os.path.dirname(os.path.realpath(__file__))

# Define template layouts.
DEFAULT = Layout(
    os.path.join(package_root_dir, "layouts/default.json"), path_relative=False
)
