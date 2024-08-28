"""Our own Python bindings to the vis.js library in JavaScript."""

from IPython.display import display, HTML, DisplayHandle
from html import escape
import random

import stormvogel.html_templates as ht


class Network:
    def __init__(self, name: str, width: int = 800, height: int = 600) -> None:
        """Create a Network.

        Args:
            name (str): Used to name the iframe. You should never create two networks with the same name, they might clash."""

        self.name: str = name
        self.width: int = width
        self.height: int = height
        self.nodes_js: str = ""
        self.edges_js: str = ""
        self.options_js: str = "var options = {}"
        self.handle: DisplayHandle | None = None

    def add_node(
        self,
        id: int,
        label: str = None,  # type: ignore
        group: str = None,  # type: ignore
        color=None,  # type: ignore
        shape=None,  # type: ignore
    ) -> None:
        """Add a node. Only use before calling show."""

        current = "{ id: " + str(id)
        if label is not None:
            current += f", label: `{label}`"
        if group is not None:
            current += f', group: "{group}"'
        current += " },\n"
        self.nodes_js += current

    def add_edge(
        self,
        from_: int,
        to: int,
        label: str = None,  # type: ignore
        color=None,  # type: ignore
        shape=None,  # type: ignore
    ) -> None:
        """Add an edge. Only use before calling show."""
        current = "{ from: " + str(from_) + ", to: " + str(to)
        if label is not None:
            current += f', label: "{label}"'
        current += " },\n"
        self.edges_js += current
        pass

    def set_options(self, options: str):
        """Set the options. The string does NOT have to start with 'var options = '. Only use before calling show."""
        self.options_js = "var options = " + options.replace("var options =", "") + ";"

    def generate_html(self) -> str:
        """Generate the html for the network."""
        js = (
            f"""
        var nodes = new vis.DataSet([{self.nodes_js}]);
        var edges = new vis.DataSet([{self.edges_js}]);
        {self.options_js}
        """
            + ht.CONTAINER_JS
        )
        html = ht.start_html(width=self.width, height=self.height).replace(
            "__JAVASCRIPT__", js
        )
        return html

    def generate_iframe(self) -> str:
        """Generate an iframe for the network, using the html."""
        return f"""
          <iframe
                id="{self.name}"
                width="{self.width}"
                height="{self.height}"
                frameborder="0"
                srcdoc="{escape(self.generate_html())}"
                border:none !important;
                allowfullscreen webkitallowfullscreen mozallowfullscreen
          ></iframe>"""

    def show(self) -> None:
        """Generate the iframe and show it using iPython HTML."""
        iframe = self.generate_iframe()
        # A random display id should avoid collisions in most cases.
        self.display_id = random.randrange(0, 10**31)
        self.handle = display(
            HTML(iframe),
            display_id=self.display_id,
            width=self.width,
            height=self.height,
        )

    def reload(self) -> None:
        """Tries to reload an existing visualization (so it uses a modified layout). If show was not called before, nothing happens."""
        if self.handle is not None:
            iframe = self.generate_iframe()
            self.handle.update(HTML(iframe))

    def update_options(self, options: str):
        """Update the options. The string DOES NOT WORK if it starts with 'var options = '"""
        self.set_options(options)
        html = f"""<script>document.getElementById('{self.name}').contentWindow.network.setOptions({options});</script>"""
        display(HTML(html))
