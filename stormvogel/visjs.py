"""Our own Python bindings to the vis.js library in JavaScript."""

import IPython.display as ipd
import ipywidgets as widgets
import html
import stormvogel.displayable
import stormvogel.html_templates
import stormvogel.local_server
import random


class Network(stormvogel.displayable.Displayable):
    EXTRA_PIXELS = 20  # To prevent the scroll bar around the Network.

    def __init__(
        self,
        name: str,
        width: int = 800,
        height: int = 600,
        output: widgets.Output = widgets.Output(),
        do_display: bool = True,
        debug_output: widgets.Output = widgets.Output(),
        server_port: int = -1,
    ) -> None:
        """Display a visjs network using IPython. The network can display by itself or you can specify an Output widget in which it should be displayed.

        Args:
            name (str): Used to name the iframe. You should never create two networks with the same name, they might clash.
            width (int): Width of the network, in pixels.
            height (int): Height of the network, in pixels.
            output (widgets.Output): An output widget within which the network should be displayed.
            do_display (bool): Set to true iff you want the Network to display. Defaults to True.
            debug_output (widgets.Output): Debug information is displayed in this output. Leave to default if that doesn't interest you.
            server_port (int): Used for internal communication.
                Never create two networks with the same server port. If left to default, it will be randomly generated."""
        super().__init__(output, do_display, debug_output)
        self.name: str = name
        self.width: int = width
        self.height: int = height
        self.nodes_js: str = ""
        self.edges_js: str = ""
        self.options_js: str = "{}"
        if server_port == -1:
            server_port = random.randrange(8000, 9000)
        self.messenger: stormvogel.local_server.JSMessenger = (
            stormvogel.local_server.JSMessenger(server_port=server_port)
        )

    def get_positions(self):
        """Get the current positions of the nodes on the canvas."""
        return self.messenger.request(
            f"""JSON.stringify(document.getElementById('{self.name}').contentWindow.network.getPositions())"""
        )

    def add_node(
        self,
        id: int,
        label: str | None = None,
        group: str | None = None,
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
        label: str | None = None,
    ) -> None:
        """Add an edge. Only use before calling show."""
        current = "{ from: " + str(from_) + ", to: " + str(to)
        if label is not None:
            current += f', label: "{label}"'
        current += " },\n"
        self.edges_js += current

    def set_options(self, options: str) -> None:
        """Set the options. Only use before calling show."""
        self.options_js = options

    def generate_html(self) -> str:
        """Generate the html for the network."""
        js = (
            f"""
        var nodes = new vis.DataSet([{self.nodes_js}]);
        var edges = new vis.DataSet([{self.edges_js}]);
        var options = {self.options_js};
        """
            + stormvogel.html_templates.NETWORK_JS
        )

        sizes = f"""
        width: {self.width}px;
        height: {self.height}px;
        border: 1px solid lightgray;
        """

        html = stormvogel.html_templates.START_HTML.replace(
            "__JAVASCRIPT__", js
        ).replace("__SIZES__", sizes)
        return html

    def generate_iframe(self) -> str:
        """Generate an iframe for the network, using the html."""
        return f"""
          <iframe
                id="{self.name}"
                width="{self.width + self.EXTRA_PIXELS}"
                height="{self.height + self.EXTRA_PIXELS}"
                frameborder="0"
                srcdoc="{html.escape(self.generate_html())}"
                border:none !important;
                allowfullscreen webkitallowfullscreen mozallowfullscreen
          ></iframe>"""

    def show(self) -> None:
        """Display the network on the output that was specified at initialization, otherwise simply display it."""
        iframe = self.generate_iframe()
        with self.output:  # Display the iframe within the Output.
            ipd.display(ipd.HTML(iframe))
        self.maybe_display_output()
        with self.debug_output:
            print("Called Network.show")

    def reload(self) -> None:
        """Tries to reload an existing visualization (so it uses a modified layout). If show was not called before, nothing happens."""
        iframe = self.generate_iframe()
        with self.output:
            ipd.clear_output()
            ipd.display(ipd.HTML(iframe))
        with self.debug_output:
            print("Called Network.reload")

    def update_options(self, options: str):
        """Update the options. The string DOES NOT WORK if it starts with 'var options = '"""
        self.set_options(options)
        html = f"""<script>document.getElementById('{self.name}').contentWindow.network.setOptions({options});</script>"""
        with self.output:
            ipd.display(ipd.HTML(html))
        with self.debug_output:
            print("Called Network.update_options")

    def clear(self) -> None:
        """Clear the output."""
        with self.output:
            ipd.clear_output()
