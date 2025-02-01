"""Our own Python bindings to the vis.js library in JavaScript."""

import IPython.display as ipd
import ipywidgets as widgets
import html
import stormvogel.displayable
import stormvogel.html_templates
import stormvogel.communication_server
import json
import random
import string
import logging
from dataclasses import dataclass


spam: widgets.Output = widgets.Output()


@dataclass
class Node:
    id: int
    label: str | None
    group: str | None
    visible: bool


@dataclass(frozen=True)
class Edge:
    from_: int
    to_: int
    label: str | None


class Network(stormvogel.displayable.Displayable):
    EXTRA_PIXELS: int = 20  # To prevent the scroll bar around the Network.

    def __init__(
        self,
        name: str | None = None,
        width: int = 800,
        height: int = 600,
        output: widgets.Output | None = None,
        do_display: bool = True,
        debug_output: widgets.Output = widgets.Output(),
        do_init_server: bool = True,
        positions: dict[str, dict[str, int]] | None = None,
    ) -> None:
        """Display a visjs network using IPython. The network can display by itself or you can specify an Output widget in which it should be displayed.

        Args:
            name (str): Used to name the iframe. ONLY SPECIFY IF YOU KNOW WHAT YOU ARE DOING. You should never create two networks with the same name, they might clash.
            width (int): Width of the network, in pixels.
            height (int): Height of the network, in pixels.
            output (widgets.Output): An output widget within which the network should be displayed.
            do_display (bool): Set to true iff you want the Network to display. Defaults to True.
            debug_output (widgets.Output): Debug information is displayed in this output. Leave to default if that doesn't interest you."""
        super().__init__(output, do_display, debug_output)
        if name is None:
            self.name: str = "".join(random.choices(string.ascii_letters, k=10))
        else:
            self.name: str = name
        self.content_window = f"document.getElementById('{self.name}').contentWindow"
        self.width: int = width
        self.height: int = height
        self.nodes_js: str = ""
        self.edges_js: str = ""
        self.options_js: str = "{}"
        if do_init_server:
            self.server: stormvogel.communication_server.CommunicationServer = (
                stormvogel.communication_server.initialize_server()
            )
        self.positions: dict[str, dict[str, int]]
        if positions is None:
            self.positions = {}
        else:
            self.positions = positions
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, set[Edge]] = {}
        self.reverse_edges: dict[int, set[Edge]] = {}

    def enable_exploration_mode(self):
        """Inject the JS that enables exploration by clicking."""
        js = f"""//js
{self.content_window}.network.on( 'click', function(properties) {{
    var nodeId = {self.content_window}.network.getNodeAt({{x:properties.event.srcEvent.offsetX, y:properties.event.srcEvent.offsetY}});
    if (! (nodeId === undefined)) {{
        FUNCTION(nodeId);
    }}
}});"""
        self.server.add_event(js, self.make_neighbors_visible)

    def make_neighbors_visible(self, id: str) -> None:
        """Make the neighbors of this node visible."""
        print("explore", id)
        for edge in self.edges[int(id)]:
            print(edge.to_)
            self.make_node_visible(edge.to_, True)

    def make_node_visible(self, id: int, two: bool = False) -> None:
        """Make the node with this id visible, i.e. adding it to the JS network.
        Add incoming edges if the source node is visible and outgoing edges if the target node is visible."""
        print("make visible", id, type(id))
        node = self.nodes[id]
        node.visible = True
        js = f"""alert('{id}');"""
        if two:
            js = f"""{self.content_window}.nodes.update({self.__create_node_code(node)});"""
        print(js)
        ipd.display(ipd.Javascript(js))

    def get_positions(self) -> dict:
        """Get the current positions of the nodes on the canvas. Returns empty dict if unsuccessful.
        Example result: {"0": {"x": 5, "y": 10}}"""
        if self.server is None:
            with self.debug_output:
                logging.warning(
                    "Server not initialized. Could not retrieve position data."
                )
            raise TimeoutError("Server not initialized.")
        try:
            positions: dict = json.loads(
                self.server.result(
                    f"""RETURN({self.content_window}.network.getPositions())"""
                )
            )
            return positions
        except TimeoutError:
            with self.debug_output:
                logging.warning("Timed out. Could not retrieve position data.")
            raise TimeoutError("Timed out. Could not retrieve position data.")

    def __create_node_code(self, node: Node) -> str:
        """Generate the code to add the node to the pre-creation Javascript."""
        current = "{ id: " + str(node.id)
        if node.label is not None:
            current += f", label: `{node.label}`"
        if node.group is not None:
            current += f', group: "{node.group}"'
        if self.positions is not None and str(node.id) in self.positions:
            current += f', x: {self.positions[str(node.id)]["x"]}, y: {self.positions[str(node.id)]["y"]}'
        current += " }"
        return current

    def add_node(
        self,
        id: int,
        label: str | None = None,
        group: str | None = None,
        visible: bool = True,
    ) -> None:
        """Add a node. Only use before calling show."""
        node = Node(id, label, group, visible)
        self.nodes[id] = node
        if visible:
            self.nodes_js += self.__create_node_code(node) + ",\n"

    def __create_edge_code(self, edge: Edge) -> str:
        """Add the edge to the pre-creation Javascript."""
        current = "{ from: " + str(edge.from_) + ", to: " + str(edge.to_)
        if edge.label is not None:
            current += f', label: "{edge.label}"'
        current += " },\n"
        return current

    def add_edge(
        self,
        from_: int,
        to_: int,
        label: str | None = None,
    ) -> None:
        """Add an edge. Only use before calling show."""
        edge = Edge(from_, to_, label)
        if from_ in self.edges:
            self.edges[from_].add(edge)
        else:
            self.edges[from_] = {edge}
        if to_ in self.reverse_edges:
            self.reverse_edges[to_].add(edge)
        else:
            self.reverse_edges[to_] = {edge}

        if self.nodes[from_].visible and self.nodes[from_].visible:
            self.edges_js += self.__create_edge_code(edge)

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
            ipd.clear_output()
            ipd.display(widgets.HTML(iframe))
        self.maybe_display_output()
        with self.debug_output:
            logging.info("Called Network.show")

    def reload(self) -> None:
        """Tries to reload an existing visualization (so it uses a modified layout). If show was not called before, nothing happens."""
        iframe = self.generate_iframe()
        with self.output:
            ipd.clear_output()
            ipd.display(widgets.HTML(iframe))
        with self.debug_output:
            logging.info("Called Network.reload")

    def update_options(self, options: str):
        """Update the options. The string DOES NOT WORK if it starts with 'var options = '"""
        self.set_options(options)
        js = f"""{self.content_window}.network.setOptions({options});"""
        with self.spam:
            ipd.display(ipd.Javascript(js))
        self.spam_side_effects()

        with self.debug_output:
            logging.info("The previous javascript error is no problem in most cases.")
        with self.debug_output:
            logging.info("Called Network.update_options")

    def clear(self) -> None:
        """Clear the output."""
        with self.output:
            ipd.clear_output()
