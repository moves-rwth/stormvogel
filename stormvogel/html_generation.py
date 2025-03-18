"""Javascript code generation functions, used in visjs.py."""


def generate_init_js(nodes_js: str, edges_js: str, options_js: str, name: str) -> str:
    return f"""//js
    var nodes_local = new vis.DataSet([{nodes_js}]);
    var edges_local = new vis.DataSet([{edges_js}]);
    var options_local = {options_js};
    var container_local = document.getElementById("{name}");
    var nw_{name} = new NetworkWrapper_{name}(nodes_local, edges_local, options_local, container_local)
    """


# An html template on which a Network is based.
def generate_html(
    nodes_js: str, edges_js: str, options_js: str, name: str, width: int, height: int
):
    """Generate HTML that renders the network.
    You should be able to locate the NetworkWrapper object as nw_{name},
    and nw_{name} has a field that is the visjs network itself."""
    # Note that double brackets {{ }} are used to escape characters '{' and '}'
    return f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>Network</title>
    <script
      type="text/javascript"
      src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"
    ></script>
    <style type="text/css">
      #{name} {{
        width: {width}px;
        height: {height}px;
        border: 1px solid lightgray;
      }}
    </style>
  </head>
  <body>
    <div id="{name}"></div>
    <script type="text/javascript">
        {generate_network_wrapper_js(name)}
    </script>
    <script type="text/javascript">
        {generate_init_js(nodes_js, edges_js, options_js, name)}
    </script>
  </body>
</html>
"""


def generate_network_wrapper_js(name: str):
    # Javascript code for finding the container and initializing the network
    # Having a separate NewtorkWrapper object allows us to have multiple networks in one notebook without them interfering.
    return (
        f"""
class NetworkWrapper_{name}"""
        + """{//js
  constructor(nodes, edges, options, container) {
    this.nodes = nodes;
    this.edges = edges;
    this.options = options;
    this.container = container;
    this.data = {
      nodes: nodes,
      edges: edges,
    };
    this.network = new vis.Network(container, this.data, options);
    var this_ = this; // Events will not work if you use 'this' directly :))))) No idea why.

    // Set user-triggered events.
    this.network.on( 'click', function(properties) {
      var nodeId = this_.network.getNodeAt({x:properties.event.srcEvent.offsetX, y:properties.event.srcEvent.offsetY});
      this_.makeNeighborsVisible(nodeId);
    });
    this.network.on( 'doubleClick', function(properties) {
        this_.network.setData(this_.data);
    });
  }

  setNodeColor(id, color) {
    var node = this.nodes.get(id);
    node["x"] = this.network.getPosition(id)["x"];
    node["y"] = this.network.getPosition(id)["y"];
    node["color"] = color;
    this.nodes.update(node);
  }

  makeNeighborsVisible(homeId) {
    if (homeId === undefined) {
      return;
    }
    var homeNode = this.nodes.get(homeId);

    // Make outgoing nodes visible
    var nodeIds = this.network.getConnectedNodes(homeId, "to");
    for (let i = 0; i < nodeIds.length; i++) {
      var toNodeId = nodeIds[i];
      var toNode = this.nodes.get(toNodeId);
      if (toNode["hidden"]) {
        toNode["hidden"] = false;
        toNode["physics"] = true;
        toNode["x"] = this.network.getPosition(homeId)["x"];
        toNode["y"] = this.network.getPosition(homeId)["y"];
        this.nodes.update(toNode);
      }
    }
    // Make edges visible, if both of the nodes are also visible
    var edgeIds = this.network.getConnectedEdges(homeId);
    for (let i = 0; i < edgeIds.length; i++) {
      var edgeId = edgeIds[i];
      var edge = this.edges.get(edgeId);
      var fromNode = this.nodes.get(edge.from);
      var toNode = this.nodes.get(edge.to);
      if ((! fromNode["hidden"]) && (! toNode["hidden"])) {
        edge["hidden"] = false;
        edge["physics"] = true;
        this.edges.update(edge);
      }
    }
  }
};
"""
    )
