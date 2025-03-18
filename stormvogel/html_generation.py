"""Javascript code generation functions, used in visjs.py."""


def generate_js(nodes_js: str, edges_js: str, options_js: str, name: str) -> str:
    return f"""//js
    var nodes_local = new vis.DataSet([{nodes_js}]);
    var edges_local = new vis.DataSet([{edges_js}]);
    var options_local = {options_js};
    var container_local = document.getElementById("{name}");
    var nw_{name} = new NetworkWrapper(nodes_local, edges_local, options_local, container_local)
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
        {NETWORK_WRAPPER_JS}
    </script>
    <script type="text/javascript">
        {generate_js(nodes_js, edges_js, options_js, name)}
    </script>
  </body>
</html>
"""


# Javascript code for finding the container and initializing the network
# Having a separate NewtorkWrapper object allows us to have multiple networks in one notebook without them interfering.
NETWORK_WRAPPER_JS = """//js
class NetworkWrapper {
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
    node["color"] = color
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

"""
}


var data = {
    nodes: nodes,
    edges: edges,
};
var network = new vis.Network(container, data, options);
function makeAllNodesInvisible() {
    ids = nodes.getIds();
    for (let i = 0; i < ids.length; i++) {
        var nodeId = ids[i];
        var node = nodes.get(nodeId);
        node["hidden"] = true;
        node["physics"] = false;
        nodes.update(node);
    }
};
function makeNeighborsVisible(homeId) {
    homeNode = nodes.get(homeId);

    // Make outgoing nodes visible
    var nodeIds = network.getConnectedNodes(homeId, "to");
    for (let i = 0; i < nodeIds.length; i++) {
      var toNodeId = nodeIds[i];
      var toNode = nodes.get(toNodeId);
      if (toNode["hidden"]) {
        toNode["hidden"] = false;
        toNode["physics"] = true;
        toNode["x"] = network.getPosition(homeId)["x"];
        toNode["y"] = network.getPosition(homeId)["y"];
        nodes.update(toNode);
      }
    }
    // Make edges visible, if both of the nodes are also visible
    var edgeIds = network.getConnectedEdges(homeId);
    for (let i = 0; i < edgeIds.length; i++) {
        var edgeId = edgeIds[i];
        var edge = edges.get(edgeId);
        var fromNode = nodes.get(edge.from);
        var toNode = nodes.get(edge.to);
        if ((! fromNode["hidden"]) && (! toNode["hidden"])) {
          edge["hidden"] = false;
          edge["physics"] = true;
          edges.update(edge);
        }
    }
};
network.on( 'click', function(properties) {
    var nodeId = network.getNodeAt({x:properties.event.srcEvent.offsetX, y:properties.event.srcEvent.offsetY});
    makeNeighborsVisible(nodeId);
});
network.on( 'doubleClick', function(properties) {
    network.setData(data);
});
function setNodeColor(id, color) {
  var node = nodes.get(id);
  node["color"] = color
  nodes.update(node);
}
"""
