"""Some constants used in visjs.py"""

# An html template on which a Network is based.
START_HTML = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>Network</title>
    <script
      type="text/javascript"
      src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"
    ></script>
    <style type="text/css">
      #mynetwork {
        __SIZES__
      }
    </style>
  </head>
  <body>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        __JAVASCRIPT__
    </script>
  </body>
</html>
"""

# Javascript code for finding the container and initializing the network
NETWORK_JS = """//js
var container = document.getElementById("mynetwork");
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
    //var ids = network.getConnectedNodes(myNode);
    homeNode = nodes.get(homeId);
    var edgeIds = network.getConnectedEdges(homeId, 'to');
    for (let i = 0; i < edgeIds.length; i++) {
        var edgeId = edgeIds[i];
        var edge = edges.get(edgeId);
        var nodeId = edge.to;
        var node = nodes.get(nodeId);

        if (node["hidden"]) {
          node["hidden"] = false;
          node["physics"] = true;
          // node["x"] = network.getPosition(homeId)["x"];
          // node["y"] = network.getPosition(homeId)["y"];
          nodes.update(node);
          // node["x"] = undefined;
          // node["y"] = undefined;
          // nodes.update(node);
        }
        edge["hidden"] = false;
        edge["physics"] = true;
        edges.update(edge);
    }
};
function makeNodeVisible(nodeId) {
    var node = nodes.get(nodeId);
    node["hidden"] = false;
    nodes.update(node);
};
network.on( 'click', function(properties) {
    var nodeId = network.getNodeAt({x:properties.event.srcEvent.offsetX, y:properties.event.srcEvent.offsetY});
    makeNeighborsVisible(nodeId);
});
"""
