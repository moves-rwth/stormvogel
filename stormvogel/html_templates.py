"""Some constants used in visjs.py"""

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
        width: 600px;
        height: 400px;
        border: 1px solid lightgray;
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

CONTAINER_JS = """
var container = document.getElementById("mynetwork");
var data = {
    nodes: nodes,
    edges: edges,
};
var network = new vis.Network(container, data, options);"""
