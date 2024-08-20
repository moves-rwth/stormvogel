"""Updated visualization with our own JavaScript generation."""

from time import sleep
from stormvogel.layout import Layout
from stormvogel.model import Model
from IPython.display import display, HTML
from html import escape


class Visualization:
    ACTION_ID_OFFSET: int = 10**10
    # In the visualization, both actions and states are nodes with an id.
    # This offset is used to keep their ids from colliding. It should be some high constant.

    def __init__(self, model: Model) -> None:
        self.model = model
        self.html = """
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
"""  # This includes the visjs networks library and has most of the html info.

    def __states(self):
        """Generate js code for adding states."""
        states_js = "var nodes = new vis.DataSet([\n"

        for state in self.model.states.values():
            states_js += (
                "{ id: "
                + str(state.id)
                + ', label: "{'
                + ",".join(state.labels)
                + '}"'
                + ', group: "states"'
                + " },\n"
            )
        states_js += "]);\n"
        print(states_js)
        return states_js

    def __temp(self):
        return """
network.on( 'click', function(properties) {
        var nodeId = network.getNodeAt({x:properties.event.srcEvent.offsetX, y:properties.event.srcEvent.offsetY});
        var update_array = [{id: nodeId, hidden: true}]
        nodes.update(update_array)
      });

function myFunction() {
          var new_options = {"nodes": {"color": {"background": "red"}}}
          var update_array = [{id: 1, hidden: false}]
          nodes.update(update_array)
          network.setOptions(new_options)
        }"""

    def set_options(self, options: str | Layout):
        sleep(0.5)
        if isinstance(options, Layout):
            options = str(options)
        html = f"""<script>document.getElementById('targetFrame').contentWindow.network.setOptions({options});</script>"""
        display(HTML(html))

    def show(self):
        js = (
            self.__states()
            + """
      // create an array with edges
      var edges = new vis.DataSet([]);

      // create a network
      var container = document.getElementById("mynetwork");
      var data = {
        nodes: nodes,
        edges: edges,
      };
      var options = {};
      var network = new vis.Network(container, data, options);"""
        )
        iframe = f"""
          <iframe
              id="targetFrame"
              width="650"
              height="450"
              frameborder="0"
              srcdoc="{escape(self.html.replace("__JAVASCRIPT__", js))}"
              border:none !important;
              allowfullscreen webkitallowfullscreen mozallowfullscreen
          ></iframe>"""
        print(iframe)
        display(HTML(iframe))
