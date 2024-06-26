"""Contains stuff for visualization"""
from pyvis.network import Network
from stormvogel.model import Model, EmptyAction
from ipywidgets import interact
from IPython.display import display
from fractions import Fraction
##
class Visualization:
    """Handles visualization of a Model using a pyvis Network."""
    def __init__(self, model: Model, name: str, notebook: bool=True, cdn_resources: str="remote") -> None:
        """Create visualization of a Model using a pyvis Network

        Args:
            model (Model): The stormvogel model to be displayed.
            name (str): The name of the resulting html file.
            notebook (bool, optional): Leave to true if you are using in a notebook. Defaults to True.
        """
        self.model = model
        if name[-5:] != ".html": # We do not require the user to explicitly type .html in their names.
            name += ".html"
        self.name = name
        self.g = Network(notebook=notebook, directed=True, cdn_resources=cdn_resources)
        self.__add_nodes()
        self.__add_transitions()
        self.__set_layout()
    
    def __set_layout(self):
        self.g.set_options("""
var options = {
    "nodes": {
        "color": {
            "background": "white",
            "border": "black"
        }
    },
    "edge": {
        "color": "blue"
    }
}""")
    
    def __add_nodes(self):
        """For each state in the model, add a node to the graph."""
        for state in self.model.states.values():
            borderWidth = 1
            if state == self.model.get_initial_state():
                borderWidth = 3
            self.g.add_node(state.id, label=",".join(state.labels), color=None, borderWidth=borderWidth)
    
    def __formatted_prob(self, prob: float) -> str:
        """Take a probability value and format it nicely"""
        return str(Fraction(prob).limit_denominator(20))
    
    def __add_transitions(self):
        """For each transition in the model, add a transition in the graph."""
        for state_id, transition in self.model.transitions.items():
            for action, branch in transition.transition.items():
                if action == EmptyAction:
                    # Only draw probabilities
                    for prob, target in branch.branch:
                        self.g.add_edge(state_id, target.id, color="red", label=self.__formatted_prob(prob))
                else:
                    raise NotImplementedError("Non-empty actions are not supported yet.")

    def show(self):
        """Show the constructed model"""
        display(self.g.show(name=self.name))

def show(model: Model, name: str, notebook: bool=True):
    """Create visualization of a Model using a pyvis Network

    Args:
        model (Model): The stormvogel model to be displayed.
        name (str): The name of the resulting html file.
        notebook (bool, optional): Leave to true if you are using in a notebook. Defaults to True.
    """
    vis = Visualization(model, name, notebook)
    vis.show()

def make_slider():
    return interact(lambda x: x, x=10)
