"""Contains the code responsible for model visualization."""

from pyvis.network import Network
from stormvogel.model import Model, EmptyAction, Number
from stormvogel.layout import Layout
from ipywidgets import interact
from IPython.display import display
from fractions import Fraction


class Visualization:
    """Handles visualization of a Model using a pyvis Network."""

    name: str
    g: Network
    ACTION_ID_OFFSET = 10**8
    layout: Layout
    # In the visualization, both actions and states are nodes with an id.
    # This offset is used to keep their ids from colliding. It should be some high constant.

    def __init__(
        self,
        model: Model,
        name: str = "model",
        notebook: bool = True,
        cdn_resources: str = "remote",
        layout: Layout | None = None,
    ) -> None:
        """Create visualization of a Model using a pyvis Network

        Args:
            model (Model): The stormvogel model to be displayed.
            name (str, optional): The name of the resulting html file. May or may not include .html extension.
            notebook (bool, optional): Leave to true if you are using in a notebook. Defaults to True.
        """
        self.model = model
        if (
            name[-5:] != ".html"
        ):  # We do not require the user to explicitly type .html in their names
            name += ".html"
        self.name = name
        self.g = Network(notebook=notebook, directed=True, cdn_resources=cdn_resources)
        self.__add_states()
        self.__add_transitions()
        if layout is None:
            self.layout = Layout(custom=False)
        else:
            self.layout = layout

        self.layout.set_nt_layout(self.g)

    def __add_states(self):
        """For each state in the model, add a node to the graph."""
        for state in self.model.states.values():
            borderWidth = 1
            if state == self.model.get_initial_state():
                borderWidth = 3
            self.g.add_node(
                state.id,
                label=",".join(state.labels),
                color=None,  # type: ignore
                borderWidth=borderWidth,
                shape="dot",
            )

    def __formatted_probability(self, prob: Number) -> str:
        """Take a probability value and format it nicely using a fraction."""
        return str(Fraction(prob).limit_denominator(20))

    def __add_transitions(self):
        """For each transition in the model, add a transition in the graph.
        Also handles actions by calling __add_action"""
        action_id = self.ACTION_ID_OFFSET
        # In the visualization, both actions and states are nodes, so we need to keep track of how many actions we already have.
        for state_id, transition in self.model.transitions.items():
            for action, branch in transition.transition.items():
                if action == EmptyAction:
                    # Only draw probabilities
                    for prob, target in branch.branch:
                        self.g.add_edge(
                            state_id,
                            target.id,
                            color="red",
                            label=self.__formatted_probability(prob),
                        )
                else:
                    # Add the action's node
                    self.g.add_node(
                        n_id=action_id,
                        color=None,  # type: ignore
                        label=action.name,
                        shape="box",
                    )
                    # Add transition from this state TO the action.
                    self.g.add_edge(state_id, action_id, color="red")  # type: ignore
                    # Add transition FROM the action to the values in its branch.
                    for prob, target in branch.branch:
                        self.g.add_edge(
                            action_id,
                            target.id,
                            color="red",
                            label=self.__formatted_probability(prob),
                        )
                    action_id += 1

    def show(self):
        """Show the constructed graph."""
        display(self.g.show(name=self.name))


def show(model: Model, name: str = "model", notebook: bool = True):
    """Create and show a visualization of a Model using a pyvis Network

    Args:
        model (Model): The stormvogel model to be displayed.
        name (str, optional): The name of the resulting html file.
        notebook (bool, optional): Leave to true if you are using in a notebook. Defaults to True.
    """
    vis = Visualization(model, name, notebook)
    vis.show()


def make_slider():
    return interact(lambda x: x, x=10)
