"""Contains the code responsible for model visualization."""

import random
from fractions import Fraction
from html import escape

from IPython.display import HTML, IFrame, display
from pyvis.network import Network

from stormvogel.layout import DEFAULT, Layout
from stormvogel.model import EmptyAction, Model, Number, State


class Visualization:
    """Handles visualization of a Model using a pyvis Network."""

    ACTION_ID_OFFSET: int = 10**10
    # In the visualization, both actions and states are nodes with an id.
    # This offset is used to keep their ids from colliding. It should be some high constant.

    def __init__(
        self,
        model: Model,
        name: str = "model",
        notebook: bool = True,
        cdn_resources: str = "remote",
        layout: Layout = DEFAULT,
    ) -> None:
        """Create visualization of a Model using a pyvis Network

        Args:
            model (Model): The stormvogel model to be displayed.
            name (str, optional): The name of the resulting html file. May or may not include .html extension.
            notebook (bool, optional): Leave to true if you are using in a notebook. Defaults to True.
        """
        self.model = model
        self.layout = layout
        if (
            name[-5:] != ".html"
        ):  # We do not require the user to explicitly type .html in their names
            name += ".html"
        self.name = name
        self.notebook = notebook
        self.cdn_resources = cdn_resources

    def __reload_nt(self):
        """(Re)load the pyvis network."""
        self.nt = Network(
            notebook=self.notebook, directed=True, cdn_resources=self.cdn_resources
        )
        self.__add_states()
        self.__add_transitions()
        self.layout.set_nt_layout(self.nt)

    def update(self):
        """Update an existing visualization (so it uses a modified layout)."""
        self.__reload_nt()
        try:
            self.nt.write_html(self.name, open_browser=False, notebook=self.notebook)
            # Not using show here to stop the print message.
            iframe = IFrame(self.name, width=self.nt.width, height=self.nt.height)
            self.handle.update(iframe)  # type: ignore
        except AttributeError:
            raise Exception(
                "show should be called at least once before calling update."
            )

    def show(self):
        """Show or update the constructed graph as a html file."""
        self.__reload_nt()

        html_code = self.nt.generate_html(name=self.name, notebook=True)

        # We build our own iframe because we want to embed the model in the
        # output instead of saving it to a file
        custom_iframe = f"""
        <iframe
            style="width: {self.nt.width}; height: calc({self.nt.height} + 50px);"
            frameborder="0"
            srcdoc="{escape(html_code)}"
            border:none !important;
            allowfullscreen webkitallowfullscreen mozallowfullscreen
        ></iframe>"""

        # We use a random id which will avoid collisions in most cases.
        return display(HTML(custom_iframe), display_id=random.randrange(0, 10**31))

    def show_editor(self):
        """Display an interactive layout editor. Use the update() method to apply changes."""
        self.layout.show_editor(self)

    def __format_rewards(self, s: State) -> str:
        """Create a string that contains the state-exit reward for this state. Starts with newline"""
        res = ""
        for reward_model in self.model.rewards:
            try:
                res += f"\n{reward_model.name}: {reward_model.get(s)}"
            except (
                KeyError
            ):  # If this reward model does not have a reward for this state.
                pass
        return res

    def __add_states(self):
        """For each state in the model, add a node to the graph."""
        for state in self.model.states.values():
            if state == self.model.get_initial_state():
                self.nt.add_node(
                    state.id,
                    label=",".join(state.labels) + self.__format_rewards(state),
                    color=self.layout.rget("init", "color"),
                    borderWidth=self.layout.rget("init", "borderWidth"),
                    shape=self.layout.rget("init", "shape"),
                )
            else:
                self.nt.add_node(
                    state.id,
                    label=",".join(state.labels) + self.__format_rewards(state),
                    color=self.layout.rget("states", "color"),
                    borderWidth=self.layout.rget("states", "borderWidth"),
                    shape=self.layout.rget("states", "shape"),
                )

    def __formatted_probability(self, prob: Number) -> str:
        """Take a probability value and format it nicely using a fraction or rounding it.
        Which one of these to pick is specified in the layout."""
        if self.layout.rget("numbers", "fractions"):
            return str(
                Fraction(prob).limit_denominator(
                    self.layout.rget("numbers", "max_denominator")
                )
            )
        else:
            return str(round(float(prob), self.layout.rget("numbers", "digits")))

    def __add_transitions(self):
        """For each transition in the model, add a transition in the graph.
        Also handles creating nodes for actions and their respective transitions.
        Note that an action may appear multiple times in the model with a different state as source."""
        action_id = self.ACTION_ID_OFFSET
        # In the visualization, both actions and states are nodes, so we need to keep track of how many actions we already have.
        for state_id, transition in self.model.transitions.items():
            for action, branch in transition.transition.items():
                if action == EmptyAction:
                    # Only draw probabilities
                    for prob, target in branch.branch:
                        self.nt.add_edge(
                            state_id,
                            target.id,
                            color=None,  # type: ignore
                            label=self.__formatted_probability(prob),
                        )
                else:
                    # Add the action's node
                    self.nt.add_node(
                        n_id=action_id,
                        label=action.name,
                        color=self.layout.rget("actions", "color"),  # type: ignore
                        borderWidth=self.layout.rget("actions", "borderWidth"),
                        shape=self.layout.rget("actions", "shape"),
                    )
                    # Add transition from this state TO the action.
                    self.nt.add_edge(state_id, action_id, color=None)  # type: ignore
                    # Add transition FROM the action to the states in its branch.
                    for prob, target in branch.branch:
                        self.nt.add_edge(
                            action_id,
                            target.id,
                            color=None,  # type: ignore
                            label=self.__formatted_probability(prob),
                        )
                    action_id += 1


def show(
    model: Model,
    name: str = "model",
    notebook: bool = True,
    cdn_resources: str = "remote",
    layout: Layout = DEFAULT,
    show_editor: bool = False,
) -> Visualization:
    """Create and show a visualization of a Model using a pyvis Network

    Args:
        model (Model): The stormvogel model to be displayed.
        name (str, optional): The name of the resulting html file.
        notebook (bool, optional): Leave to true if you are using in a notebook. Defaults to True.

    Returns: Visualization object.
    """
    vis = Visualization(model, name, notebook, cdn_resources, layout)
    if show_editor:
        vis.show_editor()
    vis.show()
    return vis
