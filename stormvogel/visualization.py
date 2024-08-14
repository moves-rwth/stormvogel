"""Contains the code responsible for model visualization."""

import random
import math
from fractions import Fraction
from html import escape

from IPython.display import HTML, display
from pyvis.network import Network

from stormvogel.layout import DEFAULT, Layout
from stormvogel.model import EmptyAction, Model, Number, State
from stormvogel.rdict import rget

from stormvogel import result


class Visualization:
    """Handles visualization of a Model using a pyvis Network."""

    ACTION_ID_OFFSET: int = 10**10
    # In the visualization, both actions and states are nodes with an id.
    # This offset is used to keep their ids from colliding. It should be some high constant.

    def __init__(
        self,
        model: Model,
        result: result.Result | None,
        name: str = "model",
        notebook: bool = True,
        cdn_resources: str = "remote",
        layout: Layout = DEFAULT(),
    ) -> None:
        """Create visualization of a Model using a pyvis Network

        Args:
            model (Model): The stormvogel model to be displayed.
            name (str, optional): The name of the resulting html file. May or may not include .html extension.
            notebook (bool, optional): Leave to true if you are using in a notebook. Defaults to True.
        """
        self.model = model
        # TODO self.result = result
        self.layout = layout
        if (
            name[-5:] != ".html"
        ):  # We do not require the user to explicitly type .html in their names
            name += ".html"
        self.result = result
        self.name = name
        self.notebook = notebook
        self.cdn_resources = cdn_resources

    def get_labels(self):
        return self.model.get_labels()

    def __update_physics_enabled(self):
        """Enable physics iff the model has less than 10000 states."""
        if "physics" not in self.layout.layout:
            self.layout.layout["physics"] = {}
        self.layout.layout["physics"]["enabled"] = len(self.model.states) < 10000

    def __reload_nt(self):
        """(Re)load the pyvis network."""
        self.nt = Network(
            notebook=self.notebook, directed=True, cdn_resources=self.cdn_resources
        )
        self.__add_states()
        self.__add_transitions()
        self.__update_physics_enabled()
        self.layout.set_nt_layout(self.nt)

    def __generate_iframe(self):
        """We build our own iframe because we want to embed the model in the output instead of saving it to a file."""
        html_code = self.nt.generate_html(name=self.name, notebook=True)
        return f"""
            <iframe
                style="width: {self.nt.width}; height: calc({self.nt.height} + 50px);"
                frameborder="0"
                srcdoc="{escape(html_code)}"
                border:none !important;
                allowfullscreen webkitallowfullscreen mozallowfullscreen
            ></iframe>"""

    def update(self):
        """Tries to update an existing visualization (so it uses a modified layout). If show was not called before, nothing happens"""
        self.__reload_nt()
        try:
            self.handle.update(HTML(self.__generate_iframe()))  # type: ignore
        except AttributeError:
            pass

    def show(self):
        """Show or update the constructed graph as a html file."""
        self.__reload_nt()

        # We use a random id which will avoid collisions in most cases.
        self.handle = display(
            HTML(self.__generate_iframe()), display_id=random.randrange(0, 10**31)
        )

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
            result_of_state = (
                self.result.get_result_of_state(state)
                if self.result is not None
                else ""
            )
            formatted_result_of_state = (
                "\n" + self.__format_probability(result_of_state)
                if result_of_state is not None
                else ""
            )
            res = formatted_result_of_state
            if state == self.model.get_initial_state():
                self.nt.add_node(
                    state.id,
                    label=",".join(state.labels) + self.__format_rewards(state) + res,
                    color=None,  # type: ignore
                    shape=None,  # type: ignore
                    group="init",
                )
            else:
                self.nt.add_node(
                    state.id,
                    label=",".join(state.labels) + self.__format_rewards(state) + res,
                    color=None,  # type: ignore
                    shape=None,  # type: ignore
                    group="states",
                )

    def __format_probability(self, prob: Number) -> str:
        """Take a probability value and format it nicely using a fraction or rounding it.
        Which one of these to pick is specified in the layout."""
        if isinstance(prob, str) or math.isinf(prob):
            return str(prob)
        else:
            if rget(self.layout.layout, ["numbers", "fractions"]):
                return str(Fraction(prob).limit_denominator(1000))
            else:
                print(rget(self.layout.layout, ["numbers", "digits"]))
                return str(
                    round(float(prob), rget(self.layout.layout, ["numbers", "digits"]))
                )

    def __add_transitions(self):
        """For each transition in the model, add a transition in the graph.
        Also handles creating nodes for actions and their respective transitions.
        Note that an action may appear multiple times in the model with a different state as source."""
        action_id = self.ACTION_ID_OFFSET
        scheduler = self.result.scheduler if self.result is not None else None
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
                            label=self.__format_probability(prob),
                        )
                else:
                    # Add the action's node
                    self.nt.add_node(
                        n_id=action_id,
                        label=action.name,
                        color=None
                        if scheduler is not None
                        and scheduler[state_id] == str(list(action.labels)[0])
                        else None,  # TODO set different color if scheduler chooses this action # type: ignore
                        shape=None,  # type: ignore
                        group="actions",
                    )
                    # Add transition from this state TO the action.
                    self.nt.add_edge(state_id, action_id, color=None)  # type: ignore
                    # Add transition FROM the action to the states in its branch.
                    for prob, target in branch.branch:
                        self.nt.add_edge(
                            action_id,
                            target.id,
                            color=None,  # type: ignore
                            label=self.__format_probability(prob),
                        )
                    action_id += 1


def show(
    model: Model,
    result: result.Result | None = None,
    name: str = "model",
    notebook: bool = True,
    cdn_resources: str = "remote",
    layout: Layout = DEFAULT(),
    show_editor: bool = False,
) -> Visualization:
    """Create and show a visualization of a Model using a pyvis Network

    Args:
        model (Model): The stormvogel model to be displayed.
        name (str, optional): The name of the resulting html file. Defaults to "model".
        notebook (bool, optional): Leave to true if you are using in a notebook. Defaults to True.
        cdn_resources (str): Related to the pyvis library. Use "remote", "inline" or "local".
            Try changing this setting if you experience rendering issues. Defaults to "remote".
        layout (Layout): Which layout should be used. Defaults to DEFAULT.
        show_editor (bool): Show an interactive layout editor. Defaults to False.

    Returns: Visualization object.
    """
    vis = Visualization(model, result, name, notebook, cdn_resources, layout)
    if show_editor:
        vis.show_editor()
    vis.show()
    return vis
