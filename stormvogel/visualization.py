"""Contains the code responsible for model visualization."""

import random
import math
from fractions import Fraction
from html import escape

from IPython.display import HTML, display
from pyvis.network import Network

from stormvogel.layout import DEFAULT, Layout
from stormvogel.model import EmptyAction, Model, Number, State, Action
from stormvogel.rdict import rget

from stormvogel import result


def und(x: str) -> str:
    """Replace space by underscore"""
    return x.replace(" ", "_")


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
        separate_edit_labels: list[str] = [],
    ) -> None:
        """Create visualization of a Model using a pyvis Network

        Args:
            model (Model): The stormvogel model to be displayed.
            name (str, optional): The name of the resulting html file. May or may not include .html extension.
            notebook (bool, optional): Leave to true if you are using in a notebook. Defaults to True.
            separate_edit_labels (list[str]): List of labels that are edited separately. Defaults to [].
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
        self.reformatted_labels = list(map(und, separate_edit_labels))
        self.layout.set_groups(self.reformatted_labels)

    def get_labels(self):
        return self.model.get_labels()

    def __update_physics_enabled(self):
        """Disable physics if the model has more than 10000 states."""
        if "physics" not in self.layout.layout:
            self.layout.layout["physics"] = {}
        if len(self.model.states) > 10000:
            self.layout.layout["physics"]["enabled"] = False

    def __reload_nt(self):
        """(Re)load the pyvis network."""
        self.nt = Network(
            notebook=self.notebook, directed=True, cdn_resources=self.cdn_resources
        )
        self.layout.set_groups(self.reformatted_labels)
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
        """Create a string that represents the state-exit reward for this state. Starts with newline."""
        res = ""
        for reward_model in self.model.rewards:
            try:
                res += f"\n{reward_model.name} {reward_model.get(s)}"
            except (
                KeyError
            ):  # If this reward model does not have a reward for this state.
                pass
        return res

    def __format_result(self, s: State) -> str:
        """Create a string that represents the result of this state. Starts with a newline."""
        if self.result is None or self.result.get_result_of_state(s) is None:
            return ""
        result_symbol = self.layout.layout["resultSymbol"]
        return (
            "\n"
            + result_symbol
            + " "
            + self.__format_probability(self.result.get_result_of_state(s))  # type: ignore
        )

    def __add_states(self):
        """For each state in the model, add a node to the graph."""
        for state in self.model.states.values():
            result = self.__format_result(state)
            reward = self.__format_rewards(state)
            # Check if this state's first label is specified in the layout.
            group = (
                und(state.labels[0])
                if len(state.labels) > 0
                and und(state.labels[0]) in self.layout.layout["groups"].keys()
                else "states"  # If no label is specified, simply add it to "states".
            )

            self.nt.add_node(
                state.id,
                label="{" + ",".join(state.labels) + "}" + result + reward,
                color=None,  # type: ignore
                shape=None,  # type: ignore
                group=group,
            )

    def __format_probability(self, prob: Number) -> str:
        """Take a probability value and format it nicely using a fraction or rounding it.
        Which one of these to pick is specified in the layout."""
        if isinstance(prob, str) or math.isinf(prob):
            return str(prob)
        elif rget(
            self.layout.layout, ["numbers", "fractions"]
        ):  # If fractions are enabled
            return str(Fraction(prob).limit_denominator(1000))
        else:
            return str(
                round(float(prob), rget(self.layout.layout, ["numbers", "digits"]))
            )

    def __determine_group(self, state_id: int, a: Action) -> str:
        """Determine the group of Action a, coming from the state with state_id."""
        if self.result is None or self.result.scheduler is None:
            return "actions"
        if self.result.scheduler[state_id] == a:
            return "scheduled_actions"
        else:
            return "actions"

    def __add_transitions(self):
        """For each transition in the model, add a transition in the graph.
        Also handles creating nodes for actions and their respective transitions.
        Note that an action may appear multiple times in the model with a different state as source."""
        action_id = self.ACTION_ID_OFFSET

        # In the visualization, both actions and states are nodes, so we need to keep track of how many actions we already have.
        for state_id, transition in self.model.transitions.items():
            for action, branch in transition.transition.items():
                if (
                    action == EmptyAction
                ):  # If an action is empty, simply draw a transition from state to state.
                    for prob, target in branch.branch:
                        self.nt.add_edge(
                            state_id,
                            target.id,
                            color=None,  # type: ignore
                            label=self.__format_probability(prob),
                        )
                else:  # An actual action.
                    group = self.__determine_group(state_id, action)
                    edge_color = None
                    if (
                        group == "scheduled_actions"
                        and self.layout.layout["schedColor"]
                    ):
                        edge_color = self.layout.layout["groups"]["scheduled_actions"][
                            "color"
                        ]["background"]
                    # Add the action's node
                    self.nt.add_node(
                        n_id=action_id,
                        label=action.name,
                        color=None,  # type: ignore
                        shape=None,  # type: ignore
                        group=group,
                    )
                    # Add transition from this state TO the action.
                    self.nt.add_edge(state_id, action_id, color=edge_color)  # type: ignore
                    # Add transition FROM the action to the states in its branch.
                    for prob, target in branch.branch:
                        self.nt.add_edge(
                            action_id,
                            target.id,
                            color=edge_color,  # type: ignore
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
    separate_edit_labels=[],
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
        separate_edit_labels (list[str]): List of labels that are edited separately. Defaults to [].

    Returns: Visualization object.
    """
    vis = Visualization(
        model, result, name, notebook, cdn_resources, layout, separate_edit_labels
    )
    if show_editor:
        vis.show_editor()
    vis.show()
    return vis
