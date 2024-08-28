"""Contains the code responsible for model visualization."""

import math
from fractions import Fraction

from stormvogel.visjs import Network

from stormvogel.layout import DEFAULT, Layout
from stormvogel.model import EmptyAction, Model, Number, State
from stormvogel.rdict import rget

from stormvogel import result

from ipywidgets import HBox, HTML
from IPython.display import display, clear_output

import random
import string


def und(x: str) -> str:
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
        name: str | None = None,
        layout: Layout = DEFAULT(),
        separate_labels: list[str] = [],
    ) -> None:
        """Create visualization of a Model using a pyvis Network

        NEVER CREATE TWO VISUALIZATIONS WITH THE SAME NAME, STUFF MIGHT BREAK.

        Args:
            model (Model): The stormvogel model to be displayed.
            name (str, optional): Internally used name. Will be randomly generated if left as None.
            notebook (bool, optional): Leave to true if you are using in a notebook. Defaults to True.
        """
        if name is None:
            self.name = "".join(random.choices(string.ascii_letters, k=10))
        self.model = model
        # TODO self.result = result
        self.layout = layout
        self.result = result
        self.name = name
        self.separate_labels = list(map(und, separate_labels))
        self.nt = Network(name=self.name)  # type: ignore
        self.__add_states()
        self.__add_transitions()
        self.__update_physics_enabled()
        self.reload()

    def __update_physics_enabled(self):
        """Enable physics iff the model has less than 10000 states."""
        if "physics" not in self.layout.layout:
            self.layout.layout["physics"] = {}
        self.layout.layout["physics"]["enabled"] = len(self.model.states) < 10000

    def reload(self):
        """Tries to reload an existing visualization (so it uses a modified layout). If show was not called before, nothing happens."""
        self.layout.set_groups(self.separate_labels)
        self.nt.set_options(str(self.layout))
        self.nt.width = self.layout.layout["width"]
        self.nt.height = self.layout.layout["height"]
        self.nt.reload()

    def update(self):
        """Tries to update an existing visualization to apply layout changes WITHOUT reloading. If show was not called before, nothing happens."""
        self.layout.set_groups(self.separate_labels)
        self.nt.update_options(str(self.layout))

    def show(self):
        """Show or update the constructed graph as a html file."""
        self.reload()
        self.nt.show()
        return self.nt.handle

    def show_editor(self, display_: bool = True) -> None:
        """Display an interactive layout editor. Use the update() method to apply changes."""
        return self.layout.show_editor(self, display_=display_)  # type: ignore

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
            res = (
                formatted_result_of_state if self.layout.layout["show_results"] else ""
            )
            rewards = (
                self.__format_rewards(state)
                if self.layout.layout["show_rewards"]
                else ""
            )

            group = "states"  # Use a specific group if specified.
            if und(state.labels[0]) in self.separate_labels:
                group = und(state.labels[0])

            self.nt.add_node(
                state.id,
                label=",".join(state.labels) + rewards + res,
                color=None,  # type: ignore
                shape=None,  # type: ignore
                group=group,
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
                        id=action_id,
                        label=action.name,
                        color=None
                        if scheduler is not None
                        and scheduler.get_choice_of_state(
                            self.model.get_state_by_id(state_id)
                        )
                        == str(list(action.labels))
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


class CustomBox:
    def __init__(self, vis) -> None:
        self.vis = vis
        display(self.create_box())

    def create_box(self) -> None:
        iframe = self.vis.nt.generate_iframe()
        editor = self.vis.layout.show_editor(
            vis=self.vis, display_=False, reload_function=clear_output
        )
        return HBox(children=[HTML(iframe), editor])  # type: ignore

    def reload(self) -> None:
        print("hi")
        clear_output(wait=True)
        # display(self.create_box(), self.vis.nt.width, self.vis.nt.height)


def show(
    model: Model,
    result: result.Result | None = None,
    name: str = "model",
    layout: Layout = DEFAULT(),
    show_editor: bool = False,
    separate_labels: list[str] = [],
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

    vis = Visualization(
        model=model,
        result=result,
        name=name,
        layout=layout,
        separate_labels=separate_labels,
    )
    if show_editor:  # Manually generate the iframe.
        CustomBox(vis)
    else:
        vis.show()

    return vis
