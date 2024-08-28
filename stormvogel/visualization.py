"""Contains the code responsible for model visualization."""

import math
from fractions import Fraction
import ipywidgets as widgets
import IPython.display as ipd
import random
import string

import stormvogel.model as svm
import stormvogel.result as svr
import stormvogel.layout as svl
import stormvogel.visjs as visjs


def und(x: str) -> str:
    """Replace spaces by underscores."""
    return x.replace(" ", "_")


class Visualization:
    """Handles visualization of a Model using a Network from stormvogel.visjs."""

    ACTION_ID_OFFSET: int = 10**10
    # In the visualization, both actions and states are nodes with an id.
    # This offset is used to keep their ids from colliding. It should be some high constant.

    def __init__(
        self,
        model: svm.Model,
        name: str | None = None,
        result: svr.Result | None = None,
        layout: svl.Layout = svl.DEFAULT(),
        separate_labels: list[str] = [],
        output: widgets.Output | None = None,
        debug_output: widgets.Output = widgets.Output(),
    ) -> None:
        """Create visualization of a Model using a pyvis Network

        NEVER CREATE TWO VISUALIZATIONS WITH THE SAME NAME, STUFF MIGHT BREAK.

        Args:
            model (Model): The stormvogel model to be displayed.
            name (str, optional): Internally used name. Will be randomly generated if left as None.
            result (Result, optional): Result corresponding to the model.
            layout (Layout, optional): Layout used for the visualization.
            separate_labels (list[str], optional): Labels that should be edited separately according to the layout.
            output (widgets.Output): An output widget within which the network should be displayed.
                If left as None, the Network will display its own output.
                If specified, display should be called on this output in order to see the result.
            debug_output (widgets.Output): Debug information is displayed in this output. Leave to default if that doesn't interest you.
        """
        self.model: svm.Model = model
        if name is None:
            self.name: str = "".join(random.choices(string.ascii_letters, k=10))
        else:
            self.name: str = name
        self.result: svr.Result = result
        self.layout: svl.Layout = layout
        self.separate_labels = list(map(und, separate_labels))
        self.layout.set_groups(self.separate_labels)

        if output is None:
            self.output: widgets.Output = widgets.Output()
            self.self_display = True
        else:
            self.output: widgets.Output = output
        self.debug_output: widgets.Output = debug_output

    def show(self) -> None:
        """(Re-)load the Network and display if self.self_display is True."""
        with self.output:
            ipd.clear_output()
        self.nt: visjs.Network = visjs.Network(
            name=self.name,
            width=self.layout.layout["width"],
            height=self.layout.layout["height"],
            output=self.output,
            debug_output=self.debug_output,
        )
        self.__add_states()
        self.__add_transitions()
        self.__update_physics_enabled()
        self.nt.set_options(str(self.layout))
        self.nt.show()
        if self.self_display:
            ipd.display(
                self.output
            )  # If we have self display enabled, also display the Output itself.

    def update(self) -> None:
        """Tries to update an existing visualization to apply layout changes WITHOUT reloading. If show was not called before, nothing happens."""
        # self.layout.set_groups(self.separate_labels) # TODO is this needed?
        self.nt.update_options(str(self.layout))

    def __add_states(self) -> None:
        """For each state in the model, add a node to the graph."""
        for state in self.model.states.values():
            if self.layout.layout["show_results"]:
                res = self.__format_result(state)
            else:
                res = ""
            if self.layout.layout["show_rewards"]:
                rewards = self.__format_rewards(state)
            else:
                rewards = ""

            group = "states"  # Default
            if (
                und(state.labels[0]) in self.separate_labels
            ):  # Use a specific group if specified.
                group = und(state.labels[0])

            self.nt.add_node(
                state.id,
                label=",".join(state.labels) + rewards + res,
                group=group,
            )

    def __add_transitions(self) -> None:
        """For each transition in the model, add a transition in the graph.
        Also handles creating nodes for actions and their respective transitions.
        Note that an action may appear multiple times in the model with a different state as source."""
        action_id = self.ACTION_ID_OFFSET
        # scheduler = self.result.scheduler if self.result is not None else None
        # In the visualization, both actions and states are nodes, so we need to keep track of how many actions we already have.
        for state_id, transition in self.model.transitions.items():
            for action, branch in transition.transition.items():
                if action == svm.EmptyAction:
                    # Only draw probabilities
                    for prob, target in branch.branch:
                        self.nt.add_edge(
                            state_id,
                            target.id,
                            label=self.__format_probability(prob),
                        )
                else:
                    # Add the action's node
                    self.nt.add_node(
                        id=action_id,
                        label=action.name,
                        group="actions",
                    )
                    # Add transition from this state TO the action.
                    self.nt.add_edge(state_id, action_id)  # type: ignore
                    # Add transition FROM the action to the states in its branch.
                    for prob, target in branch.branch:
                        self.nt.add_edge(
                            action_id,
                            target.id,
                            label=self.__format_probability(prob),
                        )
                    action_id += 1

    def __update_physics_enabled(self) -> None:
        """Enable physics iff the model has less than 10000 states."""
        if "physics" not in self.layout.layout:
            self.layout.layout["physics"] = {}
        self.layout.layout["physics"]["enabled"] = len(self.model.states) < 10000

    def __format_probability(self, prob: svm.Number) -> str:
        """Take a probability value and format it nicely using a fraction or rounding it.
        Which one of these to pick is specified in the layout."""
        if isinstance(prob, str) or math.isinf(prob):
            return str(prob)
        else:
            if self.layout.layout["numbers"]["fractions"]:
                return str(Fraction(prob).limit_denominator(1000))
            else:
                return str(round(float(prob), self.layout.layout["numbers"]["digits"]))

    def __format_rewards(self, s: svm.State) -> str:
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

    def __format_result(self, s: svm.State) -> str:
        if self.result is None:
            return ""
        result_of_state = self.result.get_result_of_state(s)
        if result_of_state is None:
            return ""
        return (
            "\n"
            + self.layout.layout["resultSymbol"]
            + " "
            + self.__format_probability(result_of_state)
        )


def show(
    model: svm.Model,
    result: svr.Result | None = None,
    name: str = "model",
    layout: svl.Layout = svl.DEFAULT(),
    show_editor: bool = False,
    separate_labels: list[str] = [],
    output: widgets.Output | None = None,
    debug_output: widgets.Output = widgets.Output(),
) -> Visualization:
    """Create and show a visualization of a Model using a pyvis Network

    Args:
        model (Model): The stormvogel model to be displayed.
        name (str, optional): Internally used name. Will be randomly generated if left as None.
        result (Result, optional): Result corresponding to the model.
        layout (Layout, optional): Layout used for the visualization.
        separate_labels (list[str], optional): Labels that should be edited separately according to the layout.
        output (widgets.Output): An output widget within which the network should be displayed.
            If left as None, the Network will display its own output.
            If specified, display should be called on this output in order to see the result.
        debug_output (widgets.Output): Debug information is displayed in this output. Leave to default if that doesn't interest you.

    Returns: Visualization object.
    """

    if show_editor:
        with debug_output:
            NotImplementedError()
        if output is None:
            # self_display = True
            output = widgets.Output()
        else:
            # self_display = False  # type: ignore
            # main_output = widgets.Output()  # type: ignore
            # vis_output = widgets.Output()  # type: ignore
            pass
    else:
        vis = Visualization(
            model=model,
            result=result,
            name=name,
            layout=layout,
            separate_labels=separate_labels,
        )
        vis.show()
    return vis  # type: ignore
