"""Contains the code responsible for model visualization."""

# Note to future maintainers: The way that IPython display behaves is very flakey sometimes.
# If you remove a with output: statement, everything might just break, be prepared.

import stormvogel.model
import stormvogel.layout
import stormvogel.result
import stormvogel.visjs
import stormvogel.displayable

import math
import fractions
import ipywidgets as widgets
import IPython.display as ipd
import random
import string
import logging


def und(x: str) -> str:
    """Replace spaces by underscores."""
    return x.replace(" ", "_")


def random_word(k: int) -> str:
    """Random word of lenght k"""
    return "".join(random.choices(string.ascii_letters, k=k))


class Visualization(stormvogel.displayable.Displayable):
    """Handles visualization of a Model using a Network from stormvogel.visjs."""

    ACTION_ID_OFFSET: int = 10**10
    # In the visualization, both actions and states are nodes with an id.
    # This offset is used to keep their ids from colliding. It should be some high constant.

    def __init__(
        self,
        model: stormvogel.model.Model,
        name: str | None = None,
        result: stormvogel.result.Result | None = None,
        layout: stormvogel.layout.Layout = stormvogel.layout.DEFAULT(),
        separate_labels: list[str] = [],
        positions: dict[str, dict[str, int]] | None = None,
        output: widgets.Output | None = None,
        do_display: bool = True,
        debug_output: widgets.Output = widgets.Output(),
        do_init_server: bool = True,
    ) -> None:
        """Create visualization of a Model using a pyvis Network

        NEVER CREATE TWO VISUALIZATIONS WITH THE SAME NAME, STUFF MIGHT BREAK.

        Args:
            model (Model): The stormvogel model to be displayed.
            name (str, optional): Internally used name. Will be randomly generated if left as None.
            result (Result, optional): Result corresponding to the model.
            layout (Layout, optional): Layout used for the visualization.
            separate_labels (list[str], optional): Labels that should be edited separately according to the layout.
            positions (dict[int, dict[str, int]] | None): A dictionary from state ids to positions.
                Determines where states should be placed in the visualization. Overrides saved positions in a loaded layout.
                Example: {1: {"x":5, "y":10}, 2: ....}
            do_display (bool): Set to true iff you want the Visualization to display. Defaults to True.
            debug_output (widgets.Output): Debug information is displayed in this output. Leave to default if that doesn't interest you.
            do_init_server (bool): Enable if you would like to start the server which is required for some visualization features. Defaults to True.
        """
        super().__init__(output, do_display, debug_output)
        if name is None:
            self.name: str = random_word(10)
        else:
            self.name: str = name + random_word(10)
        self.model: stormvogel.model.Model = model
        self.result: stormvogel.result.Result = result
        self.layout: stormvogel.layout.Layout = layout
        self.separate_labels: set[str] = set(map(und, separate_labels)).union(
            self.layout.layout["groups"].keys()
        )
        self.positions: dict[str, dict[str, int]]
        if positions is None:
            self.positions = self.layout.layout["positions"]
        else:
            self.positions = positions
        self.do_init_server: bool = do_init_server
        self.__create_nt()

    def __create_nt(self) -> None:
        self.nt: stormvogel.visjs.Network = stormvogel.visjs.Network(
            name=self.name,
            width=self.layout.layout["misc"]["width"],
            height=self.layout.layout["misc"]["height"],
            output=self.output,
            debug_output=self.debug_output,
            do_display=False,
            do_init_server=self.do_init_server,
        )

    def show(self) -> None:
        """(Re-)load the Network and display if self.do_display is True."""
        with self.debug_output:
            logging.info("Called Visualization.show()")
        with self.output:
            ipd.clear_output()
        self.__create_nt()
        if self.layout.layout["misc"]["explore"]:
            self.nt.enable_exploration_mode(self.model.get_initial_state().id)
        self.layout.set_groups(self.separate_labels)
        self.__add_states()
        self.__add_transitions()
        self.__update_physics_enabled()
        self.nt.set_options(str(self.layout))
        if self.nt is not None:
            self.nt.show()
        self.maybe_display_output()

    def update(self) -> None:
        """Tries to update an existing visualization to apply layout changes WITHOUT reloading. If show was not called before, nothing happens."""
        if self.nt is not None:
            self.nt.update_options(str(self.layout))

    def __add_states(self) -> None:
        """For each state in the model, add a node to the graph."""
        if self.nt is None:
            return
        for state in self.model.states.values():
            res = (
                self.__format_result(state)
                if self.layout.layout["state_properties"]["show_results"]
                else ""
            )
            rewards = (
                self.__format_rewards(state)
                if self.layout.layout["state_properties"]["show_rewards"]
                else ""
            )
            observations = (
                self.__format_observations(state)
                if self.layout.layout["state_properties"]["show_observations"]
                else ""
            )
            group = (  # Use a non-default group if specified.
                und(state.labels[0])
                if (
                    len(state.labels) > 0  # TODO generalize
                    and und(state.labels[0]) in self.separate_labels
                )
                else "states"
            )

            self.nt.add_node(
                state.id,
                label=",".join(state.labels) + rewards + res + observations,
                group=group,
                position_dict=self.positions,
            )

    def __add_transitions(self) -> None:
        """For each transition in the model, add a transition in the graph.
        Also handles creating nodes for actions and their respective transitions.
        Note that an action may appear multiple times in the model with a different state as source."""
        if self.nt is None:
            return
        action_id = self.ACTION_ID_OFFSET
        scheduler = self.result.scheduler if self.result is not None else None
        # In the visualization, both actions and states are nodes, so we need to keep track of how many actions we already have.
        for state_id, transition in self.model.transitions.items():
            for action, branch in transition.transition.items():
                if action == stormvogel.model.EmptyAction:
                    # Only draw probabilities
                    for prob, target in branch.branch:
                        self.nt.add_edge(
                            state_id,
                            target.id,
                            label=self.__format_probability(prob),
                        )
                else:
                    # Put the action in the group scheduled_actions if appropriate.
                    group = "actions"
                    if scheduler is not None:
                        choice = scheduler.get_choice_of_state(
                            state=self.model.get_state_by_id(state_id)
                        )
                        if choice == action:
                            group = "scheduled_actions"
                    # Add the action's node

                    self.nt.add_node(
                        id=action_id,
                        label=action.name,
                        group=group,
                        position_dict=self.positions,
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
        self.layout.layout["misc"]["enable_physics"] = len(self.model.states) < 10000

    def __format_probability(self, prob: stormvogel.model.Number) -> str:
        """Take a probability value and format it nicely using a fraction or rounding it.
        Which one of these to pick is specified in the layout."""
        if isinstance(prob, str) or math.isinf(prob):
            return str(prob)
        else:
            if self.layout.layout["numbers"]["fractions"]:
                return str(fractions.Fraction(prob).limit_denominator(1000))
            else:
                return str(round(float(prob), self.layout.layout["numbers"]["digits"]))

    def __format_rewards(self, s: stormvogel.model.State) -> str:
        """Create a string that contains the state-exit reward for this state. Starts with newline"""
        if len(self.model.rewards) == 0:
            return ""
        res = "\n" + self.layout.layout["state_properties"]["reward_symbol"]
        for reward_model in self.model.rewards:
            try:
                res += f"\t{reward_model.name}: {reward_model.get_state_reward(s)}"
            except (
                KeyError
            ):  # If this reward model does not have a reward for this state.
                return ""
        return res

    def __format_result(self, s: stormvogel.model.State) -> str:
        if self.result is None:
            return ""
        result_of_state = self.result.get_result_of_state(s)
        if result_of_state is None:
            return ""
        return (
            "\n"
            + self.layout.layout["state_properties"]["result_symbol"]
            + " "
            + self.__format_probability(result_of_state)
        )

    def __format_observations(self, s: stormvogel.model.State) -> str:
        if s.observation is None:
            return ""
        else:
            return (
                "\n"
                + self.layout.layout["state_properties"]["observation_symbol"]
                + " "
                + str(s.observation.observation)
            )

    def get_positions(self):
        """Get Network's current (interactive, dragged) node positions. Only works if show was called before (obviously)."""
        return self.nt.get_positions() if self.nt is not None else {}
