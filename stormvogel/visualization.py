"""Contains the code responsible for model visualization."""

# Note to future maintainers: The way that IPython display behaves is very flakey sometimes.
# If you remove a with output: statement, everything might just break, be prepared.

from time import sleep
from typing import Tuple
import stormvogel.model
import stormvogel.layout
import stormvogel.result
import stormvogel.simulator
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
        scheduler: stormvogel.result.Scheduler | None = None,
        layout: stormvogel.layout.Layout = stormvogel.layout.DEFAULT(),
        output: widgets.Output | None = None,
        do_display: bool = True,
        debug_output: widgets.Output = widgets.Output(),
        do_init_server: bool = True,
    ) -> None:
        """Create visualization of a Model using a pyvis Network
        Args:
            model (Model): The stormvogel model to be displayed.
            name (str, optional): Internally used name. Will be randomly generated if left as None.
            result (Result, optional): Result corresponding to the model.
            scheduler(Scheduler, optional): Scheduler. The scheduled states can be given a distinct layout.
                If not set, then the scheduler from the result will be used.
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
        # Having two visualizations with the same name might break some interactive html stuff. This is why we add a random word to it.
        if name is None:
            self.name: str = random_word(10)
        else:
            self.name: str = name + random_word(10)
        self.model: stormvogel.model.Model = model
        self.result: stormvogel.result.Result | None = result
        self.scheduler: stormvogel.result.Scheduler | None = scheduler
        # If a scheduler was not set explictely, but a result was set, then take the scheduler from the results.
        self.layout: stormvogel.layout.Layout = layout
        if self.scheduler is None:
            if self.result is not None:
                self.scheduler = self.result.scheduler
        if self.scheduler is not None:  # Enable scheduled_actions as a default.
            self.layout.set_active_groups(["states", "actions", "scheduled_actions"])

        self.do_init_server: bool = do_init_server
        self.__create_nt()
        self.network_action_map_id: dict[tuple[int, stormvogel.model.Action], int] = {}
        # Keeps track of the ids of states for us.

    def __create_nt(self) -> None:
        """Reload the node positions and create the network."""
        self.nt: stormvogel.visjs.Network = stormvogel.visjs.Network(
            name=self.name,
            width=self.layout.layout["misc"]["width"],
            height=self.layout.layout["misc"]["height"],
            output=self.output,
            debug_output=self.debug_output,
            do_display=False,
            do_init_server=self.do_init_server,
            positions=self.layout.layout["positions"],
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

        # Set the (possibly updated) possible edit groups
        underscored_labels = set(map(und, self.model.get_labels()))
        possible_groups = underscored_labels.union({"states", "actions"})
        if self.scheduler is not None:
            possible_groups.add("scheduled_actions")
        self.layout.set_possible_groups(possible_groups)

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

    def __group_state(self, s: stormvogel.model.State, default: str) -> str:
        """Return the group of this state.
        That is, the label of s that has the highest priority, as specified by the user under edit_groups"""
        und_labels = set(map(lambda x: und(x), s.labels))
        res = list(
            filter(
                lambda x: x in und_labels, self.layout.layout["edit_groups"]["groups"]
            )
        )
        return und(res[0]) if res != [] else default

    def __group_action(
        self, s_id: int, a: stormvogel.model.Action, default: str
    ) -> str:
        """Return the group of this action. Only relevant for scheduling"""
        # Put the action in the group scheduled_actions if appropriate.
        if self.scheduler is None:
            return default

        choice = self.scheduler.get_choice_of_state(self.model.get_state_by_id(s_id))
        return "scheduled_actions" if a == choice else default

    def __add_states(self) -> None:
        """For each state in the model, add a node to the graph."""
        if self.nt is None:
            return
        for state in self.model.states.values():
            res = self.__format_result(state)
            observations = self.__format_observations(state)
            rewards = self.__format_rewards(state, stormvogel.model.EmptyAction)
            group = self.__group_state(state, "states")

            self.nt.add_node(
                state.id,
                label=",".join(state.labels) + rewards + res + observations,
                group=group,
            )

    def __add_transitions(self) -> None:
        """For each transition in the model, add a transition in the graph.
        Also handles creating nodes for actions and their respective transitions.
        Note that an action may appear multiple times in the model with a different state as source."""
        if self.nt is None:
            return
        network_action_id = self.ACTION_ID_OFFSET
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
                    group = self.__group_action(state_id, action, "actions")
                    reward = self.__format_rewards(
                        self.model.get_state_by_id(state_id), action
                    )

                    # Add the action's node
                    self.nt.add_node(
                        id=network_action_id,
                        label=",".join(action.labels) + reward,
                        group=group,
                    )
                    if group == "scheduled_actions":
                        try:
                            edge_color = self.layout.layout["groups"][
                                "scheduled_actions"
                            ]["color"]["border"]
                        except KeyError:
                            edge_color = None
                    else:
                        edge_color = None

                    # Add transition from this state TO the action.
                    self.nt.add_edge(state_id, network_action_id, color=edge_color)  # type: ignore
                    # Add transition FROM the action to the states in its branch.
                    for prob, target in branch.branch:
                        self.network_action_map_id[target.id, action] = (
                            network_action_id
                        )
                        self.nt.add_edge(
                            network_action_id,
                            target.id,
                            label=self.__format_probability(prob),
                            color=edge_color,
                        )
                    network_action_id += 1

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

    def __format_rewards(
        self, s: stormvogel.model.State, a: stormvogel.model.Action
    ) -> str:
        """Create a string that contains either the state exit reward (if actions are not supported)
        or the reward of taking this action from this state. (if actions ARE supported)
        Starts with newline"""
        if not self.layout.layout["state_properties"]["show_rewards"]:
            return ""
        EMPTY_RES = "\n" + self.layout.layout["state_properties"]["reward_symbol"]
        res = EMPTY_RES
        for reward_model in self.model.rewards:
            if self.model.supports_actions():
                if a in s.available_actions():
                    reward = reward_model.get_state_action_reward(s, a)
                else:
                    reward = None
            else:
                reward = reward_model.get_state_reward(s)
            if reward is not None and not (
                not self.layout.layout["state_properties"]["show_zero_rewards"]
                and reward == 0
            ):
                res += f"\t{reward_model.name}: {reward}"
        if res == EMPTY_RES:
            return ""
        return res

    def __format_result(self, s: stormvogel.model.State) -> str:
        if (
            self.result is None
            or not self.layout.layout["state_properties"]["show_results"]
        ):
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
        if (
            s.observation is None
            or not self.layout.layout["state_properties"]["show_observations"]
        ):
            return ""
        else:
            return (
                "\n"
                + self.layout.layout["state_properties"]["observation_symbol"]
                + " "
                + str(s.observation.observation)
            )

    def get_positions(self) -> dict:
        """Get Network's current (interactive, dragged) node positions. Only works if show was called before (obviously).
        NOTE: This method only works after the network is properly loaded."""
        return self.nt.get_positions() if self.nt is not None else {}

    def __set_tuple_color(
        self,
        v: Tuple[stormvogel.model.Action, stormvogel.model.State]
        | stormvogel.model.State,
        color: str | None,
    ) -> None:
        if isinstance(v, tuple):
            action = v[0]
            state = v[1]
            if (state.id, action) in self.network_action_map_id:
                self.nt.set_node_color(
                    self.network_action_map_id[state.id, action], color
                )
            self.nt.set_node_color(state.id, color)
        else:
            self.nt.set_node_color(v.id, color)

    def highlight_path(
        self,
        path: stormvogel.simulator.Path,
        color: str,
        delay: float | None = None,
        clear: bool = False,
    ) -> None:
        """Highlight the path that is provided as an argument in the model.
        Args:
            path (stormvogel.simulator.Path): The path to highlight.
            color (str | None): The color that the highlighted states should get (in HTML color standard).
                Set to None, in order to clear existing highlights on this path.
            delay (float | None): If not None, there will be a pause of a specified time before highlighting the next state in the path.
            clear (bool): Clear the highlighting of a state after it was highlighted. Only works if delay is not None.
                This is particularly useful for highlighting paths with loops."""
        init = self.model.get_initial_state()
        self.nt.set_node_color(init.id, color)
        if delay is not None:
            sleep(delay)
            if clear:
                self.__set_tuple_color(init, None)
        for _, v in path.path.items():
            self.__set_tuple_color(v, color)
            if delay is not None:
                sleep(delay)
                if clear:
                    self.__set_tuple_color(v, None)
