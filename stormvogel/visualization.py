"""Contains the code responsible for model visualization."""

import stormvogel.model
import stormvogel.layout
import stormvogel.result
import stormvogel.simulator as simulator
import stormvogel.network
import stormvogel.displayable

import pathlib
from time import sleep
from typing import Tuple
import warnings
import ipywidgets as widgets
import IPython.display as ipd
import random
import string
import cairosvg
from fractions import Fraction


def und(x: str) -> str:
    """Replace spaces by underscores."""
    return x.replace(" ", "_")


def random_word(k: int) -> str:
    """Random word of lenght k"""
    return "".join(random.choices(string.ascii_letters, k=k))


def random_color() -> str:
    """Return a random HEX color."""
    return "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])


def blend_colors(c1: str, c2: str, factor: float) -> str:
    """Blend two colors in HEX format. #RRGGBB.
    Args:
        color1 (str): Color 1 in HEX format #RRGGBB
        color2 (str): Color 2 in HEX format #RRGGBB
        factor (float): The fraction of the resulting color that should come from color1."""
    r1 = int("0x" + c1[1:3], 0)
    g1 = int("0x" + c1[3:5], 0)
    b1 = int("0x" + c1[5:7], 0)
    r2 = int("0x" + c2[1:3], 0)
    g2 = int("0x" + c2[3:5], 0)
    b2 = int("0x" + c2[5:7], 0)
    r_res = int(factor * r1 + (1 - factor) * r2)
    g_res = int(factor * g1 + (1 - factor) * g2)
    b_res = int(factor * b1 + (1 - factor) * b2)
    return "#" + "".join("%02x" % i for i in [r_res, g_res, b_res])


class Visualization(stormvogel.displayable.Displayable):
    """Handles visualization of a Model using a Network from stormvogel.network."""

    ACTION_ID_OFFSET: int = 10**10
    # In the visualization, both actions and states are nodes with an id.
    # This offset is used to keep their ids from colliding. It should be some high constant.

    def __init__(
        self,
        model: stormvogel.model.Model,
        result: stormvogel.result.Result | None = None,
        scheduler: stormvogel.result.Scheduler | None = None,
        layout: stormvogel.layout.Layout = stormvogel.layout.DEFAULT(),
        output: widgets.Output | None = None,
        debug_output: widgets.Output = widgets.Output(),
        use_iframe: bool = False,
        do_init_server: bool = True,
        do_display: bool = True,
        max_states: int = 1000,
        max_physics_states: int = 500,
    ) -> None:
        """Create and show a visualization of a Model using a visjs Network
        Args:
            model (Model): The stormvogel model to be displayed.
            result (Result, optional): A result associatied with the model.
                The results are displayed as numbers on a state. Enable the layout editor for options.
                If this result has a scheduler, then the scheduled actions will have a different color etc. based on the layout
            scheduler (Scheduler, optional): The scheduled actions will have a different color etc. based on the layout
                If both result and scheduler are set, then scheduler takes precedence.
            layout (Layout): Layout used for the visualization.
            show_editor (bool): Show an interactive layout editor.
            use_iframe (bool): Wrap the generated html inside of an IFrame.
                In some environments, the visualization works better with this enabled.
            output (widgets.Output): The output widget in which the network is rendered.
                Whether this widget is also displayed automatically depends on do_display.
            debug_output (widgets.Output): Output widget that can be used to debug interactive features.
            do_init_server (bool): Initialize a local server that is used for communication between Javascript and Python.
                If this is set to False, then exporting network node positions and svg/pdf/latex is impossible.
            do_display (bool): The Visualization displays on its own iff this is enabled.
                This option is useful for situations where you want to manage the displaying externally.
            max_states (int): If the model has more states, then the network is not displayed.
            max_physics_states (int): If the model has more states, then physics are disabled.
        Returns: Visualization object.
        """
        super().__init__(output, do_display, debug_output)
        self.name: str = random_word(10)
        self.model: stormvogel.model.Model = model
        self.result: stormvogel.result.Result | None = result
        self.scheduler: stormvogel.result.Scheduler | None = scheduler
        self.use_iframe: bool = use_iframe
        self.max_states: int = max_states
        self.max_physics_states: int = max_physics_states
        # If a scheduler was not set explictly, but a result was set, then take the scheduler from the results.
        self.layout: stormvogel.layout.Layout = layout
        if self.scheduler is None:
            if self.result is not None:
                self.scheduler = self.result.scheduler

        # Set "scheduler" as an active group iff it is present.
        if self.scheduler is not None:
            layout.add_active_group("scheduled_actions")
        else:  # Otherwise, disable it
            layout.remove_active_group("scheduled_actions")

        self.do_init_server: bool = do_init_server
        self.__create_nt()
        self.network_action_map_id: dict[tuple[int, stormvogel.model.Action], int] = {}
        # Relate state ids and actions to the node id of the action for this state in the network.

    def __create_nt(self) -> None:
        """Reload the node positions and create the network."""
        self.nt: stormvogel.network.Network = stormvogel.network.Network(
            name=self.name,
            width=self.layout.layout["misc"]["width"],
            height=self.layout.layout["misc"]["height"],
            output=self.output,
            debug_output=self.debug_output,
            do_display=False,
            do_init_server=self.do_init_server,
            positions=self.layout.layout["positions"],
            use_iframe=self.use_iframe,
        )

    def show(self) -> None:
        """(Re-)load the Network and display if self.do_display is True.
        Important side effect: all changes to the layout are applied.
        This also includes updating the edit groups.
        """
        with self.output:  ## If there was already a rendered network, clear it.
            ipd.clear_output()
        if len(self.model.get_states()) > self.max_states:
            with self.output:
                print(
                    f"This model has more than {self.max_states} states. If you want to proceed, set max_states to a higher value."
                    f"This is to prevent the browser from crashing, be careful."
                )
            return
        if len(self.model.get_states()) > self.max_physics_states:
            with self.output:
                print(
                    f"This model has more than {self.max_physics_states} states. If you want physics, set max_physics_states to a higher value."
                    f"Physics are disabled to prevent the browser from crashing, be careful."
                )
            self.layout.layout["physics"] = False
            self.layout.copy_settings()
        self.__create_nt()
        if self.layout.layout["misc"]["explore"]:
            self.nt.enable_exploration_mode(self.model.get_initial_state().id)

        # Set the (possibly updated) possible edit groups
        underscored_labels = set(map(und, self.model.get_labels()))
        possible_groups = underscored_labels.union(
            {"states", "actions", "scheduled_actions"}
        )
        self.layout.set_possible_groups(possible_groups)

        self.__add_states()
        self.__add_transitions()
        self.nt.set_options(str(self.layout))
        if self.nt is not None:
            self.nt.show()
        self.maybe_display_output()

    def clear(self) -> None:
        """Clear visualization."""
        with self.output:
            ipd.clear_output()

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
        """For each state in the model, add a node to the graph. I"""
        if self.nt is None:
            return
        for state in self.model.get_states().values():
            res = self.__format_result(state)
            observations = self.__format_observations(state)
            rewards = self.__format_rewards(state, stormvogel.model.EmptyAction)
            group = self.__group_state(state, "states")

            color = None

            result_colors = True
            if result_colors and self.result is not None:
                result = self.result.get_result_of_state(state)
                max_result = self.result.maximum_result()
                if isinstance(result, (int, float, Fraction)) and isinstance(
                    max_result, (int, float, Fraction)
                ):
                    color1 = "#000000"
                    color2 = "#ffffff"
                    factor = result / max_result if max_result != 0 else 1
                    color = blend_colors(color1, color2, float(factor))

            self.nt.add_node(
                state.id,
                label=",".join(state.labels) + rewards + res + observations,
                group=group,
                color=color,
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
                            label=self.__format_number(prob),
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
                        self.network_action_map_id[state_id, action] = network_action_id
                        self.nt.add_edge(
                            network_action_id,
                            target.id,
                            label=self.__format_number(prob),
                            color=edge_color,
                        )
                    network_action_id += 1

    def __format_number(self, n: stormvogel.model.Number) -> str:
        """Call number_to_string in model.py while accounting for the settings specified in the layout object."""
        return stormvogel.model.number_to_string(
            n,
            self.layout.layout["numbers"]["fractions"],
            self.layout.layout["numbers"]["digits"],
            self.layout.layout["numbers"]["denominator_limit"],
        )

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
                res += f"\t{reward_model.name}: {self.__format_number(reward)}"
        if res == EMPTY_RES:
            return ""
        return res

    def __format_result(self, s: stormvogel.model.State) -> str:
        """Create a string that shows the result for this state. Starts with newline.
        If results are not enabled, then it returns the empty string."""
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
            + self.__format_number(result_of_state)
        )

    def __format_observations(self, s: stormvogel.model.State) -> str:
        """Create a String that shows the observation for this state (FOR POMDPs).
        Starts with newline."""
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

    def generate_html(self) -> str:
        """Get HTML code that can be used to show this visualization."""
        return self.nt.generate_html()

    def generate_iframe(self) -> str:
        """Get the HTML code that can be used to show this visualization, wrapped in an IFrame."""
        return self.nt.generate_iframe()

    def generate_svg(self) -> str:
        """Generate an svg image of the network."""
        return self.nt.generate_svg()

    def export(self, output_format: str, filename: str = "export") -> None:
        """
        Export the visualization to your preferred output format.
        The appropriate file extension will be added automatically.

        Parameters:
            output_format (str): Desired export format.
            filename (str): Base name for the exported file.

        Supported output formats (not case-sensitive):

            "HTML"    → An interactive .html file (e.g., draggable nodes)
            "IFrame"  → Exports as an <iframe> wrapped HTML in a .html file
            "PDF"     → Exports to .pdf (via conversion from SVG)
            "SVG"     → Exports to .svg vector image
        """
        output_format = output_format.lower()
        filename_base = pathlib.Path(filename).with_suffix(
            ""
        )  # remove extension if present

        if output_format == "html":
            html = self.generate_html()
            (filename_base.with_suffix(".html")).write_text(html, encoding="utf-8")

        elif output_format == "iframe":
            iframe = self.generate_iframe()
            (filename_base.with_suffix(".html")).write_text(iframe, encoding="utf-8")

        elif output_format == "svg":
            svg = self.generate_svg()
            (filename_base.with_suffix(".svg")).write_text(svg, encoding="utf-8")

        elif output_format == "pdf":
            svg = self.generate_svg()
            cairosvg.svg2pdf(
                bytestring=svg.encode("utf-8"), write_to=filename_base.name + ".pdf"
            )

        elif output_format == "latex":
            svg = self.generate_svg()
            # Create the 'export' folder if it doesn't exist
            export_folder = pathlib.Path(filename_base)
            export_folder.mkdir(parents=True, exist_ok=True)
            pdf_filename = filename_base.with_suffix(".pdf")
            # Convert SVG to PDF
            cairosvg.svg2pdf(
                bytestring=svg.encode("utf-8"),
                write_to=str(export_folder / pdf_filename),
            )

            # Create the LaTeX file
            latex_content = f"""\\documentclass{{article}}
\\usepackage{{graphicx}}
\\begin{{document}}
\\begin{{figure}}[h!]
\\centering
\\includegraphics[width=\\textwidth]{{{pdf_filename.name}}}
\\caption{{Generated using Stormvogel. TODO insert citing instructions}}
\\end{{figure}}
\\end{{document}}
"""
            # Write the LaTeX code to a .tex file
            (export_folder / filename_base.with_suffix(".tex")).write_text(
                latex_content, encoding="utf-8"
            )

        else:
            raise RuntimeError(f"Export format not supported: {output_format}")

    def get_positions(self) -> dict:
        """Get Network's current (interactive, dragged) node positions. Only works if show was called before.
        NOTE: This method only works after the network is properly loaded."""
        return self.nt.get_positions() if self.nt is not None else {}

    def __to_state_action_sequence(
        self, path: simulator.Path
    ) -> list[stormvogel.model.Action | stormvogel.model.State]:
        """Convert a Path to a list containing actions and states."""
        res: list[stormvogel.model.Action | stormvogel.model.State] = [
            self.model.get_initial_state()
        ]
        for _, v in path.path.items():
            if isinstance(v, tuple):
                res += list(v)
            else:
                res.append(v)
        return res

    def highlight_state(self, s_id: int, color: str | None = "red"):
        """Highlight a state in the model by changing its color. You can clear the current color by setting it to None."""
        self.nt.set_node_color(s_id, color)

    def highlilght_action(
        self, s_id: int, action: stormvogel.model.Action, color: str | None = "red"
    ):
        """Highlight an action in the model by changing its color. You can clear the current color by setting it to None."""
        try:
            nt_id = self.network_action_map_id[s_id, action]
            self.nt.set_node_color(nt_id, color)
        except KeyError:
            warnings.warn(
                "Tried to highlight an action that is not present in this model."
            )

    def highlight_state_set(self, state_ids: set[int], color: str | None = "blue"):
        """Highlight a set of states in the model by changing their color. You can clear the current color by setting it to None."""
        for s_id in state_ids:
            self.nt.set_node_color(s_id, color)

    def highlight_action_set(
        self,
        state_action_set: set[Tuple[int, stormvogel.model.Action]],
        color: str = "red",
    ):
        """Highlight a set of actions in the model by changing their color. You can clear the current color by setting it to None."""
        for s_id, a in state_action_set:
            self.highlilght_action(s_id, a, color)

    def highlight_decomposition(
        self,
        decomp: list[Tuple[set[int], set[Tuple[int, stormvogel.model.Action]]]],
        colors: list[str] | None = None,
    ):
        """Highlight a set of tuples of (states and actions) in the model by changing their color.
        Args:
            decomp: A list of tuples (states, actions)
            colors (optional): A list of colors for the decompossitions. Random colors are picked by default."""
        for n, v in enumerate(decomp):
            if colors is None:
                color = random_color()
            else:
                color = colors[n]
            self.highlight_state_set(v[0], color)
            self.highlight_action_set(v[1], color)

    def clear_highlighting(self):
        """Clear all highlighting that is currently active, returing all states to their original colors."""
        for s_id in self.model.get_states():
            self.nt.set_node_color(s_id, None)
        for a_id in self.network_action_map_id.values():
            self.nt.set_node_color(a_id, None)

    def highlight_path(
        self,
        path: simulator.Path,
        color: str,
        delay: float = 1,
        clear: bool = True,
    ) -> None:
        """Highlight the path that is provided as an argument in the model.
        Args:
            path (simulator.Path): The path to highlight.
            color (str | None): The color that the highlighted states should get (in HTML color standard).
                Set to None, in order to clear existing highlights on this path.
            delay (float): If not None, there will be a pause of a specified time before highlighting the next state in the path.
            clear (bool): Clear the highlighting of a state after it was highlighted. Only works if delay is not None.
                This is particularly useful for highlighting paths with loops."""
        seq = self.__to_state_action_sequence(path)
        for i, v in enumerate(seq):
            if isinstance(v, stormvogel.model.State):
                self.nt.set_node_color(v.id, color)
                sleep(delay)
                if clear:
                    self.nt.set_node_color(v.id, None)
            elif (
                isinstance(v, stormvogel.model.Action)
                and (seq[i - 1].id, v) in self.network_action_map_id
            ):
                node_id = self.network_action_map_id[seq[i - 1].id, v]
                self.nt.set_node_color(node_id, color)
                sleep(delay)
                if clear:
                    self.nt.set_node_color(node_id, None)
