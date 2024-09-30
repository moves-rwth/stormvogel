"""Layout editor."""

import stormvogel.communication_server
import stormvogel.dict_editor
import stormvogel.displayable
import stormvogel.layout
import stormvogel.visualization

import IPython.display as ipd
import ipywidgets as widgets
import logging


class LayoutEditor(stormvogel.displayable.Displayable):
    def __init__(
        self,
        layout: stormvogel.layout.Layout,
        visualization: stormvogel.visualization.Visualization | None = None,
        output: widgets.Output | None = None,
        do_display: bool = True,
        debug_output: widgets.Output = widgets.Output(),
    ) -> None:
        super().__init__(output, do_display, debug_output)
        self.vis: stormvogel.visualization.Visualization | None = visualization
        self.layout: stormvogel.layout.Layout = layout
        self.loaded: bool = False  # True iff the layout is done loading.
        self.editor = stormvogel.dict_editor.DictEditor(
            schema=self.layout.schema,
            update_dict=self.layout.layout,
            on_update=self.try_update,
            do_display=False,
        )

    def copy_settings(self):
        """Copy some settings from one place in the layout to another place in the layout.
        They differ because visjs requires for them to be arranged a certain way which is not nice for an editor."""
        self.layout.layout["physics"] = self.layout.layout["misc"]["enable_physics"]
        self.layout.layout["width"] = self.layout.layout["misc"]["width"]
        self.layout.layout["height"] = self.layout.layout["misc"]["height"]

    def __failed_positions_save(self):
        return f"""Could not save the node positions of this graph in {self.layout.layout['saving']['filename']}.json
Sorry for the inconvenience. Here are some possible fixes.
1) Restart the kernel and re-run.
2) Is the port {stormvogel.communication_server.localhost_address}:{stormvogel.communication_server.server_port} (from the machine where jupyterlab runs) available?
If you are working remotely, it might help to forward this port. For example: 'ssh -N -L {stormvogel.communication_server.server_port}:{stormvogel.communication_server.localhost_address}{stormvogel.communication_server.server_port} YOUR_SSH_CONFIG_NAME'.
3) You might also want to consider changing stormvogel.communication_server.localhost_address to the IPv6 loopback address if you are using IPv6.
If you cannot get the server to work, set stormvogel.communication_server.enable_server to false and re-run.
This will speed up stormvogel and ignore this message, but it means that you cannot store positions in layout files.
Please contact the stormvogel developpers if you keep running into issues."""

    def process_save_button(self):
        if self.layout.layout["saving"]["save_button"]:
            # Save iff the save button was pressed.
            self.layout.layout["saving"]["save_button"] = False
            # Also save the node positions.
            with self.debug_output:
                logging.debug(f"Status of vis {self.vis}")
            if self.vis is not None:
                with self.output:
                    if stormvogel.communication_server.server is None:
                        with self.debug_output:
                            logging.info(
                                "Did not save node positions because the server is disabled."
                            )
                        with self.output:
                            print(
                                f"Did not save the node positions of this graph in {self.layout.layout['saving']['filename']}.json because the server is disabled."
                            )
                    else:
                        try:
                            positions = self.vis.get_positions()
                            with self.debug_output:
                                logging.debug(positions)
                            self.layout.layout["positions"] = positions
                        except TimeoutError:
                            with self.debug_output:
                                logging.warning(
                                    "Failed to save node positions in layout file."
                                )
                            with self.output:
                                print(self.__failed_positions_save())

            self.layout.save(
                self.layout.layout["saving"]["filename"],
                path_relative=self.layout.layout["saving"]["relative_path"],
            )

    def process_load_button(self):
        if self.layout.layout["saving"]["load_button"]:
            # Load iff the load button was pressed.
            self.layout.layout["saving"]["load_button"] = False
            try:
                self.layout.load(
                    self.layout.layout["saving"]["filename"],
                    path_relative=self.layout.layout["saving"]["relative_path"],
                )
                self.show()  # TODO replace this with simply setting the button values so that the entire menu doesn't have to reload (looks weird).
                if self.vis is not None:
                    self.vis.show()
            except OSError:
                logging.warning("Loaded file does not exist.")
                with self.output:
                    print(
                        f"Loaded file does not exist. {self.layout.layout['saving']['filename']}"
                    )

    def process_reload_button(self):
        if self.layout.layout["reload_button"] and self.vis is not None:
            # Call show again iff the reload button was pressed.
            self.layout.layout["reload_button"] = False
            with self.debug_output:
                logging.info("Received reload button request.")
            self.vis.show()

    def try_update(self):
        """Process the updates from the layout editor where required."""
        self.copy_settings()
        if not self.loaded:
            return
        self.process_save_button()
        self.process_load_button()
        self.process_reload_button()
        if self.vis is not None:
            self.vis.update()

    def try_show(self):
        if self.vis is not None:
            self.vis.show()

    def show(self) -> None:
        """Display an interactive layout editor, according to the schema."""
        self.loaded = False
        with self.editor.output:
            ipd.clear_output()
        self.editor = stormvogel.dict_editor.DictEditor(
            schema=self.layout.schema,
            update_dict=self.layout.layout,
            on_update=self.try_update,
            do_display=False,
            output=widgets.Output(),
        )
        self.editor.show()
        box = widgets.VBox(children=[self.editor.output])
        with self.output:
            ipd.clear_output()
            ipd.display(box)
        self.maybe_display_output()
        self.loaded = True
