"""Layout editor."""

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
        output: widgets.Output = widgets.Output(),
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

    def try_update(self):
        if not self.loaded:
            return
        if self.layout.layout["saving"]["save_button"]:
            # Save iff the save button was pressed.
            self.layout.layout["saving"]["save_button"] = False
            # Also save the node positions.
            with self.debug_output:
                logging.debug(f"Status of vis {self.vis}")
            if self.vis is not None:
                with self.debug_output:
                    positions = self.vis.get_positions()
                    logging.debug(positions)
                self.layout.layout["positions"] = positions

            self.layout.save(
                self.layout.layout["saving"]["filename"],
                path_relative=self.layout.layout["saving"]["relative_path"],
            )
        if self.layout.layout["saving"]["load_button"]:
            # Load iff the load button was pressed.
            self.layout.layout["saving"]["load_button"] = False
            self.layout.load(
                self.layout.layout["saving"]["filename"],
                path_relative=self.layout.layout["saving"]["relative_path"],
            )
            self.show()  # TODO replace this with simply setting the button values so that the entire menu doesn't have to reload (looks weird).
            if self.vis is not None:
                self.vis.show()
        if self.layout.layout["reload_button"] and self.vis is not None:
            # Call show again iff the reload button was pressed.
            self.layout.layout["reload_button"] = False
            with self.debug_output:
                logging.info("Received reload button request.")
            self.vis.show()
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
