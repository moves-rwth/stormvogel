from ipywidgets import (
    interact,
    IntSlider,
    ColorPicker,
    Color,
    Checkbox,
    Text,
    Button,
    ToggleButtons,
    Output,
)
from IPython.display import display, HTML


class Editor:
    def maybe_update(self):
        """Update if auto_update is enabled."""
        if self.layout.auto_update:
            try:
                self.layout.vis.update()
            except Exception:
                pass  # Happens if auto_update is enabled but show was not called yet


class SaveEditor(Editor):
    def __init__(self, layout) -> None:
        self.layout = layout
        self.filename = "layouts/SAVE_FILE_NAME.json"
        self.path_relative = True
        # Text box for file name
        interact(self.set_filename, x=Text(value=self.filename, description="Filename"))
        # Relative checkbox
        interact(
            self.set_relative_path,
            x=Checkbox(value=self.path_relative, description="Relative path"),
        )
        # Save button
        b = Button(description="Save")
        b.on_click(self.save)
        self.output = Output()
        display(b, self.output)

    def set_relative_path(self, x: bool):
        self.path_relative = x

    def set_filename(self, x: str):
        self.filename = x

    def save(self, b):
        with self.output:
            self.layout.save(self.filename, path_relative=self.path_relative)


class NumberEditor(Editor):
    def __init__(self, layout) -> None:
        self.layout = layout
        display(HTML("<h4>Numbers</h4>"))
        # Enable fractions
        interact(
            self.set_fractions,
            x=Checkbox(
                value=self.layout.rget("numbers", "fractions"),
                description="Enable fractions",
            ),
        )
        # Rounding digits
        interact(
            self.set_round_digits,
            x=IntSlider(
                min=0,
                max=20,
                step=1,
                value=self.layout.rget("numbers", "digits"),
                description="Digits",
            ),
        )
        # Fractions max denominator for fractions.
        interact(
            self.set_max_denom,
            x=IntSlider(
                min=0,
                max=20,
                step=1,
                value=self.layout.rget("numbers", "max_denominator"),
                description="Max denom",
            ),
        )

    def set_fractions(self, x: int) -> None:
        self.layout.layout["numbers"]["fractions"] = x
        self.maybe_update()

    def set_max_denom(self, x: int) -> None:
        self.layout.layout["numbers"]["max_denominator"] = x
        self.maybe_update()

    def set_round_digits(self, x: int) -> None:
        self.layout.layout["numbers"]["digits"] = x
        self.maybe_update()


class NodeEditor(Editor):
    def __init__(self, layout) -> None:
        self.layout = layout
        display(HTML("<h4>Nodes</h4>"))
        # Background color
        interact(
            self.set_background_color,
            x=ColorPicker(
                description="Background",
                value=self.layout.layout["nodes"]["color"]["background"],
            ),
        )
        # Border color
        interact(
            self.set_border_color,
            x=ColorPicker(
                description="Border",
                value=self.layout.layout["nodes"]["color"]["border"],
            ),
        )

    def set_background_color(self, x: Color):
        self.layout.layout["nodes"]["color"]["background"] = x
        self.maybe_update()

    def set_border_color(self, x: Color):
        self.layout.layout["nodes"]["color"]["border"] = x
        self.maybe_update()


class NodeGroupEditor(Editor):
    DEFAULT_COLOR = "#ffffff"

    def __init__(self, group: str, layout) -> None:
        self.group = group
        self.layout = layout
        self.color = (
            self.DEFAULT_COLOR
            if self.layout.rget(self.group, "color") is None
            else self.layout.rget(self.group, "color")
        )
        self.color_enabled = (
            False if self.layout.rget(self.group, "color") is None else True
        )
        # Group title
        display(HTML(f"<h4>{self.group}</h4>"))
        # Border width
        interact(
            self.set_borderWdith,
            x=IntSlider(
                min=0,
                max=20,
                step=1,
                value=self.layout.rget(self.group, "borderWidth"),
                description="Border width",
            ),
        )
        # Enable color
        interact(
            self.set_color_enabled,
            x=Checkbox(value=self.color_enabled, description="Enable color"),
        )
        # Color
        interact(self.set_color, x=ColorPicker(description="Color", value=self.color))
        # Shape
        SHAPES = [
            "ellipse",
            "circle",
            "database",
            "box",
            "text",
            "diamond",
            "dot",
            "star",
            "triangle",
            "triangleDown",
            "square",
        ]
        loaded_shape = self.layout.rget(self.group, "shape")
        initial_shape = loaded_shape if loaded_shape in SHAPES else "circle"
        interact(
            self.set_shape,
            x=ToggleButtons(
                options=SHAPES, value=initial_shape, description="shape", width=30
            ),
        )

    def set_shape(self, x: str) -> None:
        self.layout.layout[self.group]["shape"] = x
        self.maybe_update()

    def set_borderWdith(self, x: int) -> None:
        self.layout.layout[self.group]["borderWidth"] = x
        self.maybe_update()

    def set_color(self, x: Color) -> None:
        self.color = x
        if self.color_enabled:
            self.layout.layout[self.group]["color"] = x
            self.maybe_update()

    def set_color_enabled(self, x: bool) -> None:
        self.color_enabled = x
        if self.color_enabled:
            self.layout.layout[self.group]["color"] = self.color
        if not self.color_enabled:
            self.layout.layout[self.group]["color"] = None
        self.maybe_update()
