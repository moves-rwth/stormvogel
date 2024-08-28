"""Save button and apply button."""

from ipywidgets import Button, Output
from IPython.display import display


class SaveButton:
    """Save button."""

    def __init__(self, layout) -> None:
        self.layout = layout
        # Save button
        self.saveButton = Button(description="Save", button_style="success")
        self.saveButton.on_click(self.save)
        self.saveOutput = Output()

    def save(self, b):
        with self.saveOutput:
            self.layout.save(
                self.layout.layout["saving"]["filename"],
                path_relative=self.layout.layout["saving"]["relative_path"],
            )


class ReloadButton:
    """Reload button."""

    def __init__(self, layout, maybe_update) -> None:
        self.layout = layout
        self.maybe_update = maybe_update
        # Apply button
        self.applyButton = Button(description="Reload", button_style="info")
        self.applyButton.on_click(self.apply)
        self.applyOutput = Output()
        display(self.applyButton, self.applyOutput)

    def apply(self, b):
        with self.applyOutput:
            self.maybe_update()
