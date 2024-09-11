import ipywidgets as widgets
import IPython.display as ipd


class Displayable:
    def __init__(
        self,
        output: widgets.Output = widgets.Output(),
        do_display: bool = True,
        debug_output: widgets.Output = widgets.Output(),
    ) -> None:
        """Abstract class for displaying something.

        Args:
            output (widgets.Output, optional): Output window. Defaults to widgets.Output().
            do_display (bool, optional): Controls if it should display. Defaults to True.
            debug_output (widgets.Output, optional): Useful for debugging. Defaults to widgets.Output().
        """
        self.output: widgets.Output = output
        self.do_display: bool = do_display
        self.debug_output: widgets.Output = debug_output

    def maybe_display_output(self):
        """Display iff do_display is enabled."""
        if self.do_display:
            ipd.display(self.output)
