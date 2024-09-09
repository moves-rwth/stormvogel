"""Shorter api for showing a model."""

import stormvogel.model
import stormvogel.layout
import stormvogel.visualization
import stormvogel.layout_editor

import ipywidgets as widgets
import IPython.display as ipd


def show(
    model: stormvogel.model.Model,
    result: stormvogel.result.Result | None = None,
    name: str = "model",
    layout: stormvogel.layout.Layout = stormvogel.layout.DEFAULT(),
    show_editor: bool = False,
    separate_labels: list[str] = [],
    debug_output: widgets.Output = widgets.Output(),
) -> stormvogel.visualization.Visualization:
    """Create and show a visualization of a Model using a visjs Network

    Args:
        model (Model): The stormvogel model to be displayed.
        name (str, optional): Internally used name. Will be randomly generated if left as None.
        result (Result, optional): Result corresponding to the model.
        layout (Layout, optional): Layout used for the visualization.
        separate_labels (list[str], optional): Labels that should be edited separately according to the layout.

    Returns: Visualization object.
    """

    do_display = not show_editor
    vis = stormvogel.visualization.Visualization(
        model=model,
        name=name,
        result=result,
        layout=layout,
        separate_labels=separate_labels,
        do_display=do_display,
        debug_output=debug_output,
    )
    vis.show()

    if show_editor:
        e = stormvogel.layout_editor.LayoutEditor(
            layout, vis, do_display=False, debug_output=debug_output
        )
        e.show()
        box = widgets.HBox(children=[vis.output, e.output])
        ipd.display(box)

    return vis
