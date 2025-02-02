"""Shorter api for showing a model."""

import stormvogel.model
import stormvogel.layout
import stormvogel.visualization
import stormvogel.layout_editor
import stormvogel.communication_server
import stormvogel.result

import ipywidgets as widgets
import IPython.display as ipd


def show(
    model: stormvogel.model.Model,
    result: stormvogel.result.Result | None = None,
    scheduler: stormvogel.result.Scheduler | None = None,
    name: str = "model",
    layout: stormvogel.layout.Layout | None = None,
    show_editor: bool = True,
    separate_labels: list[str] = [],
    debug_output: widgets.Output = widgets.Output(),
) -> stormvogel.visualization.Visualization:
    """Create and show a visualization of a Model using a visjs Network

    Args:
        model (Model): The stormvogel model to be displayed.
        result (Result): A result associatied with the model.
        name (str, optional): Internally used name. Will be randomly generated if left as None.
        result (Result, optional): Result corresponding to the model.
        layout (Layout, optional): Layout used for the visualization.
        separate_labels (list[str], optional): Labels that should be edited separately according to the layout.
        save_and_embed: Save the html to an external file and embed. Does not support editor.
    Returns: Visualization object.
    """
    if layout is None:
        layout = stormvogel.layout.DEFAULT()
    do_init_server = (  # We only need to start the server if we want to have the layout editor.
        show_editor and stormvogel.communication_server.enable_server
    )
    # do_display = not show_editor
    vis = stormvogel.visualization.Visualization(
        model=model,
        name=name,
        result=result,
        scheduler=scheduler,
        layout=layout,
        separate_labels=separate_labels,
        do_display=False,
        debug_output=debug_output,
        do_init_server=do_init_server,
    )
    vis.show()
    if show_editor:
        e = stormvogel.layout_editor.LayoutEditor(
            layout, vis, do_display=False, debug_output=debug_output
        )
        e.show()
        box = widgets.HBox(children=[vis.output, e.output])
        ipd.display(box)
    else:  # Unfortunately, the sphinx docs only work if we save the html as a file and embed.
        iframe = vis.nt.generate_iframe()
        with open(name + ".html", "w") as f:
            f.write(iframe)
        ipd.display(ipd.HTML(filename=name + ".html"))

    return vis
