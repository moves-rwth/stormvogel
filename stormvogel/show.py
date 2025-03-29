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
    show_editor: bool = False,
    debug_output: widgets.Output = widgets.Output(),
    use_iframe: bool = False,
    do_init_server: bool = True,
) -> stormvogel.visualization.Visualization:
    """Create and show a visualization of a Model using a visjs Network

    Args:
        model (Model): The stormvogel model to be displayed.
        result (Result): A result associatied with the model.
        name (str, optional): Internally used name. Will be randomly generated if left as None.
        result (Result, optional): Result corresponding to the model.
        layout (Layout, optional): Layout used for the visualization.
        show_editor (list[str], optional): Show an interactive layout editor.
        use_iframe: In some environments, the visualization works better with this enabled. Only takes effect when show_editor is disabled.
    Returns: Visualization object.
    """
    if layout is None:
        layout = stormvogel.layout.DEFAULT()
    # do_display = not show_editor
    vis = stormvogel.visualization.Visualization(
        model=model,
        name=name,
        result=result,
        scheduler=scheduler,
        layout=layout,
        do_display=False,
        debug_output=debug_output,
        do_init_server=do_init_server,
        use_iframe=use_iframe,
    )
    vis.show()
    if show_editor:
        e = stormvogel.layout_editor.LayoutEditor(
            layout, vis, do_display=False, debug_output=debug_output
        )
        e.show()
        box = widgets.HBox(children=[vis.output, e.output])
        ipd.display(box)
        vis.update()
    else:  # Unfortunately, the sphinx docs only work if we save the html as a file and embed.
        if use_iframe:
            iframe = vis.nt.generate_iframe()
        else:
            iframe = vis.generate_html()
        with open(name + ".html", "w") as f:
            f.write(iframe)
        ipd.display(ipd.HTML(filename=name + ".html"))

    return vis


def show_bird():
    m = stormvogel.model.new_dtmc(create_initial_state=False)
    m.new_state("🐦")
    m.add_self_loops()
    return show(m, show_editor=False, layout=stormvogel.layout.SV())
