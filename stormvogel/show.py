"""Shorter api for showing a model."""

import stormvogel.model
import stormvogel.layout
import stormvogel.visualization
import stormvogel.layout_editor
import stormvogel.result

import ipywidgets as widgets
import IPython.display as ipd


def show(
    model: stormvogel.model.Model,
    result: stormvogel.result.Result | None = None,
    scheduler: stormvogel.result.Scheduler | None = None,
    layout: stormvogel.layout.Layout | None = None,
    show_editor: bool = False,
    debug_output: widgets.Output = widgets.Output(),
    use_iframe: bool = False,
    do_init_server: bool = True,
    max_states: int = 1000,
    max_physics_states: int = 500,
) -> stormvogel.visualization.JSVisualization:
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
        debug_output (widgets.Output): Output widget that can be used to debug interactive features.
        use_iframe(bool): Wrap the generated html inside of an IFrame.
            In some environments, the visualization works better with this enabled.
        do_init_server(bool): Initialize a local server that is used for communication between Javascript and Python.
            If this is set to False, then exporting network node positions and svg/pdf/latex is impossible.
        max_states (int): If the model has more states, then the network is not displayed.
        max_physics_states (int): If the model has more states, then physics are disabled.
    Returns: Visualization object.
    """
    if layout is None:
        layout = stormvogel.layout.DEFAULT()

    vis = stormvogel.visualization.JSVisualization(
        model=model,
        result=result,
        scheduler=scheduler,
        layout=layout,
        debug_output=debug_output,
        do_init_server=do_init_server,
        use_iframe=use_iframe,
        max_states=max_states,
        max_physics_states=max_physics_states,
    )
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
            iframe = vis.generate_iframe()
        else:
            iframe = vis.generate_html()
        with open("model.html", "w") as f:
            f.write(iframe)
        ipd.display(ipd.HTML(filename="model.html"))

    return vis


def show_bird():
    m = stormvogel.model.new_dtmc(create_initial_state=False)
    m.new_state("🐦")
    m.add_self_loops()
    return show(
        m, show_editor=False, do_init_server=False, layout=stormvogel.layout.SV()
    )
