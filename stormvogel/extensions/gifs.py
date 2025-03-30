import stormvogel.simulator
import stormvogel.model
from typing import Callable
import imageio
from PIL.Image import Image
import IPython.display as ipd


def render_model_gif(
    model: stormvogel.model.Model,
    state_to_image: Callable[[stormvogel.model.State], Image],
    scheduler: stormvogel.result.Scheduler
    | Callable[[stormvogel.model.State], stormvogel.model.Action],
    path: stormvogel.simulator.Path | None = None,
    filename: str = "my_gif",
    max_length: int = 50,
    fps: int = 2,
    loop: int = 0,
) -> str:
    """Render a gymnasium model to a gif, using the gymnasium_scheduler (A map from state numbers to action numbers) to pick an action.
    Leave as None for a random action."""
    if model.supports_actions():
        path = stormvogel.simulator.simulate_path(model, max_length, scheduler)
    else:
        path = stormvogel.simulator.simulate_path(model, max_length)
    frames = [state_to_image(model.get_initial_state())]  # List to store frames

    for i in range(1, min(max_length, len(path) + 1)):
        state = path.get_step(i)
        if not isinstance(state, stormvogel.model.State):
            state = state[i]
        frames.append(state_to_image(state))  # type: ignore

    # Save frames as a GIF
    imageio.mimsave(
        filename + ".gif",
        frames,  # type: ignore
        fps=fps,
        loop=loop,
    )  # type: ignore
    return filename + ".gif"


def embed_gif(filename: str):
    with open("GIF" + ".html", "w") as f:
        f.write(f'<img src="{filename}">')
    ipd.display(ipd.HTML(filename="GIF" + ".html"))
    # ipd.display(ipd.HTML(f""" <img src="{filename}" onerror="this.onerror=null; this.src='docs/_build/html/_static/my_gif.gif';"> """))
