# Adapted from Coldstart Coder on Medium (2024)
# https://medium.com/@coldstart_coder/visually-rendering-python-gymnasium-in-jupyter-notebooks-4413e4087a0f

from pyvirtualdisplay import Display  # type: ignore

from stormvogel import pgc  # type: ignore

# Do not move these, it will break
display = Display(visible=False, size=(1400, 900))  # type: ignore
display.start()

from typing import Callable  # noqa: E402
from gymnasium.wrappers import RecordVideo  # noqa: E402
import io  # noqa: E402
import base64  # noqa: E402
from IPython import display  # noqa: E402
from IPython.display import HTML  # noqa: E402

import stormvogel.result  # noqa: E402
import stormvogel.model  # noqa: E402
import math  # noqa: E402


def embed_video(video_file):
    # open and read the raw data from the video
    video_data = io.open(video_file, "r+b").read()
    # now we have to encode the data into base64 to work
    # with the virtual display
    encoded_data = base64.b64encode(video_data)
    # now we use the display.display function to take some html
    # and the encoded data and embed the html into the notebook!
    display.display(
        HTML(
            data="""<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>""".format(encoded_data.decode("ascii"))
        )
    )  # type: ignore


def create_video(
    env, location, choose_action: Callable | None = None, limit: int = 20
) -> str:
    """Create a video of the current environment using the choose_action function to choose an action.
    Args:
        env: The gymnasium env
        location: Path where the video should be stored
        choose_action: function that chooses the action. Choose a random action if left as None.
        limit: limits the amount of steps. Defaults to 20

    Returns: (relative) path to video.
    """
    env = RecordVideo(env, location)
    state, inf = env.reset()
    while limit > 0:
        # render the frame, this will save it to the video file
        env.render()
        # pick a random action
        if choose_action is None:
            action = env.action_space.sample()
        else:
            action = choose_action(state)
        # run the action
        state, reward, terminated, truncated, info = env.step(action)
        # if the simulation has ended break the loop
        if terminated:
            break
        limit -= 1
    env.close()
    return location + "/rl-video-episode-0.mp4"


DIR_MAP = {"←": 0, "↓": 1, "→": 2, "↑": 3, "pickup": 4, "dropoff": 5}
DIR_MAP_INV = {v: k for k, v in DIR_MAP.items()}
ALL_ACTIONS = [pgc.Action([k]) for k in DIR_MAP]


def to_coordinate(s, env):
    """Calculate the state's coordinates. Works for FrozenLake, Cliffwalking, and Taxi"""
    num_states = env.observation_space.n
    grid_size = int(math.sqrt(num_states))
    x_target = int(s % grid_size)
    y_target = int(s // grid_size)
    return x_target, y_target


def to_state(x, y, env):
    """Calculate the state index from coordinates. Works for FrozenLake, CliffWalking, and Taxi."""
    num_states = env.observation_space.n
    grid_size = int(math.sqrt(num_states))
    return y * grid_size + x


def create_video_scheduler(
    model: stormvogel.model.Model,
    env,
    location: str,
    scheduler: stormvogel.result.Scheduler
    | Callable[[stormvogel.model.State], stormvogel.model.Action]
    | None = None,
) -> str:
    """Create a video of the current environment using the scheduler to choose an action.
    Args:
        env: The gymnasium env
        location: Path where the video should be stored
        scheduler: A function that chooses the action, or None

    Returns: (relative) path to video.
    """
    if scheduler is None:
        return create_video(env, location, None)

    def choose_action(env_sid: int):
        # TODO change this once pgc API features are a thing.
        model_state = model.get_states_with_label(str(env_sid))[0]
        if isinstance(scheduler, stormvogel.result.Scheduler):
            choice = scheduler.get_choice_of_state(model_state)
        elif callable(scheduler):
            choice = scheduler(model_state)  # type: ignore
        return DIR_MAP[list(choice.labels)[0]]

    return create_video(env, location, choose_action)


def gymnasium_to_stormvogel(env):
    """Convert a FrozenLake, Taxi, or Cliffwalking gymnasium environment to an explicit stormvogel model."""
    transitions = env.unwrapped.P
    no_actions = env.action_space.n

    def action_numer_map(a):
        return DIR_MAP[a.labels[0]]

    init = 0

    def available_actions(_):
        return ALL_ACTIONS[:no_actions]

    def delta(s, a):
        return list(map(lambda x: x[:2], transitions[s][action_numer_map(a)]))

    def rewards(s, a):
        reward = list(map(lambda x: x[2], transitions[s][action_numer_map(a)]))[0]
        return {"R": reward}

    def labels(s):
        labels = [str(s), str(to_coordinate(s, env))]
        if s == get_target_state(env):
            labels.append("target")
        return labels

    return pgc.build_pgc(
        delta=delta,
        initial_state_pgc=init,
        available_actions=available_actions,
        labels=labels,
        rewards=rewards,
        modeltype=stormvogel.model.ModelType.MDP,
    )


def get_target_state(env):
    """Calculate the target state for an env. Works for FrozenLake and Cliffwalking"""
    return env.observation_space.n - 1
