# Adapted from Coldstart Coder on Medium (2024)
# https://medium.com/@coldstart_coder/visually-rendering-python-gymnasium-in-jupyter-notebooks-4413e4087a0f

from pyvirtualdisplay import Display  # type: ignore
import warnings
from stormvogel import pgc

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
    env,
    location,
    choose_action: Callable | None = None,
    limit: int = 20,
    initial_state: int = 0,
) -> str:
    """Create a video of the current environment using the choose_action function to choose an action.
    Args:
        env: The gymnasium env
        location: Path where the video should be stored
        choose_action: function that chooses the action. Choose a random action if left as None.
        limit: limits the amount of steps. Defaults to 20

    Returns: (relative) path to video.
    """
    with warnings.catch_warnings(
        action="ignore"
    ):  # Remove annoying warning that we are overwriting; this is intended.
        env = RecordVideo(env, location)
    state, _ = env.reset()
    env.unwrapped.s = initial_state
    state = initial_state
    while limit > 0:
        # render the frame, this will save it to the video file
        env.render()
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
    limit: int = 20,
    initial_state: int = 0,
) -> str:
    """Create a video of the current environment using the scheduler to choose an action.
    Args:
        model: A stormvogel model.
        env: The gymnasium env
        location: Path where the video should be stored
        scheduler: A stormvogel scheduler or a function that chooses the action.
            In create_video, the function works on gymnasium environment state ids,
            however this function works on stormvogel model ids (in the specified model).
            A random action is chosen if None.
        limit: limits the amount of steps. Defaults to 20

    Return: (relative) path to video.
    """
    if scheduler is None:
        return create_video(env, location, None, limit, initial_state)

    def choose_action(env_sid: int):
        # TODO change this once pgc API features are a thing.
        model_state = model.get_states_with_label(str(int(env_sid)))[0]
        if isinstance(scheduler, stormvogel.result.Scheduler):
            choice = scheduler.get_choice_of_state(model_state)
        elif callable(scheduler):
            choice = scheduler(model_state)  # type: ignore
        return DIR_MAP[list(choice.labels)[0]]

    return create_video(env, location, choose_action, limit, initial_state)


def gymnasium_to_stormvogel(env, initial_state: int = 0):
    """Convert a FrozenLake, Taxi, or Cliffwalking gymnasium environment to an explicit stormvogel model."""
    transitions = env.unwrapped.P
    no_actions = env.action_space.n

    def action_numer_map(a):
        return DIR_MAP[a.labels[0]]

    init = pgc.State(n=initial_state, done=False)

    def available_actions(_):
        return ALL_ACTIONS[:no_actions]

    def delta(s, a):
        trans = transitions[s.n][action_numer_map(a)]
        return list(map(lambda x: (x[0], pgc.State(n=int(x[1]), done=x[3])), trans))

    def rewards(s, a):
        reward = list(map(lambda x: x[2], transitions[s.n][action_numer_map(a)]))[0]
        return {"R": reward}

    def labels(s):
        labels = [str(s.n), str(to_coordinate(s.n, env))]
        if s.n == get_target_state(env):
            labels.append("target")
        if s.done:
            labels.append("done")
        # labels.append("always")
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
