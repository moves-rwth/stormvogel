from typing import Callable, Any
import gymnasium as gym
from collections import defaultdict
from stormvogel import pgc
import stormvogel.model


def sample_gym(
    env: gym.Env,
    no_samples: int = 10,
    sample_length: int = 20,
    gymnasium_scheduler: Callable[[Any], int] | None = None,
    convert_obs: Callable[[Any], Any] = lambda x: x,
    max_size: int = 10000,
):
    """Sample the gym environment and convert it to a Stormvogel MDP.
    In reality, gym environments are POMDPs, and gymnasium only allows us to access the observation.
    The result is an MDP where states with the same observations (and termination) are lumped together.
    Probablities are frequentist estimates. Their accuracy depends on how often each "state" is visited.

    Args:
        env (gym.Env): Gymnasium env.
        no_samples (int): Total number of samples (starting at an initial state).
            To resolve multiple initial states, a new, single initial state is added if necessary.
        sample_length (int): The maximum length of a single sample.
        gymnasium_scheduler (Callable[[any], int] | None): A function from states to action numbers.
        convert_obs (Callable[[any], any]): Converts the observations to a hashable type. You can also apply rounding here.
    """
    # First, do the sampling
    initial_states = defaultdict(lambda: 0)
    transition_counts = defaultdict(lambda: defaultdict(lambda: 0))
    transition_samples = defaultdict(lambda: 0)
    reward_sums = defaultdict(lambda: 0.0)

    for s_no in range(no_samples):
        prev_state = None
        obs, _ = env.reset()
        state = (convert_obs(obs), False)
        initial_states[state] += 1
        for _ in range(sample_length):
            action = (
                env.action_space.sample()
                if gymnasium_scheduler is None
                else gymnasium_scheduler(state)
            )
            prev_state = state
            obs, reward, terminated, truncated, info = env.step(action)
            state = (convert_obs(obs), terminated)
            transition_counts[(prev_state, action)][state] += 1
            transition_samples[(prev_state, action)] += 1
            reward_sums[(prev_state, action)] += float(reward)
            if terminated:
                break

    # Then use the pgc API to build the model
    NEW_INITIAL_STATE = "GYM_SAMPLE_INIT"
    ALL_ACTIONS = [[str(x)] for x in range(env.action_space.n)]
    INV_MAP = {a[0]: no for no, a in enumerate(ALL_ACTIONS)}

    if len(initial_states) == 1:
        (init,) = initial_states
    else:
        init = NEW_INITIAL_STATE

    def available_actions(s):
        if s is NEW_INITIAL_STATE:
            return [pgc.PgcEmptyAction]
        elif s[1]:
            return [pgc.PgcEmptyAction]
        return [a for a in ALL_ACTIONS if transition_counts[(s, INV_MAP[a[0]])]]

    def delta(s, a):
        if s is NEW_INITIAL_STATE:
            return [(count / no_samples, s_) for s_, count in initial_states.items()]
        elif s[1]:
            return [(1, s)]
        return [
            (count / transition_samples[(s, INV_MAP[a[0]])], s_)
            for s_, count in transition_counts[(s, INV_MAP[a[0]])].items()
        ]

    def rewards(s, a) -> dict[str, stormvogel.model.Number]:
        if s is NEW_INITIAL_STATE or s[1]:
            return {"R": 0}
        return {
            "R": reward_sums[s, INV_MAP[a[0]]] / transition_samples[(s, INV_MAP[a[0]])]
        }

    def labels(s):
        if s is NEW_INITIAL_STATE:
            return []
        done = ["done"] if s[1] else []
        return [str(s[0])] + done

    sv_model = pgc.build_pgc(
        delta=delta,
        initial_state_pgc=init,
        available_actions=available_actions,
        labels=labels,
        rewards=rewards,  # type: ignore
        modeltype=stormvogel.model.ModelType.MDP,
        max_size=max_size,
    )
    sv_model.name = f"`Gymnasium sample from {env.unwrapped.spec.id} with {no_samples} samples of max length {sample_length}`"  # type: ignore
    return sv_model
