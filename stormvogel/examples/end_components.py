from stormvogel import pgc
from stormvogel.model import ModelType


def create_end_components_mdp():
    init = pgc.State(x=["init"])

    def available_actions(s: pgc.State):
        if "init" in s.x:
            return [["one"], ["two"]]
        elif "mec1" in s.x or "mec2" in s.x:
            return [["one"], ["two"]]
        return [[]]

    def delta(s: pgc.State, a: pgc.Action):
        if "init" in s.x and "one" in a.labels:
            return [(0.5, pgc.State(x=["mec1"])), (0.5, pgc.State(x=["mec2"]))]
        elif "mec1" in s.x:
            return [(1, pgc.State(x=["mec2"]))]
        elif "mec2" in s.x:
            return [(1, pgc.State(x=["mec1"]))]
        elif "init" in s.x and "two" in a.labels:
            return [(1, pgc.State(x=["mec1"]))]
        return [(1, s)]

    def labels(s):
        return s.x

    mdp = pgc.build_pgc(
        delta=delta,
        initial_state_pgc=init,
        available_actions=available_actions,
        labels=labels,
        modeltype=ModelType.MDP,
    )
    return mdp
