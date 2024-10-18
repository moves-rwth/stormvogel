import stormvogel.model
import examples.die
import examples.nuclear_fusion_ctmc
import stormvogel.simulator


def test_simulate():
    # we make a die dtmc and run the simulator with it
    dtmc = examples.die.create_die_dtmc()
    rewardmodel = dtmc.add_rewards("rewardmodel")
    for stateid in dtmc.states.keys():
        rewardmodel.rewards[stateid] = 3
    rewardmodel2 = dtmc.add_rewards("rewardmodel2")
    for stateid in dtmc.states.keys():
        rewardmodel2.rewards[stateid] = 2
    rewardmodel3 = dtmc.add_rewards("rewardmodel3")
    for stateid in dtmc.states.keys():
        rewardmodel3.rewards[stateid] = 1
    partial_model = stormvogel.simulator.simulate(dtmc, runs=5, steps=1, seed=1)

    # we make the partial model that should be created by the simulator
    other_dtmc = stormvogel.model.new_dtmc()
    other_dtmc.new_state(labels=["rolled5"])
    other_dtmc.new_state(labels=["rolled0"])
    other_dtmc.new_state(labels=["rolled1"])

    rewardmodel = other_dtmc.add_rewards("rewardmodel")
    for stateid in other_dtmc.states.keys():
        rewardmodel.rewards[stateid] = float(3)
    rewardmodel2 = other_dtmc.add_rewards("rewardmodel2")
    for stateid in other_dtmc.states.keys():
        rewardmodel2.rewards[stateid] = float(2)
    rewardmodel3 = other_dtmc.add_rewards("rewardmodel3")
    for stateid in other_dtmc.states.keys():
        rewardmodel3.rewards[stateid] = float(1)

    assert partial_model == other_dtmc

    # we make a monty hall mdp and run the simulator with it

    # we make the partial model that should be created by the simulator


def test_simulate_path():
    # we make the nuclear fusion ctmc and run simulate path with it
    ctmc = examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    path = stormvogel.simulator.simulate_path(ctmc, steps=5, seed=1)

    other_path = stormvogel.simulator.Path(
        {
            1: ctmc.get_state_by_id(1),
            2: ctmc.get_state_by_id(2),
            3: ctmc.get_state_by_id(3),
            4: ctmc.get_state_by_id(4),
        },
        ctmc,
    )

    # print(path, other_path)

    assert str(path) == str(other_path)

    # we make the path that the simulate path function should create

    # we make the monty hall pomdp and run simulate path with it
