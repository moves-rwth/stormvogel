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

    print(partial_model)

    assert partial_model == other_dtmc

    # we make a monty hall mdp and run the simulator with it
    mdp = examples.monty_hall.create_monty_hall_mdp()
    rewardmodel = mdp.add_rewards("rewardmodel")
    for i in range(67):
        rewardmodel.rewards[i] = i
    rewardmodel2 = mdp.add_rewards("rewardmodel2")
    for i in range(67):
        rewardmodel2.rewards[i] = i

    taken_actions = {}
    for id, state in mdp.states.items():
        taken_actions[id] = state.available_actions()[0]
    scheduler = stormvogel.result.Scheduler(mdp, taken_actions)

    partial_model = stormvogel.simulator.simulate(
        mdp, runs=1, steps=3, seed=1, scheduler=scheduler
    )

    # we make the partial model that should be created by the simulator
    other_mdp = stormvogel.model.new_mdp()
    other_mdp.new_state(labels=["carchosen"])
    other_mdp.new_state(labels=["open"])
    other_mdp.new_state(labels=["goatrevealed"])

    rewardmodel = other_mdp.add_rewards("rewardmodel")
    rewardmodel.rewards = {0: 0, 7: 7, 16: 16}
    rewardmodel2 = other_mdp.add_rewards("rewardmodel2")
    rewardmodel2.rewards = {0: 0, 7: 7, 16: 16}

    assert partial_model == other_mdp


def test_simulate_path():
    # we make the nuclear fusion ctmc and run simulate path with it
    ctmc = examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    path = stormvogel.simulator.simulate_path(ctmc, steps=5, seed=1)

    # we make the path that the simulate path function should create
    other_path = stormvogel.simulator.Path(
        {
            1: ctmc.get_state_by_id(1),
            2: ctmc.get_state_by_id(2),
            3: ctmc.get_state_by_id(3),
            4: ctmc.get_state_by_id(4),
        },
        ctmc,
    )

    assert path == other_path

    # we make the monty hall pomdp and run simulate path with it
    pomdp = examples.monty_hall_pomdp.create_monty_hall_pomdp()
    taken_actions = {}
    for id, state in pomdp.states.items():
        taken_actions[id] = state.available_actions()[
            len(state.available_actions()) - 1
        ]
    scheduler = stormvogel.result.Scheduler(pomdp, taken_actions)
    path = stormvogel.simulator.simulate_path(
        pomdp, steps=4, seed=1, scheduler=scheduler
    )

    # we make the path that the simulate path function should create
    other_path = stormvogel.simulator.Path(
        {
            1: (stormvogel.model.EmptyAction, pomdp.get_state_by_id(3)),
            2: (pomdp.actions["open2"], pomdp.get_state_by_id(12)),
            3: (stormvogel.model.EmptyAction, pomdp.get_state_by_id(23)),
            4: (pomdp.actions["switch"], pomdp.get_state_by_id(46)),
        },
        pomdp,
    )

    assert path == other_path
