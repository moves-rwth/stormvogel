import examples.die
import examples.monty_hall
import examples.nuclear_fusion_ctmc
import examples.monty_hall_pomdp
from stormvogel.model import EmptyAction
import stormvogel.model
import stormvogel.simulator


def test_simulate():
    # we make a die dtmc and run the simulator with it
    dtmc = examples.die.create_die_dtmc()
    rewardmodel = dtmc.add_rewards("rewardmodel")
    for stateid in dtmc.states.keys():
        rewardmodel.rewards[(stateid, EmptyAction)] = 3
    rewardmodel2 = dtmc.add_rewards("rewardmodel2")
    for stateid in dtmc.states.keys():
        rewardmodel2.rewards[(stateid, EmptyAction)] = 2
    rewardmodel3 = dtmc.add_rewards("rewardmodel3")
    for stateid in dtmc.states.keys():
        rewardmodel3.rewards[(stateid, EmptyAction)] = 1
    partial_model = stormvogel.simulator.simulate(dtmc, runs=5, steps=1, seed=1)

    # we make the partial model that should be created by the simulator
    other_dtmc = stormvogel.model.new_dtmc()
    other_dtmc.get_initial_state().set_transitions(
        [
            (1 / 6, other_dtmc.new_state("rolled5")),
            (1 / 6, other_dtmc.new_state("rolled0")),
            (1 / 6, other_dtmc.new_state("rolled1")),
        ]
    )

    rewardmodel = other_dtmc.add_rewards("rewardmodel")
    for stateid in other_dtmc.states.keys():
        rewardmodel.rewards[(stateid, EmptyAction)] = float(3)
    rewardmodel2 = other_dtmc.add_rewards("rewardmodel2")
    for stateid in other_dtmc.states.keys():
        rewardmodel2.rewards[(stateid, EmptyAction)] = float(2)
    rewardmodel3 = other_dtmc.add_rewards("rewardmodel3")
    for stateid in other_dtmc.states.keys():
        rewardmodel3.rewards[(stateid, EmptyAction)] = float(1)

    assert partial_model == other_dtmc
    ######################################################################################################################
    # we make a monty hall mdp and run the simulator with it
    mdp = examples.monty_hall.create_monty_hall_mdp()
    rewardmodel = mdp.add_rewards("rewardmodel")
    rewardmodel.set_from_rewards_vector(list(range(67)))
    rewardmodel2 = mdp.add_rewards("rewardmodel2")
    rewardmodel2.set_from_rewards_vector(list(range(67)))

    taken_actions = {}
    for id, state in mdp.states.items():
        taken_actions[id] = state.available_actions()[0]
    scheduler = stormvogel.result.Scheduler(mdp, taken_actions)

    partial_model = stormvogel.simulator.simulate(
        mdp, runs=1, steps=3, seed=1, scheduler=scheduler
    )

    # we make the partial model that should be created by the simulator
    other_mdp = stormvogel.model.new_mdp()
    other_mdp.get_initial_state().set_transitions(
        [(1 / 3, other_mdp.new_state("carchosen"))]
    )
    other_mdp.get_state_by_id(1).set_transitions([(1, other_mdp.new_state("open"))])
    other_mdp.get_state_by_id(2).set_transitions(
        [(1, other_mdp.new_state("goatrevealed"))]
    )
    action0 = other_mdp.new_action()
    action1 = other_mdp.new_action("open0")
    action2 = other_mdp.new_action()

    rewardmodel = other_mdp.add_rewards("rewardmodel")
    rewardmodel.rewards = {(0, action0): 0, (3, action1): 7, (10, action2): 16}
    rewardmodel2 = other_mdp.add_rewards("rewardmodel2")
    rewardmodel2.rewards = {(0, action0): 0, (3, action1): 7, (10, action2): 16}

    assert partial_model == other_mdp
    ######################################################################################################################

    # we test the simulator for an mdp with a lambda as Scheduler

    def scheduler(state: stormvogel.model.State) -> stormvogel.model.Action:
        actions = state.available_actions()
        return actions[0]

    mdp = examples.monty_hall.create_monty_hall_mdp()

    partial_model = stormvogel.simulator.simulate(
        mdp, runs=1, steps=3, seed=1, scheduler=scheduler
    )

    # we make the partial model that should be created by the simulator
    other_mdp = stormvogel.model.new_mdp()
    other_mdp.get_initial_state().set_transitions(
        [(1 / 3, other_mdp.new_state("carchosen"))]
    )
    other_mdp.get_state_by_id(1).set_transitions([(1, other_mdp.new_state("open"))])
    other_mdp.get_state_by_id(2).set_transitions(
        [(1, other_mdp.new_state("goatrevealed"))]
    )

    assert partial_model == other_mdp


# TODO Pim could you also finish this test? :)
# def test_simulate_path():
#     # we make the nuclear fusion ctmc and run simulate path with it
#     ctmc = examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
#     path = stormvogel.simulator.simulate_path(ctmc, steps=5, seed=1)

#     # we make the path that the simulate path function should create
#     other_path = stormvogel.simulator.Path(
#         {
#             1: ctmc.get_state_by_id(1),
#             2: ctmc.get_state_by_id(2),
#             3: ctmc.get_state_by_id(3),
#             4: ctmc.get_state_by_id(4),
#         },
#         ctmc,
#     )

#     assert path == other_path
#     ##############################################################################################
#     # we make the monty hall pomdp and run simulate path with it
#     pomdp = examples.monty_hall_pomdp.create_monty_hall_pomdp()
#     taken_actions = {}
#     for id, state in pomdp.states.items():
#         taken_actions[id] = state.available_actions()[
#             len(state.available_actions()) - 1
#         ]
#     scheduler = stormvogel.result.Scheduler(pomdp, taken_actions)
#     path = stormvogel.simulator.simulate_path(
#         pomdp, steps=4, seed=1, scheduler=scheduler
#     )

#     # we make the path that the simulate path function should create
#     other_path = stormvogel.simulator.Path(
#         {
#             1: (stormvogel.model.EmptyAction, pomdp.get_state_by_id(3)),
#             2: (pomdp.actions["open0"], pomdp.get_state_by_id(10)),
#             3: (stormvogel.model.EmptyAction, pomdp.get_state_by_id(21)),
#             4: (pomdp.actions["stay"], pomdp.get_state_by_id(41)),
#         },
#         pomdp,
#     )

#     assert path == other_path

#     ##############################################################################################
#     # we test the monty hall pomdp with a lambda as scheduler
#     def scheduler(state: stormvogel.model.State) -> stormvogel.model.Action:
#         actions = state.available_actions()
#         return actions[0]

#     pomdp = examples.monty_hall_pomdp.create_monty_hall_pomdp()
#     path = stormvogel.simulator.simulate_path(
#         pomdp, steps=4, seed=1, scheduler=scheduler
#     )

#     # we make the path that the simulate path function should create
#     other_path = stormvogel.simulator.Path(
#         {
#             1: (stormvogel.model.EmptyAction, pomdp.get_state_by_id(3)),
#             2: (pomdp.actions["open0"], pomdp.get_state_by_id(10)),
#             3: (stormvogel.model.EmptyAction, pomdp.get_state_by_id(21)),
#             4: (pomdp.actions["stay"], pomdp.get_state_by_id(41)),
#         },
#         pomdp,
#     )

#     assert path == other_path
