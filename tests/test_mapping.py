import stormvogel.examples.monty_hall
import stormvogel.examples.stormpy_mdp
import stormvogel.stormpy_utils.mapping as mapping
import stormvogel.model
import stormvogel.examples.stormpy_dtmc
import stormvogel.examples.die
import stormvogel.examples.stormpy_ctmc
import stormvogel.examples.stormpy_pomdp
import stormvogel.examples.nuclear_fusion_ctmc
import stormvogel.examples.monty_hall_pomdp
import stormvogel.examples.stormpy_ma
import stormvogel.examples.simple_ma

try:
    import stormpy
except ImportError:
    stormpy = None


def sparse_equal(
    m0: stormpy.storage.SparseDtmc | stormpy.storage.SparseMdp,
    m1: stormpy.storage.SparseDtmc | stormpy.storage.SparseMdp,
) -> bool:
    """
    outputs true if the sparse models are the same and false otherwise
    Note: this function is only here because the equality functions in storm do not work currently.
    """
    assert stormpy is not None

    # check if states are the same:
    states_equal = True
    for i in range(m0.nr_states):
        actions_equal = True
        for j in range(len(m0.states[i].actions)):
            if not m0.states[i].actions[j] == m1.states[i].actions[j]:
                actions_equal = True
        if not (
            m0.states[i].id == m1.states[i].id
            and m0.states[i].labels == m1.states[i].labels
            and actions_equal
        ):
            states_equal = False

    # check if the matrices are the same:
    # TODO check for semantic equivalence and not just syntactic
    matrices_equal = str(m0.transition_matrix) == str(m1.transition_matrix)

    # check if model types are equal:
    types_equal = m0.model_type == m1.model_type

    # check if reward models are equal:
    reward_models_equal = True
    for key in m0.reward_models.keys():
        for i in range(m0.nr_states):
            if (
                m0.reward_models[key].has_state_rewards
                and m1.reward_models[key].has_state_rewards
            ):
                if not m0.reward_models[key].get_state_reward(i) == m1.reward_models[
                    key
                ].get_state_reward(i):
                    reward_models_equal = False
            if (
                m0.reward_models[key].has_state_action_rewards
                and m1.reward_models[key].has_state_action_rewards
            ):
                if not m0.reward_models[key].get_state_action_reward(
                    i
                ) == m1.reward_models[key].get_state_action_reward(i):
                    reward_models_equal = False

    # check if exit rates are equal (in case of ctmcs):
    exit_rates_equal = (
        not m0.model_type == stormpy.ModelType.CTMC or m0.exit_rates == m1.exit_rates
    )

    # check if observations are equal (in case of pomdps):
    observations_equal = (
        not m0.model_type == stormpy.ModelType.POMDP
        or m0.observations == m1.observations
    )

    # check if markovian states are equal (in case of mas):
    markovian_states_equal = (
        not m0.model_type == stormpy.ModelType.MA
        or m0.markovian_states == m1.markovian_states
    )

    return (
        matrices_equal
        and types_equal
        and states_equal
        and reward_models_equal
        and exit_rates_equal
        and observations_equal
        and markovian_states_equal
    )


def test_stormpy_to_stormvogel_and_back_dtmc():
    # we test it for an example stormpy representation of a dtmc
    stormpy_dtmc = stormvogel.examples.stormpy_dtmc.example_building_dtmcs_01()
    # print(stormpy_dtmc.transition_matrix)
    stormvogel_dtmc = mapping.stormpy_to_stormvogel(stormpy_dtmc)
    # print(stormvogel_dtmc)
    assert stormvogel_dtmc is not None
    new_stormpy_dtmc = mapping.stormvogel_to_stormpy(stormvogel_dtmc)
    # print(new_stormpy_dtmc.transition_matrix)

    assert sparse_equal(stormpy_dtmc, new_stormpy_dtmc)


def test_stormvogel_to_stormpy_and_back_dtmc():
    # we test it for the die dtmc
    stormvogel_dtmc = stormvogel.examples.die.create_die_dtmc()

    # we test if rewardmodels work:
    rewardmodel = stormvogel_dtmc.add_rewards("rewardmodel")
    for stateid in stormvogel_dtmc.states.keys():
        rewardmodel.rewards[(stateid, stormvogel.model.EmptyAction)] = 1
    rewardmodel2 = stormvogel_dtmc.add_rewards("rewardmodel2")
    for stateid in stormvogel_dtmc.states.keys():
        rewardmodel2.rewards[(stateid, stormvogel.model.EmptyAction)] = 2

    # print(stormvogel_dtmc)
    stormpy_dtmc = mapping.stormvogel_to_stormpy(stormvogel_dtmc)
    # print(stormpy_dtmc)
    new_stormvogel_dtmc = mapping.stormpy_to_stormvogel(stormpy_dtmc)
    # print(new_stormvogel_dtmc)

    assert new_stormvogel_dtmc == stormvogel_dtmc


def test_stormpy_to_stormvogel_and_back_mdp():
    # we test it for an example stormpy representation of an mdp
    stormpy_mdp = stormvogel.examples.stormpy_mdp.example_building_mdps_01()
    # print(stormpy_mdp)
    stormvogel_mdp = mapping.stormpy_to_stormvogel(stormpy_mdp)
    # print(stormvogel_mdp)
    assert stormvogel_mdp is not None
    new_stormpy_mdp = mapping.stormvogel_to_stormpy(stormvogel_mdp)
    # print(new_stormpy_mdp)

    assert sparse_equal(stormpy_mdp, new_stormpy_mdp)


def test_stormvogel_to_stormpy_and_back_mdp():
    # we test it for monty hall mdp
    stormvogel_mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

    # we additionally test if reward models work
    rewardmodel = stormvogel_mdp.add_rewards("rewardmodel")
    rewardmodel.set_from_rewards_vector(list(range(67)))
    rewardmodel2 = stormvogel_mdp.add_rewards("rewardmodel2")
    rewardmodel2.set_from_rewards_vector(list(range(67)))

    # print(stormvogel_mdp)
    stormpy_mdp = mapping.stormvogel_to_stormpy(stormvogel_mdp)
    # print(stormpy_mdp)
    new_stormvogel_mdp = mapping.stormpy_to_stormvogel(stormpy_mdp)

    # rew0 = sorted(stormvogel_mdp.rewards)[0].rewards
    # rew1 = sorted(new_stormvogel_mdp.rewards)[0].rewards
    # for k,v in rew0.items():
    #     if v != rew1[k]:
    #         print("original:", v, "\tsv:", rew1[k])
    # print(rew0 == rew1)
    # quit()

    # print(new_stormvogel_mdp)
    # print(sorted(stormvogel_mdp.rewards)[0] == sorted(new_stormvogel_mdp.rewards)[0])
    # quit()
    # print(sorted(stormvogel_mdp.rewards)[0].rewards)
    # print()
    # print(sorted(new_stormvogel_mdp.rewards)[0].rewards)
    # print(sorted(new_stormvogel_mdp.rewards)[0].rewards == sorted(stormvogel_mdp.rewards)[0].rewards)
    # print(sorted(new_stormvogel_mdp.rewards)[0] == sorted(stormvogel_mdp.rewards)[0])
    # print(new_stormvogel_mdp == stormvogel_mdp)
    # quit()

    assert new_stormvogel_mdp == stormvogel_mdp


def test_stormvogel_to_stormpy_and_back_ctmc():
    # we create a stormpy representation of an example ctmc
    stormvogel_ctmc = (
        stormvogel.examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    )
    # print(stormvogel_ctmc)
    stormpy_ctmc = mapping.stormvogel_to_stormpy(stormvogel_ctmc)
    # print(stormpy_ctmc)
    new_stormvogel_ctmc = mapping.stormpy_to_stormvogel(stormpy_ctmc)
    # print(new_stormvogel_ctmc)

    assert new_stormvogel_ctmc == stormvogel_ctmc


def test_stormpy_to_stormvogel_and_back_ctmc():
    # we create a stormpy representation of an example ctmc
    stormpy_ctmc = stormvogel.examples.stormpy_ctmc.example_building_ctmcs_01()
    # print(stormpy_ctmc)
    stormvogel_ctmc = mapping.stormpy_to_stormvogel(stormpy_ctmc)
    # print(stormvogel_ctmc)
    assert stormvogel_ctmc is not None
    new_stormpy_ctmc = mapping.stormvogel_to_stormpy(stormvogel_ctmc)
    # print(new_stormpy_ctmc)

    assert sparse_equal(stormpy_ctmc, new_stormpy_ctmc)


def test_stormvogel_to_stormpy_and_back_pomdp():
    # we create a stormpy representation of an example pomdp
    stormvogel_pomdp = stormvogel.examples.monty_hall_pomdp.create_monty_hall_pomdp()
    # print(stormvogel_pomdp)
    stormpy_pomdp = mapping.stormvogel_to_stormpy(stormvogel_pomdp)
    # print(stormpy_pomdp)
    new_stormvogel_pomdp = mapping.stormpy_to_stormvogel(stormpy_pomdp)
    # print(new_stormvogel_pomdp)
    # print(stormvogel_pomdp.actions)
    # print()
    # print(new_stormvogel_pomdp.actions)
    # print(stormvogel_pomdp.actions == new_stormvogel_pomdp.actions)

    assert new_stormvogel_pomdp == stormvogel_pomdp


def test_stormpy_to_stormvogel_and_back_pomdp():
    # we create a stormpy representation of an example pomdp
    stormpy_pomdp = stormvogel.examples.stormpy_pomdp.example_building_pomdps_01()
    # print(stormpy_pomdp)
    stormvogel_pomdp = mapping.stormpy_to_stormvogel(stormpy_pomdp)
    # print(stormvogel_pomdp)
    assert stormvogel_pomdp is not None
    new_stormpy_pomdp = mapping.stormvogel_to_stormpy(stormvogel_pomdp)
    # print(new_stormpy_pomdp)

    assert sparse_equal(stormpy_pomdp, new_stormpy_pomdp)


# TODO for some reason, this test crashes but only in Github workflows?
# def test_stormvogel_to_stormpy_and_back_ma():
#     # we create a stormpy representation of an example ma
#     stormvogel_ma = stormvogel.examples.simple_ma.create_simple_ma()
#     # print(stormvogel_ma)
#     stormpy_ma = mapping.stormvogel_to_stormpy(stormvogel_ma)
#     # print(stormpy_ma)
#     new_stormvogel_ma = mapping.stormpy_to_stormvogel(stormpy_ma)
#     # print(new_stormvogel_ma)

#     assert new_stormvogel_ma == stormvogel_ma


def test_stormpy_to_stormvogel_and_back_ma():
    # we create a stormpy representation of an example ma
    stormpy_ma = stormvogel.examples.stormpy_ma.example_building_mas_01()
    # print(stormpy_ma)
    stormvogel_ma = mapping.stormpy_to_stormvogel(stormpy_ma)
    # print(stormvogel_ma)
    assert stormvogel_ma is not None
    new_stormpy_ma = mapping.stormvogel_to_stormpy(stormvogel_ma)
    # print(new_stormpy_ma)

    assert sparse_equal(stormpy_ma, new_stormpy_ma)
