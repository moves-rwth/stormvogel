import stormvogel.model

mdp = stormvogel.model.new_mdp("Monty Hall")

init = mdp.get_initial_state()

# first choose car position
init.set_transitions(
    [(1 / 3, mdp.new_state("carchosen", {"car_pos": i})) for i in range(3)]
)

# we choose a door in each case
for s in mdp.get_states_with("carchosen"):
    s.set_transitions(
        [
            (
                mdp.action(f"open{i}"),
                mdp.new_state("open", s.features | {"chosen_pos": i}),
            )
            for i in range(3)
        ]
    )

# the other goat is revealed
for s in mdp.get_states_with("open"):
    car_pos = s.features["car_pos"]
    chosen_pos = s.features["chosen_pos"]
    other_pos = {0, 1, 2} - {car_pos, chosen_pos}
    s.set_transitions(
        [
            (
                1 / len(other_pos),
                mdp.new_state("goatrevealed", s.features | {"reveal_pos": i}),
            )
            for i in other_pos
        ]
    )

# we must choose whether we want to switch
for s in mdp.get_states_with("goatrevealed"):
    car_pos = s.features["car_pos"]
    chosen_pos = s.features["chosen_pos"]
    reveal_pos = s.features["reveal_pos"]
    other_pos = list({0, 1, 2} - {reveal_pos, chosen_pos})[0]
    s.set_transitions(
        [
            (
                mdp.action("stay"),
                mdp.new_state(
                    ["done"] + (["target"] if chosen_pos == car_pos else []),
                    s.features | {"chosen_pos": chosen_pos},
                ),
            ),
            (
                mdp.action("switch"),
                mdp.new_state(
                    ["done"] + (["target"] if other_pos == car_pos else []),
                    s.features | {"chosen_pos": other_pos},
                ),
            ),
        ]
    )

print(mdp.to_dot())
