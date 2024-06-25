import stormvogel.model

mdp = stormvogel.model.new_mdp("Monty Hall")

init = mdp.init()

# first choose car position
init.add_transitions([
    (1/3, mdp.new_state("carchosen", {"car_pos": i})) for i in range(3)
])

# we choose a door in each case
for s in mdp.get_states_with("carchosen"):
    s.add_transitions([
        (mdp.new_action("open", i), mdp.new_state("open", s.properties | {"chosen_pos": i})) for i in range(3)
    ])

# the other goat is revealed
for s in mdp.states_with("open"):
    car_pos = s.properties["car_pos"]
    chosen_pos = s.properties["chosen_pos"]
    other_pos = {0, 1, 2} - {car_pos, chosen_pos}
    s.add_transitions([
        (1/len(other_pos), mdp.state("goatrevealed", s.properties | {"reveal_pos": i})) for i in other_pos
    ])

# we must choose whether we want to switch
for s in mdp.states_with("goatrevealed"):
    car_pos = s.properties["car_pos"]
    chosen_pos = s.properties["chosen_pos"]
    reveal_pos = s.properties["reveal_pos"]
    other_pos = ({0, 1, 2} - {car_pos, chosen_pos})[0]
    s.add_transitions([
        (mdp.action("stay"),   mdp.state("done", s.properties | {"chosen_pos": chosen_pos, "target": chosen_pos == car_pos})),
        (mdp.action("switch"), mdp.state("done", s.properties | {"chosen_pos": other_pos,  "target": chosen_pos == car_pos}))
    ])
