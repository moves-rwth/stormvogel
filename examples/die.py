import stormvogel.model

# Create a new model with the name "Die"
dtmc = stormvogel.model.new_dtmc("Die")

init = dtmc.get_initial_state()

# From the initial state, add the transition to 6 new states with probability 1/6th.
init.set_transitions(
    [(1 / 6, dtmc.new_state(f"rolled{i}", {"rolled": i})) for i in range(6)]
)

# Print the resulting model in dot format.
print(dtmc.to_dot())
