import stormvogel.model

dtmc = stormvogel.model.new_dtmc("Die")

init = dtmc.get_initial_state()

# roll die
init.set_transitions(
    [(1 / 6, dtmc.new_state(f"rolled{i}", {"rolled": i})) for i in range(6)]
)

print(dtmc.to_dot())
