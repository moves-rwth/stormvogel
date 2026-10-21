"""MEC detection and elimination for MDPs.

Uses stormpy for the heavy lifting; exposes a stormvogel-native API.
"""

try:
    import stormpy
except ImportError:
    stormpy = None  # type: ignore[assignment]

import stormvogel.model as model
from stormvogel.model.distribution import Distribution
from stormvogel.stormpy_utils import mapping


def _sv_state_name(s: model.State, sp_id: int) -> str:
    return s.friendly_name if s.friendly_name is not None else str(sp_id)


def detect_mecs(
    mdp: model.Model,
) -> list[frozenset[model.State]]:
    """Return the maximal end components of *mdp* as frozensets of stormvogel states.

    Each MEC is returned as a :class:`frozenset` of :class:`~stormvogel.model.State`
    objects from the original model.  Trivial MECs (single states with a
    self-loop only) are included.

    :param mdp: A stormvogel MDP.
    :returns: List of MECs, each represented as a frozenset of states.
    :raises RuntimeError: If stormpy is not available.
    """
    if stormpy is None:
        raise RuntimeError("stormpy is required for MEC detection.")

    sp_model = mapping.stormvogel_to_stormpy(mdp)
    sp_id_to_sv: dict[int, model.State] = {i: s for s, i in mdp.stormpy_id.items()}

    return [
        frozenset(sp_id_to_sv[s_id] for s_id, _ in mec)
        for mec in stormpy.analysis.get_maximal_end_components(sp_model)
    ]


def eliminate_mecs(
    mdp: model.Model,
    remove_representative_selfloops: bool = False,
    make_representatives_absorbing: bool = False,
) -> tuple[model.Model, dict[model.State, model.State]]:
    """Eliminate all end components from *mdp* and return the resulting model.

    Each MEC is collapsed to a single representative state that inherits the
    union of labels from all states that were merged into it.

    Two mutually exclusive post-processing options control how representative
    states (those produced by merging a non-trivial MEC) are treated:

    - ``remove_representative_selfloops=True``: strip the stormpy-added
      self-loop from any representative that still has exit actions.  Useful
      for *maximum* reachability IVI so the upper bound can descend.
    - ``make_representatives_absorbing=True``: strip all exit actions from
      each representative, leaving only the self-loop.  Useful for *minimum*
      reachability: the representative becomes a trap (value 0), reflecting
      that the scheduler can choose to stay in the MEC forever.

    :param mdp: A stormvogel MDP.
    :param remove_representative_selfloops: Remove self-loop choices from
        representative states that have at least one exit action.
    :param make_representatives_absorbing: Remove all non-self-loop choices
        from representative states, making them absorbing sinks.
    :returns: A pair ``(new_mdp, state_map)`` where *state_map* maps every
        state of *mdp* to its corresponding state in *new_mdp*.
    :raises RuntimeError: If stormpy is not available.
    :raises ValueError: If both post-processing options are requested.
    """
    if remove_representative_selfloops and make_representatives_absorbing:
        raise ValueError(
            "remove_representative_selfloops and make_representatives_absorbing "
            "are mutually exclusive."
        )
    if stormpy is None:
        raise RuntimeError("stormpy is required for MEC elimination.")

    # Storm's API exposes MEC decomposition, but no longer a matrix-level
    # eliminator. Build the quotient directly and retain the original state map.
    mecs = detect_mecs(mdp)
    containing = {state: mec for mec in mecs for state in mec}
    quotient = model.new_mdp(create_initial_state=False)
    for parameter in mdp.parameters:
        quotient.declare_parameter(parameter)
    state_map: dict[model.State, model.State] = {}
    groups: list[list[model.State]] = []
    state_order = {state: index for index, state in enumerate(mdp.states)}
    for state in mdp.states:
        if state in state_map:
            continue
        group = (
            sorted(containing[state], key=state_order.__getitem__)
            if state in containing
            else [state]
        )
        labels = sorted({label for s in group for label in s.labels})
        representative = quotient.new_state(labels)
        if len(group) > 1:
            representative.set_friendly_name(f"_mec_{sum(len(g) > 1 for g in groups)}")
        elif state.friendly_name is not None:
            representative.set_friendly_name(state.friendly_name)
        for old in group:
            state_map[old] = representative
        groups.append(group)

    for group in groups:
        representative = state_map[group[0]]
        in_mec = group[0] in containing
        merged = len(group) > 1
        choices = {}
        for old in group:
            for action, distribution in mdp.transitions.get(old, []):
                # Internal actions are replaced by the representative's stay action.
                if in_mec and distribution.support <= set(group):
                    continue
                label = action.label
                if merged:
                    label = f"{label if label is not None else 'exit'}_{_sv_state_name(old, mdp.get_state_index(old))}"
                new_action = quotient.action(label) if label is not None else action
                choices[new_action] = Distribution(
                    [
                        (probability, state_map[successor])
                        for probability, successor in distribution
                    ]
                )
        if in_mec:
            if merged and make_representatives_absorbing:
                choices.clear()
            if not (merged and remove_representative_selfloops and choices):
                # Preserve the label of a trivial sink's sole action.
                stay = quotient.action("stay")
                if not merged:
                    for action, distribution in mdp.transitions.get(group[0], []):
                        if distribution.support <= set(group):
                            stay = action
                            break
                choices[stay] = Distribution([(1.0, representative)])
        representative.set_choices(choices)

    return quotient, state_map
