import stormvogel.examples.pmc


def test_pmc():
    dtmc = stormvogel.examples.pmc.create_die_dtmc()

    # TODO valuations

    print(dtmc)
