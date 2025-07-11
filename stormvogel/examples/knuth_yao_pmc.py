from stormvogel import parametric
from stormvogel import model, pgc
from stormvogel.show import show


def create_knuth_yao_pmc():
    #we first make polynomials 'x' and '1-x'
    x = parametric.Polynomial(["x"])
    x.add_term((1,),1)

    invx = parametric.Polynomial(["x"])
    invx.add_term((1,),-1)
    invx.add_term((0,),1)

    #we build the knuth yao dice using the pgc model builder
    def delta(s: pgc.State):
        match s.s:
            case 0:
                return [(x, pgc.State(s=1)), (invx, pgc.State(s=2))]
            case 1:
                return [(x, pgc.State(s=3)), (invx, pgc.State(s=4))]
            case 2:
                return [(x, pgc.State(s=5)), (invx, pgc.State(s=6))]
            case 3:
                return [(x, pgc.State(s=1)), (invx, pgc.State(s=7, d=1))]
            case 4:
                return [
                    (x, pgc.State(s=7, d=2)),
                    (invx, pgc.State(s=7, d=3)),
                ]
            case 5:
                return [
                    (x, pgc.State(s=7, d=4)),
                    (invx, pgc.State(s=7, d=5)),
                ]
            case 6:
                return [(x, pgc.State(s=2)), (invx, pgc.State(s=7, d=6))]
            case 7:
                return [(1, s)]

    knuth_yao = pgc.build_pgc(
        delta=delta,
        initial_state_pgc=pgc.State(s=0),
        modeltype=model.ModelType.DTMC,
    )