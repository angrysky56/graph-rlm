def verify_barber_paradox():
    """
    Verify the Barber Paradox using propositional logic and SymPy.
    """
    try:
        from sympy import Symbol
        from sympy.logic.boolalg import Iff, Not, satisfiable

        p = Symbol("p")  # p: barber shaves himself

        axiom_prop = Iff(p, Not(p))

        sat_result = satisfiable(axiom_prop)

        fol_axiom = "forall x (Shaves(barber, x) <-> ~Shaves(x, x))"

        instantiation = "Shaves(barber, barber) <-> ~Shaves(barber, barber)"

        result = {
            "formalization_fol": fol_axiom,
            "instantiation": instantiation,
            "propositional_axiom": "P <-> ~P",
            "satisfiable": bool(sat_result),
            "models": list(sat_result) if sat_result else None,
            "conclusion": "The axiom is unsatisfiable, proving the paradox. No such barber can exist.",
        }
        return result
    except ImportError:
        return {
            "error": "SymPy not available",
            "manual_verification": "P <-> ~P is clearly a contradiction.",
            "conclusion": "Paradox confirmed.",
        }
