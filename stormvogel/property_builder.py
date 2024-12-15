import stormvogel.model
import examples.monty_hall


def build_property_string_interactive(model: stormvogel.model.Model) -> str:
    """When a model is provided, this interative property string builder will help beginner
    users to create a property string"""

    prop = ""
    print("Welcome to the stormvogel property string builder.")
    if input("\nCheck probabilities (p) or rewards (r): ") == "p":
        prop += "P"
        if (
            input(
                "\nDo you want to obtain a truth value (t) or a probability value (p): "
            )
            == "p"
        ):
            if model.supports_actions():
                print(
                    "\nThe model you provided supports actions, hence the reachability probability depends on the scheduler."
                )
                if (
                    input(
                        "Do you want the maximum (max) or minimum (min) probability? "
                    )
                    == "max"
                ):
                    prop += "max=?"
                else:
                    prop += "min=?"
            else:
                prop += "=?"
        else:
            op = input(
                "\nFor what operator do you want to know the truth value (<), (>), (<=), (>=) or (=): "
            )
            prop += op
            if (
                val := float(
                    input(
                        "\n For what probability value do you want to check the truth value: "
                    )
                )
            ) >= 0 and val <= 1:
                prop += str(val)
            else:
                raise RuntimeError(
                    "Not a valid probability. Choose a value between 0 and 1."
                )
        labels = model.get_all_state_labels()
        print("\nThese are all the state labels in the model:\n", labels)
        if (
            label := input(
                "\nFor what state label do you want to know the reachability probability? "
            )
        ) in labels:
            prop += f' [F "{label}"]'
        else:
            raise RuntimeError("This label is not part of the model.")
    else:
        prop += "R"

    print("\nThe resulting property string is: \n", prop)
    return prop


if __name__ == "__main__":
    mdp = examples.monty_hall.create_monty_hall_mdp()

    build_property_string_interactive(mdp)
