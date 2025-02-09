import stormvogel.model
import examples.monty_hall


def build_property_string_interactive(model: stormvogel.model.Model) -> str:
    """When a model is provided, this interative property string builder will help beginner
    users to create a property string"""

    def probability_or_reward() -> str:
        while True:
            if model.rewards == []:
                print(
                    "\nThis model does not have reward models, therefore we can only do model checking for probabilities."
                )
                return "P"
            choice = input("\nCheck for probabilities (p) or rewards (r): ").lower()
            if choice in {"p", "r"}:
                if choice == "r":
                    if len(model.rewards) > 1:
                        print("\nThis model has multiple reward models.")
                        print([r.name for r in model.rewards])
                        rewardmodel = input("\nChoose one of the above: ")
                        return choice.upper() + '{"' + rewardmodel + '"}'
                else:
                    return choice.upper()
            else:
                print("Invalid input. Please choose 'p' or 'r'.")

    def compare_or_obtain() -> str:
        while True:
            choice = input(
                "\nDo you want to check if a certain property holds (c) or obtain a value (o): "
            ).lower()
            if choice in {"c", "o"}:
                return choice
            print("Invalid input. Please choose 'c' or 'o'.")

    def max_or_min() -> str:
        print(
            "\nThe model you provided supports actions, hence the values will depend on the scheduler, \ntherefore you must choose between the minimum and maximum value over all schedulers."
        )
        while True:
            choice = input(
                "Do you want the maximum (max) or minimum (min) value: "
            ).lower()
            if choice in {"max", "min"}:
                return choice
            print("Invalid input. Please choose 'max' or 'min'.")

    def operator() -> str:
        while True:
            choice = input(
                "\nFor what operator do you want to know the truth value (<), (>), (<=), (>=) or (=): "
            )
            if choice in {"<", ">", "<=", ">=", "="}:
                return choice
            print("Invalid input. Please choose '<', '>', '<=', '>=' or '='.")

    def value() -> str:
        while True:
            choice = float(
                input("\nFor what value do you want to check the truth value: ")
            )
            if value_type == "P":
                if 0 <= choice and choice <= 1:
                    return str(choice)
                print("Invalid input. Please choose a value between 0 and 1.")
            else:
                return str(choice)

    def labels() -> str:
        labels = model.get_all_state_labels()
        print("\nThese are all the state labels in the model:\n", labels)
        s = ""
        while True:
            choice = input("\nChoose a label to append to the path: ")
            if choice in labels:
                s += choice
            else:
                print("Invalid input. Please choose a label from the list.")
            if (
                input("\nDo you want to append more labels to the path? (y) or (n): ")
                == "n"
            ):
                return s
            else:
                s += '" & "'

    print("Welcome to the stormvogel property string builder.")
    prop = probability_or_reward()
    value_type = prop

    if compare_or_obtain() == "o":
        prop += f"{max_or_min()}=?" if model.supports_actions() else "=?"
    else:
        prop += operator()
        prop += value()
    prop += f' [F "{labels()}"]'

    print("\nThe resulting property string is: \n", prop)
    return prop


if __name__ == "__main__":
    mdp = examples.monty_hall.create_monty_hall_mdp()

    build_property_string_interactive(mdp)
