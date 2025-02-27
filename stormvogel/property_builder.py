import stormvogel.model
import ipywidgets as widgets
from stormvogel.dict_editor import DictEditor
import IPython.display as ipd


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


def build_property_string(model: stormvogel.model.Model):
    """Lets the user build a property string using a widget"""

    debug_output = widgets.Output()

    values = {
        "type of task": {
            "type": "probability",
            "task": "obtain",
            "maxmin": "max",
            "operator": "=",
            "value": 0.00,
        },
        "path": {"path": "init"},
    }
    schema = {
        "type of task": {
            "__collapse": True,
            "type": {
                "__html": "<p>Do you want to learn about probabilities or rewards?</p>",
                "__description": "Choose one",
                "__widget": "Dropdown",
                "__kwargs": {
                    "options": [
                        "probability",
                        "reward",
                    ]
                },
            },
            "task": {
                "__html": "<p>Do you want to compare values to check if a certain property holds or obtain a value?</p>",
                "__description": "Choose one",
                "__widget": "Dropdown",
                "__kwargs": {
                    "options": [
                        "obtain",
                        "compare",
                    ]
                },
            },
            "maxmin": {
                "__html": "<p>Since your model has actions your result depends on the scheduler. Do you want the maximum or minimum value?</p>",
                "__description": "Choose one",
                "__widget": "Dropdown",
                "__kwargs": {
                    "options": [
                        "max",
                        "min",
                    ]
                },
            },
            "operator": {
                "__html": "<p>If you chose 'compare', you should select a comparison operator.</p>",
                "__description": "Choose one",
                "__widget": "Dropdown",
                "__kwargs": {
                    "options": [
                        "<",
                        ">",
                        ">=",
                        "<=",
                        "=",
                    ]
                },
            },
            "value": {
                "__html": "<p>If you chose 'compare', you need to choose a value.</p>",
                "__description": "Choose one",
                "__widget": "FloatSlider",
            },
        },
        "path": {
            "__collapse": True,
            "path": {
                "__html": "<p>Select the states you want to append to your path</p>",
                "__description": "",
                "__widget": "TagsInput",
                "__kwargs": {
                    "options": [
                        "init",
                    ]
                },
            },
        },
    }

    class S:
        x = 0
        out = widgets.Output()

        def on_update(self):
            self.x = self.x + 1
            with self.out:
                ipd.clear_output()
                print(s.x)

    s = S()

    de = DictEditor(schema, values, s.on_update, debug_output=debug_output)
    de.show()

    s.out

    if values["type of task"]["type"] == "reward":
        print("Rewards")

    return values["type of task"]["type"]


"""
    prop = ""
    if values["type of task"]["type"] == "reward":
        prop += "R"
    else:
        prop += "P"

    if values["type of task"]["task"] == "obtain":
        prop += f"{"max" if values["type of task"]["maxmin"] == "max" else "min"}=?" if model.supports_actions() else "=?"
    else:
        prop += values["type of task"]["operator"]
        prop += values["type of task"]["value"]
    prop += f' [F "{path}"]'

    print("\nThe resulting property string is: \n", prop)
    return prop
"""


if __name__ == "__main__":
    import examples.monty_hall

    mdp = examples.monty_hall.create_monty_hall_mdp()

    build_property_string(mdp)
