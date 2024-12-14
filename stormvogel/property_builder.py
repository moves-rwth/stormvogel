import stormvogel.model


class PropertyBuilder:
    """
    Aimed to let beginner users build property strings

    Args:
    model: the model that the property string will be used for
    """

    model: stormvogel.model.Model
    prop: str

    def __init__(self, model: stormvogel.model.Model):
        self.model = model
        self.prop = ""

    def get_reachabilty_probability(self, label: str, value: str = "default") -> str:
        """To create a property that asks for the reachability probabilities of the states"""

        print(value)

        assert len(self.model.get_states_with_label(label)) > 0

        if value == "default":
            string = f'P=? [F "{label}"]'
            self.prop = string
            return string
        elif value == "max":
            string = f'Pmax=? [F "{label}"]'
            self.prop = string
            return string
        elif value == "min":
            string = f'Pmin=? [F "{label}"]'
            self.prop = string
            return string
        else:
            raise RuntimeError(
                "Please choose as value either 'max', 'min' or 'default'."
            )


def build_property_string_interactive():
    prop = ""
    print("Welcome to the stormvogel property string builder.")
    if input("Check propabilities (p) or rewards (r): ") == "p":
        prop += "P"
        if input("Does the model support actions (y) or (n)? ") == "y":
            if (
                input("Do you want the maximum (max) or minimum (min) probability? ")
                == "max"
            ):
                prop += "max=?"
            else:
                prop += "min=?"
        else:
            prop += "=?"

        label = input(
            "For what state label do you want to know the reachability probability? "
        )
        prop += f' [F "{label}"]'
    else:
        prop += "R"

    print("The resulting property string is: ", prop)
    return prop


if __name__ == "__main__":
    build_property_string_interactive()
