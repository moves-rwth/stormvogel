"""Generate editor menu from a schema dict."""

from typing import Any, Callable
from IPython.display import display
from ipywidgets import (
    interactive,
    IntSlider,
    ColorPicker,
    Checkbox,
    Text,
    Dropdown,
    Accordion,
    VBox,
    HTML,
    Widget,
)
import copy

from stormvogel.rdict import rget, rset


class WidgetWrapper:
    """Creates a widget specified in the arguments.
    Changing the value of the widget will change the value specified in path in the update_dict."""

    convert_dict = {
        "IntSlider": IntSlider,
        "ColorPicker": ColorPicker,
        "Checkbox": Checkbox,
        "Text": Text,
        "Dropdown": Dropdown,
    }

    def __init__(
        self,
        description: str,
        widget: str,
        path: list[str],
        initial_value: Any,
        update_dict: dict,
        on_update: Callable,
        **kwargs,
    ) -> None:
        """Creates a widget which automatically updates a dictonary value.

        Args:
            title (str): 'description' of the widget.
            widget (Widget): A str name for a widget.
            path (list[str]): path to the value in update_dict to be changed.
            initial_value (Any): initial value of the widget (aka. 'value')
            update_dict (dict): The dict that should be updated.
            on_update (Callable): A function that is called whenever a value is updated.
        """
        w = self.convert_dict[widget]
        self.update_dict = update_dict
        self.path = path
        self.on_update = on_update
        self.widget = interactive(
            self.on_edit, x=w(value=initial_value, description=description, **kwargs)
        )

    def on_edit(self, x: Any) -> None:
        """Called when a user changes something in the widget."""
        rset(self.update_dict, self.path, x)
        self.on_update()


class Editor:
    """Create an interactive json editor from a schema using ipy widgets."""

    def __init__(self, schema: dict, update_dict: dict, on_update) -> None:
        """Create an interactive json editor from a schema using ipy widgets.

        Args:
            schema (dict): The dict that specifies the schema.
                Quite closely follows the structure of the actual dict. Supports macros.
                See layouts/schema.json for an example. Better docs TODO.
            update_dict (dict): The dict that will be updated.
            on_update (_type_): Function that is called whenever the dict is updated.
        """
        self.on_update = on_update
        self.update_dict = update_dict
        self.macros = {}
        result = self.recurse_create(schema, [])
        print(result.children)
        display(result)

    def recurse_create(self, sub_schema: dict, path: list) -> Widget:
        acc_items = []
        for k, v in sub_schema.items():
            new_path = copy.deepcopy(path)
            new_path.append(k)
            if k == "__html":
                acc_items.append(HTML(v))
            if k == "__macros":
                self.macros = v
            elif isinstance(v, dict):
                if "__html" in v and "__widget" in v:
                    acc_items.append(HTML(v["__html"]))
                if "__use_macro" in v:
                    macro_value = self.macros[v["__use_macro"]]
                    acc_items.append(self.recurse_create(macro_value, new_path))
                elif (
                    "__description" in v
                ):  # v is a widget, because it has a defined __description.
                    if "__kwargs" in v:  # Also pass arguments if relevant.
                        w = WidgetWrapper(
                            description=v["__description"],
                            widget=v["__widget"],
                            initial_value=rget(self.update_dict, new_path),
                            path=new_path,
                            update_dict=self.update_dict,
                            on_update=self.on_update,
                            **v["__kwargs"],
                        )
                    else:
                        w = WidgetWrapper(
                            description=v["__description"],
                            widget=v["__widget"],
                            initial_value=rget(self.update_dict, new_path),
                            path=new_path,
                            update_dict=self.update_dict,
                            on_update=self.on_update,
                        )
                    acc_items.append(w.widget)
                else:  # v is not a widget or macro, then call recursively.
                    acc_items.append(self.recurse_create(v, new_path))
        if "__collapse" in sub_schema and sub_schema["__collapse"]:
            acc = Accordion(children=[VBox(children=acc_items)], titles=[path[-1]])
        else:
            acc = VBox(children=acc_items)
        return acc
