"""Generate editor menu for a schema dict."""

from functools import reduce
from typing import Any
from IPython.display import display, HTML
from ipywidgets import (
    interact,
    IntSlider,
    ColorPicker,
    Checkbox,
    Text,
    Dropdown,
)
import copy


class WidgetWrapper:
    convert_dict = {
        "IntSlider": IntSlider,
        "ColorPicker": ColorPicker,
        "Checkbox": Checkbox,
        "Text": Text,
        "Dropdown": Dropdown,
    }

    def __init__(
        self, title, widget, path, initial_value, update_dict, on_update, **kwargs
    ) -> None:
        print(
            "title",
            title,
            "widget",
            widget,
            "path",
            path,
            "initial",
            initial_value,
            "kwargs",
            kwargs,
        )
        w = self.convert_dict[widget]
        self.update_dict = update_dict
        self.path = path
        self.on_update = on_update
        interact(self.on_edit, x=w(value=initial_value, description=title, **kwargs))

    def on_edit(self, x: Any) -> None:
        rset(self.update_dict, self.path, x)
        self.on_update(self.update_dict)


def rget(d: dict, path: list) -> Any:
    """Recursively get dict value."""
    return reduce(
        lambda c, k: c.__getitem__(k), path, d
    )  # Throws KeyError if key not present.


def rset(d: dict, path: list, value: Any) -> None:
    """Recursively set dict value."""
    if len(path) == 0:
        return

    def __rset(d: dict, path: list, value: Any):
        first = path.pop(0)
        if len(path) == 0:
            d[first] = value
        else:
            __rset(d[first], path, value)

    __rset(d, copy.deepcopy(path), value)


class Editor:
    def __init__(self, schema: dict, update_dict: dict, on_update) -> None:
        self.on_update = on_update
        self.update_dict = update_dict
        self.macros = {}
        self.recurse_create(schema, [])

    def recurse_create(self, sub_schema: dict, path: list) -> None:
        for k, v in sub_schema.items():
            new_path = copy.deepcopy(path)
            new_path.append(k)
            if k == "__define_macro":
                self.macros[v["__name"]] = v["__value"]
            elif k == "__html":
                display(HTML(v))
            elif isinstance(v, dict):
                if "__use_macro" or "__title" in v:
                    if "__html" in v:  # Also render html within marcros.
                        display(HTML(v["__html"]))
                if "__use_macro" in v:
                    macro_value = self.macros[v["__use_macro"]]
                    self.recurse_create(macro_value, new_path)
                elif "__title" in v:  # v is a widget, because it has a defined title.
                    if "__kwargs" in v:  # Also pass arguments if relevant.
                        WidgetWrapper(
                            title=v["__title"],
                            widget=v["__widget"],
                            initial_value=rget(self.update_dict, new_path),
                            path=new_path,
                            update_dict=self.update_dict,
                            on_update=self.on_update,
                            **v["__kwargs"],
                        )
                    else:
                        WidgetWrapper(
                            title=v["__title"],
                            widget=v["__widget"],
                            initial_value=rget(self.update_dict, new_path),
                            path=new_path,
                            update_dict=self.update_dict,
                            on_update=self.on_update,
                        )
                else:  # v is not a widget, then call recursively.
                    self.recurse_create(v, new_path)
