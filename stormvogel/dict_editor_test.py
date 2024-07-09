from stormvogel.dict_editor import Editor
import os
import json

PACKAGE_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(PACKAGE_ROOT_DIR, "layouts/default.json")) as f:
    default_str = f.read()
    default_dict = json.loads(default_str)

with open(os.path.join(PACKAGE_ROOT_DIR, "layouts/schema.json")) as f:
    schema_str = f.read()
    schema = json.loads(schema_str)


def on_update(d: dict):
    print(d)


Editor(schema, default_dict, on_update)
