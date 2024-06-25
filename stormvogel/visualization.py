"""Contains stuff for visualization"""

from stormvogel.model import Model
from ipywidgets import interact


def show(m: Model):
    pass


def make_slider():
    return interact(lambda x: x, x=10)
