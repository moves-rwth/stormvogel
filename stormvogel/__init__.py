"""The stormvogel package"""


def is_in_notebook():
    try:
        from IPython.core.getipython import get_ipython

        ipython = get_ipython()

        if ipython is None or "IPKernelApp" not in ipython.config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if is_in_notebook():
    # Import and init magic
    from stormvogel import magic as magic
