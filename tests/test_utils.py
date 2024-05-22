import sys

def set_root_pypath():
    """
    Appends the project's root directory to the PYTHONPATH.
    This helps Python to find the main package `my_project` for imports.
    """
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
