import sys

def set_ml_pypath():
    """
    Appends the project's openFlowML directory to the PYTHONPATH.
    This helps Python to find the main package `my_project` for imports.
    """
    import os
    # Navigate up one level from the current directory where this file is located,
    # then to the 'openFlowML' directory.
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'openFlowML'))
    if path not in sys.path:
        sys.path.append(path)
