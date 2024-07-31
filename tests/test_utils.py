import sys
import os

def set_ml_pypath():
    """
    Appends the project's openFlowML directory to the PYTHONPATH.
    This helps Python to find the main package `my_project` for imports.
    """
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'openFlowML')))
    print(f'Test path set: {sys.path}')