import pkgutil
import importlib
import logging
import sys


logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

current_package = sys.modules[__name__]

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):

    if module_name == __name__:
        continue

    full_name = f"{__name__}.{module_name}"
    imported_module = importlib.import_module(full_name)
    
    setattr(current_package, module_name, imported_module)

__all__ = ["colab_rag"]

__version__ = "0.0.1"