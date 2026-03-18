# Auto-import all layerwise probe modules so they register themselves
import importlib
import pkgutil
import os

_pkg_dir = os.path.dirname(__file__)
for _finder, _name, _ispkg in pkgutil.walk_packages([_pkg_dir], prefix=__name__ + "."):
    try:
        importlib.import_module(_name)
    except Exception:
        pass
