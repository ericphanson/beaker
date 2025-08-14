import importlib
import importlib.metadata as im
import inspect

import rfdetr

print("rfdetr module:", rfdetr)
print("__file__:", getattr(rfdetr, "__file__", None))
print("__path__:", getattr(rfdetr, "__path__", None))
print("dir(rfdetr)[:40]:", dir(rfdetr)[:40])

# Which distribution provides top-level 'rfdetr'?
print("packages_distributions['rfdetr']:", im.packages_distributions().get("rfdetr"))

# Try the suspected defining modules:
for mod in ("rfdetr.detr", "rfdetr.models", "rfdetr.models.lwdetr"):
    try:
        m = importlib.import_module(mod)
        print(f"Imported {mod} from", inspect.getsourcefile(m))
        print("  has RFDETRNano:", hasattr(m, "RFDETRNano"))
    except Exception as e:
        print(f"Import {mod} failed:", repr(e))
