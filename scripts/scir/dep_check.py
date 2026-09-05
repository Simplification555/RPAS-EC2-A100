import importlib.util
import sys
for name in ("tree_sitter_python", "git", "loguru", "anthropic"):
    print(name, bool(importlib.util.find_spec(name)))
print(sys.executable)
print("site", [p for p in sys.path if "site-packages" in p])
