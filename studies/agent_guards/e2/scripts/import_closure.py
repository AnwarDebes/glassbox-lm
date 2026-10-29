"""The project files an entry-point script imports (rev2, review item 6: every module on the path is pinned). Loads
each script as a module named "_closure" (so its main() does not run), then lists every loaded module file under
.. that is not inside a venv. Only scripts guarded by `if __name__ == "__main__":` are loaded:
a script without that guard runs its code on import, so it is refused (the closure is then taken by hand).
    <venv>/bin/python -I -B scripts/import_closure.py SCRIPT [SCRIPT ...]   # JSON on stdout
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
_WS_ROOT = _os.path.dirname(_REPO_ROOT)  # the folder that holds the repository and tools/
import importlib.util
import json
import sys
from pathlib import Path

ROOT = f"{_WS_ROOT}"


def main() -> None:
    out = {}
    for script in sys.argv[1:]:
        src = Path(script).read_text()
        if 'if __name__ == "__main__":' not in src:
            raise SystemExit(f"STOP: {script} has no main guard; loading it would run it")
        spec = importlib.util.spec_from_file_location("_closure", script)
        mod = importlib.util.module_from_spec(spec)
        argv, sys.argv = sys.argv, [script]
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.argv = argv
        files = set()
        for m in list(sys.modules.values()):
            f = getattr(m, "__file__", None)
            if f and f.startswith(ROOT) and "/.venv" not in f and "site-packages" not in f:
                files.add(str(Path(f).resolve()).replace(ROOT, ""))
        files.add(str(Path(script).resolve()).replace(ROOT, ""))
        out[str(Path(script).resolve()).replace(ROOT, "")] = sorted(files)
        sys.modules.pop("_closure", None)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
