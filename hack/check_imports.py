"""Resolve every ``from gpustack.x import y`` under ./gpustack, lazy ones included.

pyflakes cannot see an import that sits inside a function, so a name removed from
a module only fails when that code path first runs in a live server. Run from the
directory that contains the ``gpustack`` package:

    python hack/check_imports.py

Exits non-zero when any import does not resolve.
"""
import ast, importlib, pathlib, sys
bad = []
for f in pathlib.Path("gpustack").rglob("*.py"):
    if "migrations" in f.parts: continue
    try: tree = ast.parse(f.read_text())
    except SyntaxError as e: bad.append((str(f), "syntax", str(e))); continue
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("gpustack") and n.level == 0:
            try: m = importlib.import_module(n.module)
            except Exception as e: bad.append((str(f), n.module, repr(e)[:80])); continue
            for a in n.names:
                if a.name != "*" and not hasattr(m, a.name):
                    try: importlib.import_module(f"{n.module}.{a.name}")
                    except Exception: bad.append((str(f), n.module, a.name))
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("gpustack"):
                    try: importlib.import_module(a.name)
                    except Exception as e: bad.append((str(f), a.name, repr(e)[:80]))
print(len(bad), "unresolved")
for b in bad:
    print(b)
sys.exit(1 if bad else 0)
