"""One line per installed distribution of the running interpreter's environment: name==version and the sha256 of its
RECORD file (which lists every installed file with its own hash), sorted. Pins a venv (rev2, review item 6):
    <venv>/bin/python -I -B scripts/venv_freeze.py > data/pins/venv_<name>.txt
"""
import hashlib
import importlib.metadata as md
import sys

lines = [f"# python {sys.version.split()[0]} prefix {sys.prefix}"]
for d in md.distributions():
    name = (d.metadata["Name"] or "?").lower()
    rec = d.read_text("RECORD")
    lines.append(f"{name}=={d.version} record_sha256={hashlib.sha256(rec.encode()).hexdigest() if rec else 'none'}")
print("\n".join([lines[0]] + sorted(set(lines[1:]))))
