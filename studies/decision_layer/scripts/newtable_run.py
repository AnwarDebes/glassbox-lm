#!/usr/bin/env python
"""Run abcd_explore.py or abcd_confirm.py on a NEW count table, through table_run.py unchanged (e82; added 2026-10-21).

    newtable_run.py --table-dir data/tables/<folder> --name <name> scripts/abcd_explore.py --dataset star \
        --encoder none --parts mem,words,xcust,xrecent --arms tm --clauses 16000 --s 16 --seed 1 --save --tag ...

Plan: logs/plan_e82_refined_table_agent.md. A table exported by the text-understanding line (for example the refined table of the text
stack's 3a) is copied into data/tables/<folder>/ with its MANIFEST.sha256, table.npz and tokenize_ref.py. This script
registers it in dl.count_text.TABLES for its own process only: the name, the folder, and table.npz's SHA-256 from the
MANIFEST, after checking that the file on disk has that hash. Then it runs scripts/table_run.py's main with
"--table <name>", so every check of table_run.py and dl.count_text.CountTable applies:
- the file hash, against the registered value and the MANIFEST;
- tokenize_ref.py equal to the reference that the C4 port follows (a table with another tokenizer is refused there);
- --encoder none and the no-encoder parts;
- potion's table cannot load.
No frozen file is edited (src/dl/count_text.py, scripts/table_run.py and the run scripts stay byte for byte).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

from dl import count_text  # noqa: E402


def register(table_dir: Path, name: str) -> str:
    """Put the table into dl.count_text.TABLES for this process; return table.npz's SHA-256."""
    d = (ROOT / table_dir).resolve() if not table_dir.is_absolute() else table_dir
    if d.parent != ROOT / "data" / "tables":
        raise SystemExit(f"{d}: the table folder must be in data/tables/")
    man = dict(reversed(line.split()) for line in (d / "MANIFEST.sha256").read_text().splitlines() if line.strip())
    want = man.get("table.npz")
    got = hashlib.sha256((d / "table.npz").read_bytes()).hexdigest()
    if not want or got != want:
        raise SystemExit(f"{d / 'table.npz'}: sha256 {got} != MANIFEST {want}")
    if name in count_text.TABLES and count_text.TABLES[name] != (d.name, want):
        raise SystemExit(f"table name {name} is taken by {count_text.TABLES[name]}")
    count_text.TABLES[name] = (d.name, want)
    return want


def main() -> None:
    a = sys.argv[1:]
    if len(a) < 5 or a[0] != "--table-dir" or a[2] != "--name":
        raise SystemExit(__doc__)
    table_dir, name, rest = Path(a[1]), a[3], a[4:]
    sha = register(table_dir, name)
    print(f"newtable_run: registered {name} -> {table_dir} (table.npz {sha[:16]})", flush=True)
    import table_run

    sys.argv = ["table_run.py", "--table", name] + rest
    table_run.main()


if __name__ == "__main__":
    main()
