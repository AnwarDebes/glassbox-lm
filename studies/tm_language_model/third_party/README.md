# third_party

- `GraphTsetlinMachine/`: cair/GraphTsetlinMachine master c6129e14 (MIT). The files kept here are unmodified;
  `UPSTREAM_COMMIT` and `LICENSE` are kept. Not included from upstream: `.devcontainer/`, `.gitignore` and
  `.gitattributes`. It is the stock reference for every equivalence test and is not edited (patched copies live in
  `src/tmlm/kernels/graphtm_patched/`).
- `ref/`: reference clones of other repositories (read and cited, never imported). They are not distributed with
  this repository; the upstream repositories and the commits recorded for them are listed in
  [docs/upstream-repositories.md](../../../docs/upstream-repositories.md) at the repository root.
  The hierarchical package PyHierarchicalTsetlinMachineCUDA is not installed from a reference clone:
  `env/htm_install.sh` installs it from GitHub at the commit pinned in that script, into a separate `.venv-htm`.
