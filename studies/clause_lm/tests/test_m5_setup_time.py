"""src/m5_setup.py: the download time in the setup record is read from the data file, not written as a fixed string.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv-cdllm/bin/python tests/test_m5_setup_time.py [M5_SETUP]

Run from studies/clause_lm. CPU only, under a second. M5_SETUP defaults to src/m5_setup.py; give another copy (for
example the version before 2026-10-25) to test it the same way. Only the module's config function runs: no setup run,
no subprocess, nothing written into results/ or work/.

Asserts:
- the source holds no fixed time stamp of the form YYYY-MM-DDTHH:MM:SSZ;
- setup_config() takes the time from the data file: a temporary file with a set modification time gives that time;
- on the machine that ran M5 step 1 (work/ present), the time read from the CLINC150 file equals the time the setup
  run of 2026-10-02 recorded (results/20261002T195152Z_m5_setup_sorted/config.json); skipped without work/.
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()
RECORDED = toyutil.STUDY / "results" / "20261002T195152Z_m5_setup_sorted" / "config.json"


def main():
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else toyutil.SRC / "m5_setup.py"
    print(f"m5_setup under test: {target}")
    src = target.read_text()
    stamps = re.findall(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", src)
    check(stamps == [], f"no fixed time stamp in the source ({stamps})")
    with tempfile.TemporaryDirectory(prefix="clm_setup_") as tmp:
        toyutil.use_src(Path(tmp))
        mod, _ = toyutil.load_module_copy(target, "m5_setup_under_test")
        if not hasattr(mod, "setup_config"):
            check(False, "the module has setup_config()")
            check.done()
        real_data = str(toyutil.STUDY / "work" / "Clause-Driven-LLM" / "data" / "clinc150_data_full.json")
        f = Path(tmp) / "data_full.json"
        f.write_text("{}")
        os.utime(f, (1_700_000_000, 1_700_000_000))          # 2023-11-14T22:13:20Z
        mod.DATA = str(f)
        cfg = mod.setup_config("sorted", "/nowhere")
        check(cfg["data_downloaded_utc"].startswith("2023-11-14T22:13:20Z"),
              f"the time comes from the file ({cfg['data_downloaded_utc']})")
        check(cfg["data_path"] == str(f) and cfg["label_order"] == "sorted", "the config names the file and the order")
        if os.path.exists(real_data) and RECORDED.exists():
            mod.DATA = real_data
            now = mod.setup_config("sorted", "/nowhere")["data_downloaded_utc"][:20]
            rec = json.loads(RECORDED.read_text())["data_downloaded_utc"][:20]
            check(now == rec, f"the CLINC150 file gives the recorded time ({now} vs recorded {rec})")
        else:
            print("skip: no work/ copy of the CLINC150 file or no recorded setup run on this machine")
    check.done()


if __name__ == "__main__":
    main()
