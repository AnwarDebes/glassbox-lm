"""Dry check of a `tau2 run` argument list WITHOUT running anything.

It replaces tau2.cli.run_domain with a stub, so tau2's own argparse and the TextRunConfig (pydantic) validation run,
then prints the resulting config, the task ids that would be selected and the results path. No simulation starts,
no LLM is called, no server is needed.

Usage: .venv/bin/python -I -B scripts/dry_parse_run_args.py run --domain retail ... (same arguments as tau2 run)
"""

import json
import sys
from pathlib import Path


def main():
    import tau2.cli as cli
    from tau2.runner.helpers import get_tasks
    from tau2.utils.utils import DATA_DIR

    captured = {}

    def stub_run_domain(config):
        captured["config"] = config
        return None

    cli.run_domain = stub_run_domain
    sys.argv = ["tau2"] + sys.argv[1:]
    cli.main()
    cfg = captured.get("config")
    if cfg is None:
        print("DRY: no run config was built")
        return 1
    d = cfg.model_dump(mode="json")
    keep = ["domain", "task_set_name", "task_split_name", "task_ids", "num_tasks", "agent", "llm_agent",
            "llm_args_agent", "user", "llm_user", "llm_args_user", "num_trials", "max_steps", "max_errors",
            "timeout", "save_to", "max_concurrency", "seed", "auto_review", "auto_resume"]
    print("DRY config:", json.dumps({k: d.get(k) for k in keep}, indent=1, default=str))
    tasks = get_tasks(task_set_name=cfg.task_set_name or cfg.domain, task_split_name=cfg.task_split_name,
                      task_ids=cfg.task_ids, num_tasks=cfg.num_tasks)
    print("DRY selected task ids:", [t.id for t in tasks])
    if cfg.save_to:
        print("DRY results path:", Path(DATA_DIR / "simulations" / cfg.save_to / "results.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
