"""Dump one MiniWoB observation (screenshot as PNG, dom_elements with refs as JSON) per task, seed 0.
Needs MINIWOB_CHROME_BINARY / MINIWOB_CHROMEDRIVER (the wrappers in browser/). Run with python -I."""
import json
import os
import sys

import gymnasium
import miniwob
from PIL import Image

gymnasium.register_envs(miniwob)
outdir = sys.argv[1]
os.makedirs(outdir, exist_ok=True)
for task in sys.argv[2:]:
    env = gymnasium.make(f"miniwob/{task}-v1", action_space_config="all_supported")
    try:
        obs, info = env.reset(seed=0, options={"record_screenshots": True})
        Image.fromarray(obs["screenshot"]).save(os.path.join(outdir, f"obs_{task}_seed0.png"))
        els = [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in e.items()} for e in obs["dom_elements"]]
        with open(os.path.join(outdir, f"obs_{task}_seed0.json"), "w") as f:
            json.dump({"utterance": obs["utterance"], "fields": list(obs["fields"]), "dom_elements": els}, f, indent=1)
        print(task, "|", obs["utterance"], "|", len(els), "elements |",
              [(e["ref"], e["tag"], e["text"][:20]) for e in els if e["flags"][3]][:12])
    finally:
        env.close()
