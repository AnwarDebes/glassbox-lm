"""Print MiniWoB observation/action spaces using the package code only (no browser needed)."""
from miniwob.observation import get_observation_space
from miniwob.action import ActionSpaceConfig
from miniwob.constants import TASK_WIDTH, TASK_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT, MAX_FIELDS, MAX_REF, DEFAULT_ALLOWED_KEYS
obs = get_observation_space(TASK_WIDTH, TASK_HEIGHT)
print("task area (w x h):", TASK_WIDTH, "x", TASK_HEIGHT, "; browser window:", WINDOW_WIDTH, "x", WINDOW_HEIGHT)
print("observation keys:", list(obs.spaces))
print("dom element keys:", list(obs["dom_elements"].feature_space.spaces))
print("screenshot space:", obs["screenshot"].shape, obs["screenshot"].dtype)
for name in ["all_supported", "liu18", "shi17", "humphreys22"]:
    cfg = ActionSpaceConfig.get_preset(name)
    cfg.screen_width, cfg.screen_height = TASK_WIDTH, TASK_HEIGHT
    sp = cfg.get_action_space()
    print(f"preset {name}: {len(cfg.action_types)} action types {[a.value for a in cfg.action_types]}")
    print("   action dict keys:", {k: str(v) for k, v in sp.spaces.items()})
print("MAX_FIELDS", MAX_FIELDS, "MAX_REF", MAX_REF, "allowed keys:", len(DEFAULT_ALLOWED_KEYS))
