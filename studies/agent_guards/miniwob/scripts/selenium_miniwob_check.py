"""Route 2 check: MiniWoB gym env via Selenium with a userspace Chrome + chromedriver."""
import os, time, traceback
import gymnasium, miniwob
gymnasium.register_envs(miniwob)
print("MINIWOB_CHROME_BINARY =", os.environ.get("MINIWOB_CHROME_BINARY"))
print("MINIWOB_CHROMEDRIVER  =", os.environ.get("MINIWOB_CHROMEDRIVER"))
t0 = time.time()
env = gymnasium.make("miniwob/click-test-2-v1")
try:
    obs, info = env.reset(seed=0)
    print("RESET OK; utterance:", obs["utterance"])
except Exception as e:
    print("RESET FAILED:", type(e).__name__)
    print("".join(traceback.format_exception_only(type(e), e))[:2500])
finally:
    try:
        env.close()
    except Exception as e:
        print("close error:", type(e).__name__, str(e)[:300])
print("seconds", round(time.time() - t0, 2))
