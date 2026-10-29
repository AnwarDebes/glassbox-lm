# Note: energy per decision, LLM* against the readable agent (cost only, no rule; written 2026-10-28)

## Measured

- LLM* (Qwen3.6-35B-A3B Q4_K_M, 50 examples, llama-server on the V100): nvidia-smi every 5 s for 3,600 s during the
  registered ABCD test run (reg_llm_abcd), 07:46:25-08:46:24Z, 720 samples
  (runs/cost/gpu_power_reg_llm_abcd_20261028T074625Z.csv, sha256 871bb5dfcdc0ea80). Board power: mean 188.5 W, median
  164.2 W, p10 142.8 W, p90 279.7 W, min 89.2, max 337.9; GPU busy 82% on average; 24.4 GB in use (only this server
  ran on the GPU: one job at a time). The trace reads nothing from the run's outputs.
- Time per decision: LLM* mean seconds per row on its 500 dev rows (runs/llm_q3_L35k50/<ds>/preds.jsonl, field s):
  STAR 10.57, ABCD 11.54, MultiWOZ 11.49. The agent's registered seed-1 models on one CPU thread
  (runs/cost/agent_<ds>_seed1_cpu.json, 1,000 dev rows): median 2.71 / 0.50 / 2.98 ms, p90 2.98 / 0.55 / 3.29 ms.

## Computed (energy = power x time per decision)

| | LLM*: GPU board only | AGENT charged the whole CPU package (205 W) | AGENT charged one core's share (205/24 W) | Ratio (package / core share) |
|---|---|---|---|---|
| STAR | 1,993 J | 0.556 J | 0.023 J | 3,588x / 86,115x |
| ABCD | 2,175 J | 0.103 J | 0.0043 J | 21,184x / 508,426x |
| MultiWOZ | 2,165 J | 0.611 J | 0.0255 J | 3,542x / 85,019x |

- CPU: Intel Xeon Platinum 8168 (lscpu), TDP 205 W, 24 cores, 48 threads (Intel's product page, SKU 120504,
  https://www.intel.com/content/www/us/en/products/sku/120504/intel-xeon-platinum-8168-processor-33m-cache-2-70-ghz.html,
  read 2026-10-28). The container exposes no CPU energy counter (the RAPL files are absent), so the agent's energy
  is bounded from the rated power, not measured. TDP is a sustained rating; short turbo bursts can exceed it.

## Reading

- Conservative: even when the agent is charged a whole 205 W CPU package while it runs one thread, and the LLM only
  its GPU board (not the host CPU, memory or power supply), LLM* uses about 3,500x (STAR, MultiWOZ) to 21,000x (ABCD)
  more energy per decision. With the agent's p90 times in place of the medians: 3,265x, 19,259x and 3,210x.
- With one core's share of the package for the agent, the ratios are about 85,000x to 510,000x.
- Limits: power was traced during ABCD only; STAR and MultiWOZ use the same mean power (their prompts differ in
  length). LLM* time is from dev rows; the registered report gives its test medians, which replace them in the paper.
