# Note: five LLM jobs refused at 01:14Z (written 2026-10-27T01:21:25Z)

- At 01:14:07-01:14:48Z serve_headline.sh refused to start a server five times ("a GPU compute process runs"): the
  ablation's L35-noG part and the jobs llm_q3_L35t, llm_q3_L35k50, llm_q3_L9t_star, llm_q3_L35t_mwoz (queue exit 3 each).
  No LLM answer of these arms was produced.
- Cause, checked: the CPU-only synthetic grid (run_zipf.sh, tmu with platform CPU) opened a CUDA context of 306 MiB on
  the GPU; it disappeared when that run was stopped. The launcher's check counts any GPU compute process, so it refused.
- Fix: run_zipf.sh now hides the GPU (CUDA_VISIBLE_DEVICES=""); its interrupted run is redone (restart-safe). The five
  jobs are queued again after ft_star_roberta_s1; the ablation job keeps its finished L9 rows (restart-safe client).
