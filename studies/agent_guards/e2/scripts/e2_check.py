"""E2 checker (rev2, review item 6; rev3): compares LITERAL hashes before every stage, checks the server's identity,
the replay store's integrity between stages, the write-once freeze record and the logged stops and reruns. Standard
library only; read-only except for the campaign records it keeps (runs/server_identity_ref.json, written once by
E2-0; runs/store_chain.jsonl, one line per stage end; runs/stopped_stages.jsonl, one line per stop and per rerun).

    python -I -B scripts/e2_check.py --pins                       # every pinned file (literal sha256 below)
    python -I -B scripts/e2_check.py --venvs                      # both venvs' package lists (scripts/venv_freeze.py)
    python -I -B scripts/e2_check.py --stage S --begin --port P   # pins + venvs + server identity + store chain
    python -I -B scripts/e2_check.py --stage S --begin --rerun --port P   # the one rerun of a stopped stage
    python -I -B scripts/e2_check.py --stage S --end --run-dir D  # append the store's {bytes, sha256} to the chain
    python -I -B scripts/e2_check.py --stage S --stopped --run-dir D --exit-code 3   # log a transport stop

Stage start (--begin):
- every pinned file and both venvs match (any mismatch or missing file: STOP);
- the server: /health ok, /v1/models lists the alias, /props, the llama-server process that listens on the port
  (its command line and executable from /proc), the GGUF at /props' model_path hashed in full (read-only, about 30 s);
  the pinned literals must hold (build b1-4c3c581 = b11742's commit 4c3c581, the GGUF's sha256, one slot, the binary,
  -np 1, --jinja, --reasoning-budget 0, -fa off); E2-0 then writes runs/server_identity_ref.json (only on a fresh
  campaign: no store, no chain, no reference yet), and every later stage must equal that record field by field;
- the store: at E2-0 it must not exist yet; later it must be byte-identical to the last chain record (nothing written
  between stages, and its history unchanged: the chain's prefix check);
- E2-0's stop/go gate: every stage after E2-0 needs runs/E2-0_gate.json (report_e2.py smoke-gate) with go = true;
- the freeze (rev3, the decision 4): every stage after E2-0 needs runs/E2_frozen.json (scripts/freeze_e2.py,
  write-once): the registration file must still have the recorded sha256 and must name this checker's own sha256
  (its Appendix C); the freeze time must precede the stage start and follow E2-0's end; runs/server_identity_ref.json,
  runs/E2-0_gate.json, runs/budget.json and runs/E2-0_summary.json must still have the hashes recorded at the freeze
  (the E2-0 plan is final); the freeze record's sha256 is printed (the stage's slot.log) and every chain record after
  E2-0's carries it (frozen_sha256);
- E2-0 starts a fresh campaign: runs/ may hold only the current E2-0 run folder (the job file creates it just before
  this check); anything of an earlier campaign is moved aside unopened first;
- stops and reruns (rev3, the decision 3): a stage that stopped on a transport error is logged with
  --stopped (its store record at the stop); no other stage starts until it is rerun; it is rerun once (--rerun), in
  full, on the same store, which may then extend the last chain record only by the stopped attempt's bytes (the store
  must equal the stop record); a second stop of the same stage leaves it incomplete (no second rerun);
- the final models: the build stage must start without models/final (and without models/final.d6tmp); at its end every
  file in models/final is hashed into runs/models_final_pins.json, and arms, a2 and a3 start only if models/final still
  matches it.
Exit code 0 = go; anything else = STOP with the reason printed.
"""

from __future__ import annotations
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
_WS_ROOT = _os.path.dirname(_REPO_ROOT)  # the folder that holds the repository and tools/
_HOME = _os.path.expanduser("~")

import argparse
import calendar
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

E2 = Path(__file__).resolve().parents[1]
ROOT = Path(f"{_WS_ROOT}")
RUNS = E2 / "runs"
STORE = RUNS / "sealed" / "replay_store.jsonl"  # rev3: sealed until the freeze (only the driver, build_d6 and this)
CHAIN = RUNS / "store_chain.jsonl"
IDENTITY_REF = RUNS / "server_identity_ref.json"
MODELS_FINAL = E2 / "models" / "final"
MODELS_PINS = RUNS / "models_final_pins.json"  # written once at the build stage's end; checked by arms, a2 and a3
SMOKE_GATE = RUNS / "E2-0_gate.json"  # report_e2.py smoke-gate; every stage after E2-0 needs go = true, not fake
BUDGET = RUNS / "budget.json"  # report_e2.py budget (E2-0); final at the freeze
FROZEN = RUNS / "E2_frozen.json"  # scripts/freeze_e2.py, write-once; every stage after E2-0 needs it
E2_0_SUMMARY = RUNS / "E2-0_summary.json"  # the job file's copy of E2-0's open summary (the A3 gate and the report read
#                                            the optimizer refusal rule from it); hashed at the freeze
STOP_LOG = RUNS / "stopped_stages.jsonl"  # transport stops (--stopped) and their one rerun (--begin --rerun)
VENVS = {"agentdojo": E2.parent / "agentdojo" / ".venv" / "bin" / "python",
         "dl": ROOT / "glassbox-lm" / "studies" / "decision_layer" / ".venv-dl" / "bin" / "python"}
VENV_PIN_FILES = {"agentdojo": E2 / "data" / "pins" / "venv_agentdojo.txt", "dl": E2 / "data" / "pins" / "venv_dl.txt"}
STAGES = ["smoke", "d0", "d1", "build", "arms", "a2", "a3"]

# The server, pinned now (read-only hashes of 2026-10-21, logs/rev2/hash_server_files.log). /props' build_info is
# "b<build number>-<commit>"; the b11742 source is a shallow clone, so the build number reads 1.
SERVER_PINS = {
    "build_info": "b1-4c3c581",
    "commit": "4c3c5810f87efe177949f1b85ae509d38a2a311f",
    "model_alias": "qwen3.5-9b-q8_0",
    "total_slots": 1,
    "gguf_sha256": "ff3d8ff122c21c37f2ca52502db705e36903c581657f1e8d3e930a9cfb4d50aa",
    "gguf_bytes": 9786060672,
    "exe_sha256": {"llama-server": "2fc2870a983daedf21cc248a890752fb761fd9df85e04d68b1264bc61a84652b",
                   "libllama-server-impl.so": "22be23afb6ca5593cab0c89eccbab0e8234dc3242b64e1d29113a735efafced4",
                   "libllama.so.0.7.0": "52828c64f37a0fab9a4571d044aa39d36f05eb85a1d885e6ca91f7a4297f13cb",
                   "libllama-common.so.0.7.0": "9980440e054a95c1ed32fec798640d337cace43e9b1c06fc304bb4e693fe8fc4",
                   "libggml.so.0.27.0": "06a3d6200f8ead292f38e2f265eb657ccf315891f1d69d6728a4d83ac090cc43",
                   "libggml-base.so.0.27.0": "02396b5e005ca26e1731bac7c11e9b4bb0048be49bb1a8ab93425f3bcfefa0c0",
                   "libggml-cpu.so.0.27.0": "d317381205a1a687214aae1344c6bf11dc13f4c6507fb1d9125391a7becc5648",
                   "libggml-cuda.so.0.27.0": "5b695d777bdb870e6cb1bcb48d2f9ba02c9a5a2933905c9d37039ca0b6d59d6b",
                   "libmtmd.so.0.7.0": "5ecccabbb1e1dc8b557ffb2835fc0d9ecdbbd6e613ea889ed1cfa9f02d044765"},
    "flags": {"-np": "1", "--jinja": True, "--reasoning-budget": "0", "-fa": "off"},
    "min_n_ctx": 32768,
}

# BEGIN PINS (written by scripts/hash_table.py --write-pins; paths relative to ..)
PINS: dict[str, str] = {
    'glassbox-lm/studies/decision_layer/scripts/cpu_load.py': '2e300054a9284ef4bb0d395048fb5988d3176c100aa184384d22a3cc5eb04328',
    'glassbox-lm/studies/decision_layer/scripts/verify_policy.py': 'c10c7722bb9f3defb018e13e1421188309d85c8d605283483d48dc780e4f67ed',
    'glassbox-lm/studies/decision_layer/src/dl/boolean.py': 'b5f5917f390da8a6f73a215b08fe713f2fd0de24d7c950abbf5b31e8af36cc57',
    'glassbox-lm/studies/decision_layer/src/dl/cpu_tm_fast.py': 'bf1c4a9086d76448d9d77cf7775e0bb0e5f4bffc926d777b867f89148279423e',
    'glassbox-lm/studies/decision_layer/src/dl/data.py': '0c4f8d22576e817c6cb676582c496d4b5e3ee613208f878e1176161799159a1f',
    'glassbox-lm/studies/decision_layer/src/dl/explain.py': '1b6f79a8965f31851f59a7f70121886e9a1d8133fc2a11a4c65a15c32576e780',
    'glassbox-lm/studies/decision_layer/src/dl/linear.py': 'e3ef06b96ef60740af7757c9f8c0672d7c6258404d67936bd4106723d3a0a65d',
    'glassbox-lm/studies/decision_layer/src/dl/tm.py': 'f649043f5801605b35174c526b290a0cfbadbdb66ade5bf04dafc122289d3424',
    'glassbox-lm/studies/decision_layer/src/dl/tm_init.py': '557ff697065297387c0b698ec77ae1159496e10fd9c1547dc79dbf2ee67ef577',
    'glassbox-lm/studies/agent_guards/agentdojo/harness/guard.py': 'f2727c78e3e47665749cae1633332a26a5d8f54fa87532c1570be46e96f4de2d',
    'glassbox-lm/studies/agent_guards/agentdojo/harness/llamaserver_llm.py': '613766639e737eba9d8ebf234a227a32fd946b748e21e1d0daea35e912c92a17',
    'glassbox-lm/studies/agent_guards/agentdojo/harness/pipelines.py': '82c2d965c612010b6476ce4c69968b084dcf892a01e77df91f72711a0d9ce456',
    'glassbox-lm/studies/agent_guards/agentdojo/harness/quiet_logger.py': 'f67eecd91046a98caeda7550ee0ec493d5a205b9440eee5adb7c6a719e17a2ba',
    'glassbox-lm/studies/agent_guards/agentdojo/harness/replay.py': '4a26e72558c6fb03cc2e568356bcd84b4f72420949cc2d93ef46405f246827bd',
    'glassbox-lm/studies/agent_guards/agentdojo/pip_freeze.txt': 'd45aab3552935ae524c32d231c419f924a8429ca75cb72333fff6200067ffcab',
    'glassbox-lm/studies/agent_guards/e2/data/attack_selections_v1.2.2.json': '5a3d3eefaf0d9bac525d2c1af85a1e11323c219186c09a25f77f899f78d425b4',
    'glassbox-lm/studies/agent_guards/e2/data/pins/a3_descriptions.json': '076818bfebe52491807042fe9da5f3ac0373e1e226d14b5900f51cc45c9aee87',
    'glassbox-lm/studies/agent_guards/e2/data/pins/import_closure.json': '29ef0f48b9c1878a6f3fc64b9ae7e2f3efd98686d0618e7425cc13b751b3de2d',
    'glassbox-lm/studies/agent_guards/e2/data/pins/venv_agentdojo.txt': '570389187deda4d1e2d338d5f149143b18b6596916206f551625d398597ec123',
    'glassbox-lm/studies/agent_guards/e2/data/pins/venv_dl.txt': 'f878117fab534b69a48a0119eaa187b188d9d17ad56a1b6723d61ac164d8dddd',
    'glassbox-lm/studies/agent_guards/e2/data/pins/veto_share_offline.json': '5cb5e6a5cd348ef7e1b9e8eada6fc949acfbb817da1b4081441e45495c442cd4',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/SUMMARY.json': '2ade220466a34bd171a9aa150743b37a329ef9084ff5abdb4abb34159109d807',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_banking.json': '3d7443edacae4a0c8bfb8fc40f6f6a34b2f204d480a3a92b5b6bcf1088124309',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_banking__inner_slack.json': 'f0a9486783afbb3ed9d1cc2cf88c4e0c6642f68c35de499956d462c75898ea07',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_banking__inner_travel.json': 'd77a979412ca412c15913d57421014eb00b7250cd1da325afbf82d43930646b3',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_banking__inner_workspace.json': '4ccc2a56147a6ac00142e3b915cc6dde680f4ba2cd030d045001c9e039b63d0f',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_slack.json': 'c0284dd1ce07e92540746ded36595530441c859c47bfb107f437c54b7ed2e7e4',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_slack__inner_banking.json': '7e66815488669d79dc428c32f902aff0d73772e3d152fb5ed18711c40888a99a',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_slack__inner_travel.json': 'bce5f4ae30bb86df1d064749b40f6582f3c2189c19a5a0803f8d7478c75b5b45',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_slack__inner_workspace.json': '98e9e4dbc0f60c087fb7a4bfa1963366ebc55dba3f937c0bb7287650b13b52ad',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_travel.json': '11c3f556ca6468371b2c70023d19e59e0b039711d88b00bb8927c84af98a3cdd',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_travel__inner_banking.json': '7592a9c3cb4d12a490ba4b74ab3b4af0376e4bba28b1a8c300a09d90e4500871',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_travel__inner_slack.json': 'b7aa25f2f39c3a8333b9329c935b0d5da330af4a422bac9e5f5a10820f96b5bd',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_travel__inner_workspace.json': '3a2ec009db88c5220d0d7f7c6447a6b13c5667d18be7f4b5e57666dadee046d2',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_workspace.json': '35669eaa17caea979e322656046982135967d127b41ea09fc29b072ade966f1e',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_workspace__inner_banking.json': '1c8d5812581f847a128aabb1f4222276db46093607c7cb1581daab8d5ea04a25',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_workspace__inner_slack.json': 'a58ad3a04cda47478d4763081f0d3091c71fecd4d08a33b6ddea26db695b4438',
    'glassbox-lm/studies/agent_guards/e2/data/query_bits/fold_workspace__inner_travel.json': '0605c038478b618af4e7a7fac8fdbb5ce01cb5c892b257138ff624f4a8eea61e',
    'glassbox-lm/studies/agent_guards/e2/data/rows_gt_v1.2.2.jsonl': '8ab892d8c11ed65048f0c990344d41bd98300e85671c238ade1670817596e47d',
    'glassbox-lm/studies/agent_guards/e2/data/rows_gt_v1.2.2.meta.json': 'cde2565485949bcf6c4aeef3af80c76fef5dde0134f6acc1c21c207b310e8700',
    'glassbox-lm/studies/agent_guards/e2/data/rows_standin_traces.jsonl': '323ac9ecdc41b4875afaa8215297f79cccb892eb49f2b41ae005a4abbad00f56',
    'glassbox-lm/studies/agent_guards/e2/data/rows_standin_traces.meta.json': '58b716bb092e88a04cd00083d1c99b077d5ff20cf5b318a03e1835e16dbd8f72',
    'glassbox-lm/studies/agent_guards/e2/data/tasks_v1.2.2.json': '3035c921e524e2516e5f4a9612af23e63000755491e54f9285f0c6b9fb4dae78',
    'glassbox-lm/studies/agent_guards/e2/data/tool_docs_v1.2.2.json': '49450eae01e4108db9653b7232c5c36ec234c9c6f88bf84157c40a11ef121652',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d4_tm_banking.json': 'a60d2059ac9a68720e8cfd49544aa0c3bbd17921d4eb977ad40f023e51121249',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d4_tm_slack.json': '37ea1349af1274965baa48eca3aadb10186aa8b7a626ad0f256ac6dc87b56d13',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d4_tm_travel.json': '77dd089025629f54371b57eddecfd7d0a294e1178269eda5d19fcc7b271a7071',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d4_tm_workspace.json': '9d680a2b452ece1a190f24c5605190ac75d102f21f13f985a29afb9b51014f18',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d5_lr_banking.json': 'be586f57f27e98eb668701411f459e932df5615a324977f2643885929509c604',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d5_lr_slack.json': '8a4a100a85094a59df0aeacc42c48ba85b0a88bc38d184bc0b310c9fae6e31e5',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d5_lr_travel.json': 'd6edaa42b6e3960bc88e35258b6bccbbcce4216fb49ed7b7b67fe42a809e88b5',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d5_lr_workspace.json': 'c7eaafb769a536668c340c25c29400232c622bc21ae1a633f1be18098a4333c1',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d5_tree_banking.json': '7c316dca0fb85920c2909f1829246972671e272f1fedbda879a1c2fc7e7809c1',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d5_tree_slack.json': 'fa32d3397f67eb785af2164b7941780838bed1a37d831a8a5adb06d2f541658f',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d5_tree_travel.json': '5e19901af7295c4f14920d26d7eb21526a0fdc32b53f726e95660495ef612fa1',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d5_tree_workspace.json': 'b008be841e348964590f1fa1545c8aaefdac4e923d0c45b9270f2bae72bf19f6',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d6_rules_banking.json': 'c5544b488234d0d0f507f129331e233b825afb6dbd52e10af932ae0a57d57059',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d6_rules_slack.json': '33cf9ee720e01db3d2b298f7c3c6c1a1e6f4f82d3e4f695997c1b171b77a7f35',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d6_rules_travel.json': 'e4ee8baf00f5383c4f928afd95e867a7972a64e91d011705005545cfbd31ce68',
    'glassbox-lm/studies/agent_guards/e2/models/standin/d6_rules_workspace.json': '8bcb89ab8a8b8b55d54e389c510c0b48cd593f18c464d5dfec036204b51702b0',
    'glassbox-lm/studies/agent_guards/e2/models/standin/train_log.json': '4e44b49e8c6629c2af23565c3f0fdd38a4181ea8e12a871a337c4697b0438a2d',
    'glassbox-lm/studies/agent_guards/e2/results/milp_E2.5_standin.jsonl': 'b5ca647ca9715e2420dd25dfe90f0f88aeaef711c61f92da1abaf4ea9f5c05e2',
    'glassbox-lm/studies/agent_guards/e2/results/offline_e2e_standin.json': '5ca615343a06431eb12acef309bdf5cae321134c065a3976187734f3f1ecc925',
    'glassbox-lm/studies/agent_guards/e2/results/rule_rows_gt.json': '4bd26316f3873c4519925b6f37dd064739df10483f067ba3cd2d45fa6982b09a',
    'glassbox-lm/studies/agent_guards/e2/run_e2_gpu_slot.sh': 'de7e5a7f1ee16dee1e3be5e3a4a0dc77da895102a7fbd2853b6cc12442f54286',
    'glassbox-lm/studies/agent_guards/e2/scripts/build_d6.py': '245dadf755425264cfca74214abb271a82f891e4fc3fcc1158217d1272aa6597',
    'glassbox-lm/studies/agent_guards/e2/scripts/check_twins.py': '8d29c3217e0fe8209022aa1806086bdecfe21ec2f091faa24b41537486bbbf8c',
    'glassbox-lm/studies/agent_guards/e2/scripts/dump_tasks.py': '28fc00591032323c82b565fe64fd1e524eb456eddc30c9ad171931d4a0fdc5e6',
    'glassbox-lm/studies/agent_guards/e2/scripts/dump_tool_docs.py': 'bcff79c4b40d7fa8a5154d0c2a0806b9eb4c6dbbe342e303d160e61b7db9d554',
    'glassbox-lm/studies/agent_guards/e2/scripts/freeze_e2.py': 'f23b44a705c6f7eddfc537c9cab57abd39b2042c2043e69a7c097eb6066d551e',
    'glassbox-lm/studies/agent_guards/e2/scripts/hash_table.py': '6c554b2f55474ee0e71696c4a30aeb9c43349b759e96d3a49ed975cd72064a0b',
    'glassbox-lm/studies/agent_guards/e2/scripts/import_closure.py': 'b49457980b802ab824f7bac3649959e2f917419517fc08a87f1570f5f998f482',
    'glassbox-lm/studies/agent_guards/e2/scripts/make_rows.py': '26891a249b16dbfd2ba933fa591b5713c535838d1190a85b73de47e78047f378',
    'glassbox-lm/studies/agent_guards/e2/scripts/milp_check.py': 'a7a9a5028a777c0bcaab34b5ff1fd21a35f18293991b28e278c3aeeba01d540c',
    'glassbox-lm/studies/agent_guards/e2/scripts/offline_e2e.py': '63ef02d5ff57e19db928e44c5058a1d6afb5e2d52a19c972cca63354a7fc07be',
    'glassbox-lm/studies/agent_guards/e2/scripts/power_e2.py': '8091fe8069aef3bf4f3f8c2763bcd8a9bb8422871d8efeaecf23c4b89d8fd61c',
    'glassbox-lm/studies/agent_guards/e2/scripts/query_bits.py': '988e60999160809db8aaebf8688fc9f783ab51e419300e7f5cbefa02583c1483',
    'glassbox-lm/studies/agent_guards/e2/scripts/report_e2.py': '5bd30d23c4e224b368ff0d51f1a6f19db989c67fe8aa48dadb904a733672fc80',
    'glassbox-lm/studies/agent_guards/e2/scripts/rule_rows.py': '7ee533f52de6adbbb909aed02cd5ef83eab4bc0576fd9b5ae5b1047f3379040d',
    'glassbox-lm/studies/agent_guards/e2/scripts/run_e2_gpu.py': '8a39b2277260291eb8c5b1e680932782f003f116d023302029df0f91529d46c5',
    'glassbox-lm/studies/agent_guards/e2/scripts/seed_check.py': 'a4845b1f2772d41c4e0675ad945e19c58868a85adeef0c01f12428f97c57c236',
    'glassbox-lm/studies/agent_guards/e2/scripts/standin_traces.py': '6814ac10b17b75bbee8c460713b9b4abac2eee7afd1b931ad95ff5837c5f337e',
    'glassbox-lm/studies/agent_guards/e2/scripts/time_fit.py': 'bca4a057c2d095a46d9aa441e68809a5b2ef6eea237f957a4f1f8afb9e08e09e',
    'glassbox-lm/studies/agent_guards/e2/scripts/traces_to_rows.py': '912f20906efa5dbf0af23ae8e8b8fefcdb781b80da9985d41130d72d11409f42',
    'glassbox-lm/studies/agent_guards/e2/scripts/train_loso.py': 'b9661aa5708055ea25e30084fed3e5809cdf6438fa549ba65456deecccdc1584',
    'glassbox-lm/studies/agent_guards/e2/scripts/venv_freeze.py': '8f96e6ae782d91070b174e403af9e58546c128606e56613f8d9d08b6bbeeb50c',
    'glassbox-lm/studies/agent_guards/e2/scripts/veto_share_offline.py': '628a052f48380fc77a77592a0b144e857d601f2922deaf6ac4a9e1c5dbef29d4',
    'glassbox-lm/studies/agent_guards/e2/scripts/wait_cpu.sh': '82b4e87fb94662882c5e7a524afd713f2f1f3a60ae69c003468ff2d0db364f21',
    'glassbox-lm/studies/agent_guards/e2/scripts/write_description_pins.py': 'd518df145c0f1186ec8b5d22c00342ab2e8885ccaa4753adc33ea1a9e6edf2b4',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/__init__.py': 'b4e94df90e2965a5b9eb956ff2336e9ecab3cf35100797eadc12c33f6a9a1118',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/attacks_a3.py': 'e775b5a551d0d0c614cc41b69680735740c0140e24f14c8e43e84d7357557ab3',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/bits.py': 'a40cb02218e14dab58f3edefb3dc61f94a677e1d6f5d72e6346727c687054c25',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/d2_rules.json': '12cc901f2b7221aab18479c2ce4711754613ad934e09e3652381c500191be6fe',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/d6_actgov.py': '62d25d22ac9b7d9840c493292ef17c1a4c5ffc60e35bb2d5f1f7593b751a5303',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/facts.py': 'd04fe546a5b1cefb922fd5b8ee4deb32ea74f62763d16352880ced9904228fd2',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/failures.py': '8bd10eb4fd079687a0200be4ad82850fbabe2aeb4c5fddc7355348db1bac22f3',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/fake_llm.py': '163df08a25f3c4ec8ae3baa2992d207dce654376a0b47aff28239e9004c5e701',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/frozen_clock.py': '1a3cbf9e374d2a8578f4591417317dd0ef2403fc4c618518736f66c50c09935d',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/labels.py': 'a0338d2029ee3acc496280073efd3e8cb8ad96efe49d6086f7736a77278d1443',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/models.py': '7d8205810fe523e1fb23d8f5e2388bc8d118e69f61df1bcb9782cd8833f465b3',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/normalize.py': '0d8c2a410124432d28f2462a577bd425693babc4620c5e3f5ceb8c5aa566d0ef',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/policies.py': '4f29b53dee593438bf3314ba2a182ae3ec6174050ddc50590343a8b81bfff108',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/replay_agents.py': 'a12c795684d3fc6d258250338282bf9c36e2a303b40924778c982eb40534e454',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/replay_store.py': '48dfabb37963d98d528e5353036197b0c0e47fba9f00d1bbdf52f02a8b5d988e',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/rules.py': '67ca5207da340c6fe3618e1ec49f362c903ec752c0dda99c7233a7390cb98342',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/store_llm.py': '64e285cad7b13403588216b752f2887bc96bdf063b8f83da77088c204eccf772',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/tool_table.json': 'c51d7f777ed3a057963b6e9c5dbcb40aa5da31390f0734b6853deb26225bcec9',
    'glassbox-lm/studies/agent_guards/e2/src/e2guard/trace_rows.py': '7d76d3b03f4a16db886eb971cc3b8d255ecc1f2051bedd26889371ef18e933bc',
    'glassbox-lm/studies/agent_guards/e2/tests/check_smoke.py': 'f27ce5c9e5419acfeadd2179f3634fcf6d2c843509dd732696e670d9478df4fd',
    'glassbox-lm/studies/agent_guards/e2/tests/dl_side_tests.py': '48199eb8e08cfad021df3fd7412235019e106376a185be815f56c3375bec555e',
    'glassbox-lm/studies/agent_guards/e2/tests/smoke_e2e.sh': '6941b33bbd871b81f424f787855db62dd8c9b6d791cf372103de236137d3ad0c',
    'glassbox-lm/studies/agent_guards/e2/tests/stub_server.py': 'd21d6cfe49f1053a42d4cf4970b3cd955d115576ad87921e0e9bee2bec2de38a',
    'glassbox-lm/studies/agent_guards/e2/tests/test_facts.py': 'ed12cb2d5ec7af4b878217f50a95c72f8483ecbf145bcd0fd3c345fb28768034',
    'glassbox-lm/studies/agent_guards/e2/tests/test_http_clients.py': 'b99f919ab41a3ad5bc193426666c42ee7039da848c39089789a628760dc75f31',
    'glassbox-lm/studies/agent_guards/e2/tests/test_labels.py': '655f93cd1f5833a541933362361d1100e00a180ec5ed0ce8127831cb49d556c9',
    'glassbox-lm/studies/agent_guards/e2/tests/test_models.py': '569922f4b25b6cd0dab63dcb59dc50dd9926e38ab45a660570e1aa55fe7f07b9',
    'glassbox-lm/studies/agent_guards/e2/tests/test_rev2.py': 'af3ca9cbfabdddb0bd1a79b8089af617a8c1a6462f5975e2a60b4a0f48fd31ef',
    'glassbox-lm/studies/agent_guards/e2/tests/test_rev3.py': 'abf6385192f043bdc820b661c32c0f0e38956721304a8977a36d162b33e53cdc',
    'glassbox-lm/studies/text_understanding/exists_tm.py': '7da0aee9f4eef7969976463f7a8a65319a14466e0235bfcc2500c632107d3942',
    'glassbox-lm/studies/text_understanding/w1/tables/ppw_S1000_d500_e0.5.npz': 'e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a',
    'glassbox-lm/studies/text_understanding/w1/wordtab_feats.py': '001a93b259a1b34f677d317794c3c00c121eb04b0dbf58951cf8b4139c461776',
    'tools/llama.cpp/b11742/build-cuda129/bin/libggml-base.so.0.27.0': '02396b5e005ca26e1731bac7c11e9b4bb0048be49bb1a8ab93425f3bcfefa0c0',
    'tools/llama.cpp/b11742/build-cuda129/bin/libggml-cpu.so.0.27.0': 'd317381205a1a687214aae1344c6bf11dc13f4c6507fb1d9125391a7becc5648',
    'tools/llama.cpp/b11742/build-cuda129/bin/libggml-cuda.so.0.27.0': '5b695d777bdb870e6cb1bcb48d2f9ba02c9a5a2933905c9d37039ca0b6d59d6b',
    'tools/llama.cpp/b11742/build-cuda129/bin/libggml.so.0.27.0': '06a3d6200f8ead292f38e2f265eb657ccf315891f1d69d6728a4d83ac090cc43',
    'tools/llama.cpp/b11742/build-cuda129/bin/libllama-common.so.0.7.0': '9980440e054a95c1ed32fec798640d337cace43e9b1c06fc304bb4e693fe8fc4',
    'tools/llama.cpp/b11742/build-cuda129/bin/libllama-server-impl.so': '22be23afb6ca5593cab0c89eccbab0e8234dc3242b64e1d29113a735efafced4',
    'tools/llama.cpp/b11742/build-cuda129/bin/libllama.so.0.7.0': '52828c64f37a0fab9a4571d044aa39d36f05eb85a1d885e6ca91f7a4297f13cb',
    'tools/llama.cpp/b11742/build-cuda129/bin/libmtmd.so.0.7.0': '5ecccabbb1e1dc8b557ffb2835fc0d9ecdbbd6e613ea889ed1cfa9f02d044765',
    'tools/llama.cpp/b11742/build-cuda129/bin/llama-server': '2fc2870a983daedf21cc248a890752fb761fd9df85e04d68b1264bc61a84652b',
}
# END PINS


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def code_hashes() -> dict:
    """The code a stage runs, as the driver records it in its open meta.json (code_sha256): e2guard's modules and
    tables, every script, the harness modules and (rev4) the job file. freeze_e2.py refuses when the code differs from
    E2-0's record (rev4, round-3 m-14). Keys are relative to harness_build/."""
    h = E2.parent
    files = sorted((E2 / "src" / "e2guard").glob("*.py")) + sorted((E2 / "src" / "e2guard").glob("*.json"))
    files += sorted((E2 / "scripts").glob("*.py")) + sorted((h / "agentdojo" / "harness").glob("*.py"))
    files += [E2 / "run_e2_gpu_slot.sh"]
    return {str(p.relative_to(h)): sha256_file(p) for p in files}


def check_pins(pins: dict[str, str] | None = None) -> list[str]:
    bad = []
    pins = PINS if pins is None else pins
    if not pins:
        return ["no pins written (run scripts/hash_table.py --write-pins)"]
    for rel, want in sorted(pins.items()):
        p = ROOT / rel
        if not p.exists():
            bad.append(f"MISSING {rel}")
        elif sha256_file(p) != want:
            bad.append(f"CHANGED {rel}")
    return bad


def venv_listing(python: Path) -> str:
    env = {"PATH": "/usr/bin:/bin", "HOME": f"{_HOME}", "LANG": "C.UTF-8", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([str(python), "-I", "-B", str(E2 / "scripts" / "venv_freeze.py")], capture_output=True, text=True,
                       env=env, timeout=600, check=True)
    return r.stdout


def check_venvs() -> list[str]:
    bad = []
    for name, py in VENVS.items():
        pin = VENV_PIN_FILES[name]
        if not pin.exists():
            bad.append(f"MISSING venv pin file {pin}")
            continue
        if venv_listing(py) != pin.read_text():
            bad.append(f"CHANGED venv {name} (its package list or a RECORD differs from {pin.name})")
    return bad


# ------------------------------------------------------------------ the server

def http_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def find_process(port: int) -> dict:
    hits = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode("utf-8", "replace") for a in argv if a]
        if not argv or "llama-server" not in Path(argv[0]).name:
            continue
        ports = [argv[k + 1] for k, a in enumerate(argv[:-1]) if a == "--port"] + \
            [a.split("=", 1)[1] for a in argv if a.startswith("--port=")]
        if str(port) in ports:
            try:
                exe = os.readlink(d / "exe")
            except OSError:
                exe = None
            hits.append({"pid": int(d.name), "cmdline": argv, "exe": exe})
    if len(hits) != 1:
        return {"unavailable": f"{len(hits)} llama-server processes listen on --port {port}"}
    return hits[0]


def flag_value(argv: list[str], names: tuple[str, ...]):
    for k, a in enumerate(argv):
        for n in names:
            if a == n:
                return argv[k + 1] if k + 1 < len(argv) and not argv[k + 1].startswith("-") else True
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def server_identity(port: int, hash_gguf: bool = True) -> dict:
    root = f"http://127.0.0.1:{port}"
    h = http_json(root + "/health")
    models = [m.get("id") for m in http_json(root + "/v1/models").get("data", [])]
    pr = http_json(root + "/props")
    gs = pr.get("default_generation_settings") or {}
    props = {"build_info": pr.get("build_info"), "model_path": pr.get("model_path"), "model_alias": pr.get("model_alias"),
             "total_slots": pr.get("total_slots"), "n_ctx": gs.get("n_ctx"),
             "chat_template_sha256": sha256_text(pr.get("chat_template") or ""),
             "chat_template_tool_use_sha256": sha256_text(pr.get("chat_template_tool_use") or ""),
             "modalities": pr.get("modalities"),
             "default_params_sha256": sha256_text(json.dumps(gs.get("params") or {}, sort_keys=True))}
    proc = find_process(port)
    if "cmdline" in proc:
        argv = proc["cmdline"]
        proc["flags"] = {"-np": flag_value(argv, ("-np", "--parallel")), "--jinja": flag_value(argv, ("--jinja",)),
                         "--reasoning-budget": flag_value(argv, ("--reasoning-budget",)),
                         "-fa": flag_value(argv, ("-fa", "--flash-attn")), "-c": flag_value(argv, ("-c", "--ctx-size")),
                         "-m": flag_value(argv, ("-m", "--model"))}
        if proc.get("exe") and Path(proc["exe"]).exists():
            bdir = Path(proc["exe"]).parent
            proc["exe_sha256"] = {n: (sha256_file(bdir / n) if (bdir / n).exists() else None)
                                  for n in SERVER_PINS["exe_sha256"]}
        proc.pop("pid", None)
    gg: dict = {"path": props["model_path"]}
    if props["model_path"] and Path(props["model_path"]).exists():
        gg["bytes"] = Path(props["model_path"]).stat().st_size
        if hash_gguf:
            gg["sha256"] = sha256_file(Path(props["model_path"]))
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "port": port, "health": h.get("status"),
            "models": models, "props": props, "process": proc, "gguf": gg}


def pinned_identity_problems(ident: dict) -> list[str]:
    p, pr, gg = ident["props"], ident["process"], ident["gguf"]
    bad = []
    if ident["health"] != "ok":
        bad.append(f"/health {ident['health']}")
    if SERVER_PINS["model_alias"] not in ident["models"]:
        bad.append(f"/v1/models lacks {SERVER_PINS['model_alias']}")
    if p["build_info"] != SERVER_PINS["build_info"]:
        bad.append(f"build_info {p['build_info']} != {SERVER_PINS['build_info']}")
    if p["total_slots"] != SERVER_PINS["total_slots"]:
        bad.append(f"total_slots {p['total_slots']} != 1")
    if not isinstance(p["n_ctx"], int) or p["n_ctx"] < SERVER_PINS["min_n_ctx"]:
        bad.append(f"n_ctx {p['n_ctx']} < {SERVER_PINS['min_n_ctx']}")
    if gg.get("bytes") != SERVER_PINS["gguf_bytes"]:
        bad.append(f"GGUF size {gg.get('bytes')} != {SERVER_PINS['gguf_bytes']}")
    if "sha256" in gg and gg["sha256"] != SERVER_PINS["gguf_sha256"]:
        bad.append("GGUF sha256 differs from the pin")
    if "unavailable" in pr:
        bad.append("server process: " + pr["unavailable"])
    else:
        for k, want in SERVER_PINS["flags"].items():
            got = pr["flags"].get(k)
            if (want is True and got is None) or (want is not True and str(got) != want):
                bad.append(f"flag {k} = {got!r}, want {want!r}")
        for n, want in SERVER_PINS["exe_sha256"].items():
            if (pr.get("exe_sha256") or {}).get(n) != want:
                bad.append(f"server file {n} differs from the pin")
    return bad


def _argv_without_port(argv):
    out, skip_next = [], False
    for a in argv or []:
        if skip_next:
            skip_next = False
            continue
        if a == "--port":
            skip_next = True
            continue
        if a.startswith("--port="):
            continue
        out.append(a)
    return out


def identity_diff(ref: dict, cur: dict, skip: tuple[str, ...] = ()) -> list[str]:
    """Fields that differ between two identity records (the port may differ between slots; nothing else may)."""
    out = []
    for k in sorted(set(ref["props"]) | set(cur["props"])):
        if ref["props"].get(k) != cur["props"].get(k):
            out.append(f"props.{k}")
    if ref["models"] != cur["models"]:
        out.append("models")
    rp, cp = ref.get("process") or {}, cur.get("process") or {}
    if _argv_without_port(rp.get("cmdline")) != _argv_without_port(cp.get("cmdline")):
        out.append("process.cmdline")
    for k in ("exe", "exe_sha256", "flags", "unavailable"):
        if rp.get(k) != cp.get(k):
            out.append(f"process.{k}")
    for k in ("path", "bytes", "sha256"):
        if f"gguf_{k}" in skip:
            continue
        if (ref.get("gguf") or {}).get(k) != (cur.get("gguf") or {}).get(k):
            out.append(f"gguf.{k}")
    return out


# ------------------------------------------------------------------ the store chain

def store_record(path: Path) -> dict:
    h, n = hashlib.sha256(), 0
    if path.exists():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                n += len(chunk)
    return {"bytes": n, "sha256": h.hexdigest()}


def prefix_ok(path: Path, rec: dict) -> bool:
    want = int(rec["bytes"])
    if want == 0:
        return True
    if not path.exists() or path.stat().st_size < want:
        return False
    h, left = hashlib.sha256(), want
    with path.open("rb") as f:
        while left > 0:
            chunk = f.read(min(1 << 20, left))
            if not chunk:
                return False
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest() == rec["sha256"]


def last_chain() -> dict | None:
    if not CHAIN.exists():
        return None
    lines = [json.loads(x) for x in CHAIN.read_text().splitlines() if x.strip()]
    return lines[-1] if lines else None


def chain_records() -> list[dict]:
    return [json.loads(x) for x in CHAIN.read_text().splitlines() if x.strip()] if CHAIN.exists() else []


def utc_ts(s: str) -> int:
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def stop_log() -> list[dict]:
    return [json.loads(x) for x in STOP_LOG.read_text().splitlines() if x.strip()] if STOP_LOG.exists() else []


def campaign_leftovers() -> list[str]:
    """E2-0 starts a fresh campaign: runs/ may hold only the current E2-0 run folder (the job file creates it just
    before this check). Anything else (a store, a chain, an identity record, a gate, a budget plan, a freeze record, a
    stop log, another stage's or another E2-0's folder) belongs to an earlier campaign, which is moved aside unopened."""
    if not RUNS.exists():
        return []
    out = []
    for p in sorted(RUNS.iterdir()):
        if p.name == "E2-0" and p.is_dir():
            subs = sorted(q.name for q in p.iterdir())
            if len(subs) > 1:
                out.append(f"E2-0/ holds {len(subs)} run folders")
            continue
        out.append(p.name)
    return out


def pending_stop() -> dict | None:
    """The last logged stop if its stage has not been rerun since (None when every stop was rerun)."""
    log = stop_log()
    for k in range(len(log) - 1, -1, -1):
        e = log[k]
        if e["kind"] == "stopped":
            return None if any(x["kind"] == "rerun" and x["stage"] == e["stage"] for x in log[k + 1:]) else e
    return None


def check_frozen(now: int | None = None) -> list[str]:
    """Every stage after E2-0: the write-once freeze record holds, the registration is unchanged and names this
    checker's own sha256, the freeze time precedes the stage start and follows E2-0's end, and E2-0's identity record,
    gate, budget plan and summary copy are unchanged since the freeze. When all hold, the record's sha256 is printed."""
    if not FROZEN.exists():
        return ["no runs/E2_frozen.json: the registration is not frozen (scripts/freeze_e2.py, after E2-0)"]
    fr = json.loads(FROZEN.read_text())
    reg = ROOT / fr["registration"]
    if not reg.exists():
        return [f"the frozen registration {fr['registration']} is missing"]
    if sha256_file(reg) != fr["registration_sha256"]:
        return ["the registration changed since the freeze (its sha256 differs from runs/E2_frozen.json)"]
    if sha256_file(Path(__file__)) not in reg.read_text():
        return ["this checker's own sha256 is not in the frozen registration (Appendix C): the checker changed"]
    now = int(time.time()) if now is None else now
    if not utc_ts(fr["utc"]) < now:
        return [f"the freeze time {fr['utc']} does not precede this stage's start"]
    e20 = [r for r in chain_records() if r.get("stage") == "smoke"]
    if not e20 or utc_ts(fr["utc"]) < utc_ts(e20[0]["utc"]):
        return ["the freeze record does not follow E2-0's end record"]
    for name, path in (("identity_ref_sha256", IDENTITY_REF), ("gate_sha256", SMOKE_GATE), ("budget_sha256", BUDGET),
                       ("e2_0_summary_sha256", E2_0_SUMMARY)):
        if not path.exists() or sha256_file(path) != fr.get(name):
            return [f"runs/{path.name} changed since the freeze (the E2-0 plan and records are final)"]
    # on record in the stage's slot.log: the freeze record this stage starts under
    print(f"freeze record ok: runs/E2_frozen.json sha256 {sha256_file(FROZEN)} (frozen at {fr['utc']}, before this "
          "stage's start)")
    return []


def begin(stage: str, port: int | None, skip_server: bool = False, rerun: bool = False) -> list[str]:
    bad = check_pins() + check_venvs()
    if bad:
        return bad
    stop = None
    if stage == "smoke":
        if rerun:
            return ["E2-0 is never rerun on its campaign: move runs/ aside unopened and start a fresh campaign"]
        old = campaign_leftovers()
        if old:
            return [(f"runs/ holds files of an earlier campaign ({', '.join(old[:6])}): E2-0 starts a fresh campaign; "
                     "move runs/ aside unopened (never delete it)")]
    else:
        if not IDENTITY_REF.exists():
            return ["no runs/server_identity_ref.json: E2-0 has not run"]
        gate = json.loads(SMOKE_GATE.read_text()) if SMOKE_GATE.exists() else None
        if gate is None or gate.get("go") is not True or gate.get("fake") is not False:
            return ["E2-0's stop/go gate (runs/E2-0_gate.json) is missing, not go, or from a fake run"]
        bad = check_frozen()
        if bad:
            return bad
        last = last_chain()
        if last is None:
            return ["no store chain: E2-0 has not ended"]
        if not prefix_ok(STORE, last):
            return ["the replay store's history differs from the chain's last record"]
        stop = pending_stop()
        if rerun:
            if stop is None or stop["stage"] != stage:
                return [f"no logged stop of stage {stage} waits for its rerun (a stage is rerun once, after a logged stop)"]
            if any(x["kind"] == "rerun" and x["stage"] == stage for x in stop_log()):
                return [f"stage {stage} was rerun once already: it stays incomplete"]
            if store_record(STORE) != {"bytes": stop["bytes"], "sha256": stop["sha256"]}:
                return ["the replay store differs from its record at the stop (bytes were written after the stop)"]
        else:
            if stop is not None and any(x["kind"] == "rerun" and x["stage"] == stop["stage"] for x in stop_log()):
                return [(f"stage {stop['stage']} stopped again in its one rerun ({stop['run_dir']}): it stays "
                         "incomplete; E2 stops here and the failure is logged")]
            if stop is not None:
                return [(f"stage {stop['stage']} stopped ({stop['run_dir']}) and has not been rerun: rerun it first "
                         "(--rerun); no other stage starts before")]
            if store_record(STORE) != {"bytes": last["bytes"], "sha256": last["sha256"]}:
                return ["the replay store changed between stages (bytes were added outside a stage)"]
    if stage == "build" and (MODELS_FINAL.exists() or MODELS_FINAL.with_name("final.d6tmp").exists()):
        return [("models/final (or models/final.d6tmp) exists before the build stage (never overwritten; a stopped "
                 "build's d6tmp is moved into its run folder)")]
    if stage in ("arms", "a2", "a3"):
        bad += check_models_final()
        if bad:
            return bad
    if not skip_server:
        if port is None:
            return ["--port is required"]
        ident = server_identity(port, hash_gguf=True)
        bad = pinned_identity_problems(ident)
        if bad:
            return bad
        if stage == "smoke":
            RUNS.mkdir(parents=True, exist_ok=True)
            IDENTITY_REF.write_text(json.dumps(ident, indent=1))
            print(f"wrote {IDENTITY_REF} sha256 {sha256_file(IDENTITY_REF)}")
        else:
            ref = json.loads(IDENTITY_REF.read_text())
            d = identity_diff(ref, ident)
            if d:
                return [f"server identity differs from E2-0's record: {d}"]
            print(f"server identity = E2-0's record (sha256 {sha256_file(IDENTITY_REF)})")
    if rerun:
        with STOP_LOG.open("a") as f:
            f.write(json.dumps({"kind": "rerun", "stage": stage, "rerun_of": stop["run_dir"],
                                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        print(f"rerun of stage {stage} logged (rerun of {stop['run_dir']})")
    return []


def models_final_hashes() -> dict:
    return {"models/final/" + str(f.relative_to(MODELS_FINAL)): sha256_file(f)
            for f in sorted(MODELS_FINAL.rglob("*")) if f.is_file()}


def check_models_final() -> list[str]:
    if not MODELS_PINS.exists():
        return ["no runs/models_final_pins.json: the build stage has not ended"]
    want = json.loads(MODELS_PINS.read_text())
    got = models_final_hashes() if MODELS_FINAL.exists() else {}
    return [f"models/final differs from the build stage's record: {sorted(set(want.items()) ^ set(got.items()))[:4]}"] \
        if got != want else []


def end(stage: str, run_dir: str) -> dict:
    rec = {"stage": stage, "run_dir": run_dir, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           **store_record(STORE)}
    log = stop_log()
    if log and log[-1]["kind"] == "rerun" and log[-1]["stage"] == stage:
        rec["rerun_of"] = log[-1]["rerun_of"]
    if stage != "smoke" and FROZEN.exists():  # rev3: the freeze record this stage started under, on record
        rec["frozen_sha256"] = sha256_file(FROZEN)
    RUNS.mkdir(parents=True, exist_ok=True)
    if stage == "build" and MODELS_FINAL.exists():
        assert not MODELS_PINS.exists(), "runs/models_final_pins.json exists (written once)"
        MODELS_PINS.write_text(json.dumps(models_final_hashes(), indent=1))
        rec["models_final_pins_sha256"] = sha256_file(MODELS_PINS)
    with CHAIN.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    if run_dir:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "store_end.json").write_text(json.dumps(rec, indent=1))
    return rec


def stopped(stage: str, run_dir: str, exit_code: int) -> dict:
    """Log a transport stop (the job file calls this when the driver or build_d6.py exits with 3): no chain record."""
    rec = {"kind": "stopped", "stage": stage, "run_dir": run_dir, "exit_code": exit_code,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **store_record(STORE)}
    RUNS.mkdir(parents=True, exist_ok=True)
    with STOP_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pins", action="store_true")
    ap.add_argument("--venvs", action="store_true")
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--begin", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="with --begin: the one rerun of a stopped stage")
    ap.add_argument("--end", action="store_true")
    ap.add_argument("--stopped", action="store_true", help="log a transport stop of --stage (--run-dir, --exit-code)")
    ap.add_argument("--exit-code", type=int, default=3)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--skip-server", action="store_true", help="code tests only: no server identity check")
    a = ap.parse_args()
    bad: list[str] = []
    if a.pins:
        bad += check_pins()
        print(f"pins: {len(PINS)} files, {len(bad)} problems")
    if a.venvs:
        b = check_venvs()
        bad += b
        print(f"venvs: {len(VENVS)}, {len(b)} problems")
    if a.begin:
        assert a.stage, "--stage is required"
        bad += begin(a.stage, a.port, a.skip_server, rerun=a.rerun)
    if a.end:
        assert a.stage, "--stage is required"
        print(json.dumps(end(a.stage, a.run_dir)))
    if a.stopped:
        assert a.stage and a.run_dir, "--stage and --run-dir are required"
        print(json.dumps(stopped(a.stage, a.run_dir, a.exit_code)))
    for b in bad:
        print("STOP:", b)
    print("e2_check:", "OK" if not bad else f"{len(bad)} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
