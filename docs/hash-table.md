# Hash table

Several records of the studies pin files by SHA-256: configs pin source files, a hash file pins the
scripts and data of a frozen analysis, run records hold the hash of the pre-registration, a result
document quotes the hashes of two patched files, and in decision_layer and tally a freeze record holds
the hash of each frozen text and of the code at the time of the freeze. Some pinned files were reworded
for publication: the code in comments, docstrings, message strings and absolute path constants only,
the frozen texts in wording only. This page gives, for every pinned file that is part of the
repository, the SHA-256 as registered and the SHA-256 of the file as published.

For tm_language_model, hierarchical_tm and clause_lm, where the two differ and the pin itself is a file
of the repository, the pin was replaced by the published value, so that the checks in the code still
compare equal values. The freeze records of decision_layer and tally are published unchanged, because
each records a freeze at a stated time; their sections give both values side by side. The one pin in
the code of these two studies that checks a file, `SAMPLE_SHA` in
`studies/decision_layer/scripts/e71_k35_sample.py`, holds the published value.

## The pre-registration of tm_language_model

| file | SHA-256 as registered (last revision, recorded in run records) | SHA-256 as published |
|---|---|---|
| `studies/tm_language_model/docs/13-preregistration.md` | `0fd3fb8b1c47277d166ec2117e4c0e40477348d605c6959d8a0210e8a93d4938` | `805ce0da83be57b1048a419d738701f7231fa84b258ec11a4f50c1ff5502c9e9` |

The registered value is the `prereg_sha256` of 384 of the 643 run records of tm_language_model and of
all 319 run records of hierarchical_tm. The published copy differs in wording only; the note at the top
of the file says what was edited. Ten earlier revisions, recorded by 249 run records, were not retained.

## Pinned files in the repository

| file | pinned by | SHA-256 as registered | SHA-256 as published | note |
|---|---|---|---|---|
| `studies/tm_language_model/src/tmlm/cvtm/__init__.py` | cvtm source, pinned in 7 configs (cvtm_sources_sha256) | `26331ed20da275ea86a35e9d3f7f5cb2e93d7e73a3048bd21d3adc2259af150e` | `c5d9259ac33bf0bcf2c288746aa4ede76f2f7006f3857995b28693998abe3630` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/tm_language_model/src/tmlm/cvtm/cache.py` | cvtm source, pinned in 7 configs (cvtm_sources_sha256) | `7366c5dd5d488e1988a2439ed08687a810f4178dc855525d37f7de0da13d866d` | `7366c5dd5d488e1988a2439ed08687a810f4178dc855525d37f7de0da13d866d` | unchanged |
| `studies/tm_language_model/src/tmlm/cvtm/doc_clauses.py` | cvtm source, pinned in 7 configs (cvtm_sources_sha256) | `aa3287b56dd9630113357abc781df6ac8359d03fc6368d658c6b2ae72f0f7453` | `e835c70904e9a0f28c4e2d54425249bfb71273c08f6a1c4e10827faff44d1f40` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/tm_language_model/src/tmlm/cvtm/energy.py` | cvtm source, pinned in 7 configs (cvtm_sources_sha256) | `852f53fa9dc947a41e19b6c9bffde4f98009d58c53f0cd32efc58172fb8c4f60` | `852f53fa9dc947a41e19b6c9bffde4f98009d58c53f0cd32efc58172fb8c4f60` | unchanged |
| `studies/tm_language_model/src/tmlm/cvtm/kn_gpu.py` | cvtm source, pinned in 7 configs (cvtm_sources_sha256) | `582507f75a8a792f9716bdaed3f12188ac78200cfe5b74e1bc6472b10ba97fc4` | `582507f75a8a792f9716bdaed3f12188ac78200cfe5b74e1bc6472b10ba97fc4` | unchanged |
| `studies/tm_language_model/src/tmlm/cvtm/mixture.py` | cvtm source, pinned in 7 configs (cvtm_sources_sha256) | `e8ff8ed4cb1e55c89b55128e3a40f12c1cd229a933a25d83b3e7f52b6d1b630d` | `e1522c60b1b722ab53bb5990b2a21a606a172bcda3c1f1bba57ada1d3473cf11` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/tm_language_model/src/tmlm/cvtm/model.py` | cvtm source, pinned in 7 configs (cvtm_sources_sha256) | `27200cc29c167dd0517a82e56ff475aa22ff30a6866384c77e9f8ca7d1f2f0d8` | `27200cc29c167dd0517a82e56ff475aa22ff30a6866384c77e9f8ca7d1f2f0d8` | unchanged |
| `studies/tm_language_model/src/tmlm/cvtm/protocol.py` | cvtm source, pinned in 7 configs (cvtm_sources_sha256) | `ef0cc0c0d4f86a8c34f0f02f4c44228e259690c8b5f241db88b61cebbcea3e5a` | `ef0cc0c0d4f86a8c34f0f02f4c44228e259690c8b5f241db88b61cebbcea3e5a` | unchanged |
| `studies/tm_language_model/src/tmlm/cvtm/suffix.py` | cvtm source, pinned in 7 configs (cvtm_sources_sha256) | `ea16dde565031d4086224411b3f81b610e08dbd2faf3a539a5fefbdcaa824d6f` | `ea16dde565031d4086224411b3f81b610e08dbd2faf3a539a5fefbdcaa824d6f` | unchanged |
| `studies/tm_language_model/experiments/cvtm_t1_readout.py (the kernel source string T1_SRC)` | kernel string T1_SRC in experiments/cvtm_t1_readout.py, pinned by tests/test_ssl_r1_qpred_fast.py | `0a16974f13f031d11540d57cb796dcf371e162699ad6d56aedb201ae3bdcbd2c` | `0a16974f13f031d11540d57cb796dcf371e162699ad6d56aedb201ae3bdcbd2c` | pinned by a test; the string is unchanged |
| `studies/tm_language_model/src/tmlm/kernels/graphtm_patched/kernels.py` | stock kernels.py, pinned in kernels/graphtm_patched/README.md | `9d97a1fb3a586beda467d9749b41d9a3f35bf42679450bfedc55081f7860332a` | `9d97a1fb3a586beda467d9749b41d9a3f35bf42679450bfedc55081f7860332a` | unchanged |
| `studies/hierarchical_tm/queue/run_queue_mps.sh` | script, pinned in frozen_settings_hashes.json | `57c69a5585eaabd39eeeea04a21a5b6900e89e9a1992936b8b903cd2e45886e4` | `dae7738a023d02d812ee38ac4fe583e565e67d463aca0c4f2c276fe61c9663d9` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/apply_rules.py` | script, pinned in frozen_settings_hashes.json | `a3b74277803649091fb71d9e4170b0d747009924e4b1422eaa265f6bf847da5d` | `a3b74277803649091fb71d9e4170b0d747009924e4b1422eaa265f6bf847da5d` | unchanged |
| `studies/hierarchical_tm/src/build_agnews.py` | script, pinned in frozen_settings_hashes.json | `4131b5e0bab00f61d0c79c4883121632517ef11d78ffee9fbb944e0ffd1b8ffc` | `c4fb7a262a8cf2f45e2234369b01ca2a29945906c39fcd07d5dbb6d278c07410` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/build_ns_all.py` | script, pinned in frozen_settings_hashes.json | `e79070ee0770e3aed4fc907fe173157fa4f4db7973280da9162bc872c7557b65` | `d02794e933a16edbc329b4110634d577a8407430b910736663d202cb5fc2ffa5` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/build_pairs.py` | script, pinned in frozen_settings_hashes.json | `44ba48433ec71f3ca16ad5a853ebe99131e67d8e13610a7862d008bba46467a0` | `8a1c56c95a11943ea0e2d0871ed605816030b72e02973a324c06e3e404aae56e` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/compare_runs.py` | script, pinned in frozen_settings_hashes.json | `33d5530610503827ba9ae96be6750063b7f6b07c59c4cb153ef150ad06451a83` | `33d5530610503827ba9ae96be6750063b7f6b07c59c4cb153ef150ad06451a83` | unchanged |
| `studies/hierarchical_tm/src/crossfit_acc.py` | script, pinned in frozen_settings_hashes.json | `493b5813a7c70b74e4da4e51ce8945907bce8007ad4cadd57d0a28520c4cb828` | `5f9063f03e5a528d44790bb5b0eeb0cdb996d4e3ed173905587cf920bb419ac8` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/crossfit_splits.py` | script, pinned in frozen_settings_hashes.json | `2ea2e08cd8b657a10f4ebcae625153ec5ea3aa105956b57d298ef385ab5c36ae` | `d73a8ca89221c3084cd08d577a0c7b900b7afc1a4c39746364b0ca41fdaa402a` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/data_checks.py` | script, pinned in frozen_settings_hashes.json | `72b0d717d354c9e7e0e6dd2a8a3bf6c0fc285f43f6be4128d479cdabbd17ca4c` | `896f2bdc7505fbd1dd965cd01c0da2e4876f7f4068d72c3971ca6b1395a26447` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/e5a_check2c.py` | script, pinned in frozen_settings_hashes.json | `3d240eb5e7a606dc76e89aaf1254f5e750319c110e625b04bb8b6857bc895a00` | `5bfda94cbc9e39839480fc2e42e1b67cb79f95152a965932b99a56247b01046e` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/e5a_choice.py` | script, pinned in frozen_settings_hashes.json | `3381aad3a00499e31ca6bcbe3fd0d81dd534e5577f3578aac3eed2724cdd4691` | `eb9da9dae6d16d80504769578ee06c99b2d69a4d97a57b909589fac50df5a7e3` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/final_test_report.py` | script, pinned in frozen_settings_hashes.json | `eff80607a13d739144e829b7e90fec1ec8c9d9d23c418df8046e94f7ef72d245` | `ca6a1ca38bfca82cb8931d3cd25fd5e252059574ee2585d66898553a5ecbdfb3` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/htm_driver.py` | script, pinned in frozen_settings_hashes.json | `dc31407e1057b2ecbf85de469d59d5bd645d8778d605dd7550fdd5c52707df01` | `0073b2a4bd61a3b86bdac07f7a3022efeaa401a8fec1e09f65927f910fd0f478` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/npmi_groups.py` | script, pinned in frozen_settings_hashes.json | `8989262ab621ca51102f66b32c2b466617f8f83cd03766813da010ae25f89635` | `4c93f9a10040c183a677baf8743f1b9814272361bf2d96ed2ab84345ae1130b1` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/pairdata.py` | script, pinned in frozen_settings_hashes.json | `82d98e94f8546c46740c42456557fc45ec270e82befada4ac75a68b38bd4b1e0` | `82d98e94f8546c46740c42456557fc45ec270e82befada4ac75a68b38bd4b1e0` | unchanged |
| `studies/hierarchical_tm/src/pcp_purity.py` | script, pinned in frozen_settings_hashes.json | `77bcba68faf47f8458c274e66bef3eec1738c8a3639525d855ed4a4969a57d03` | `77bcba68faf47f8458c274e66bef3eec1738c8a3639525d855ed4a4969a57d03` | unchanged |
| `studies/hierarchical_tm/src/pcp_recovery.py` | script, pinned in frozen_settings_hashes.json | `e1f0f0fdf7b5f831e0f2f393f273ffb37861d60d6db6ff1cbe493bf8eb1fe6f9` | `e1f0f0fdf7b5f831e0f2f393f273ffb37861d60d6db6ff1cbe493bf8eb1fe6f9` | unchanged |
| `studies/hierarchical_tm/src/score_saved.py` | script, pinned in frozen_settings_hashes.json | `6633a80cb0e5afc92261c05dc3e666a722ac87974a8858c2daa7a5dd00f4c55e` | `6633a80cb0e5afc92261c05dc3e666a722ac87974a8858c2daa7a5dd00f4c55e` | unchanged |
| `studies/hierarchical_tm/src/slide10.py` | script, pinned in frozen_settings_hashes.json | `e50a6a40b59d1530eb9282d1e00f8b169c0de0f5df791b285b7a2e89efb6c5a3` | `00c0d0812b472f0b38532bd3296f8944956b7fc1e7e8c07d5111a0a7fd81b6f8` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/hierarchical_tm/src/text_driver.py` | script, pinned in frozen_settings_hashes.json | `8ee7063b229a0bb739ac83ffe921a1863f48b28ab72b64a1d8bb9f9d6587f327` | `6f5a2f3a574c8dbbc0db4688cdd35e687dbbee05f9c30a0c34a0e5e59b63620f` | the pin disagreed with the file before publication (the file had SHA-256 `51e339cf84ec80dec4c3865223b33ae2ef00b5b88933a6a9939d0eec697d9d50` then: it was changed after the freeze, which the study's record discloses); the pin is published as registered and was not repaired |
| `studies/hierarchical_tm/src/wiki_groups.py` | script, pinned in frozen_settings_hashes.json | `d40e185690e410f1cba440085cec09cb1f2eb06a0540e64c2bb6b97627905074` | `7ce4a9fc34313e6753f8d51c6cf2632ba2dc2e37318414d33e5b3ac9f4246a6d` | reworded for publication (comments, docstrings, message strings or absolute path constants); the pin in the tree was replaced by the published value |
| `studies/clause_lm/src/clause_driven_llm/cdllm_data__init__.py` | added loader, recorded by m5_setup.py and m5_run.py | `b851309171cb4d6d2615a84b90437a9c16a7f5d8e8812f62dfa11579b2a07e5e` | `b851309171cb4d6d2615a84b90437a9c16a7f5d8e8812f62dfa11579b2a07e5e` | unchanged (byte identical copy of the loader that the runs recorded) |

## decision_layer

The study writes a freeze record next to every frozen registration, dev plan and dated note: 33 in
`studies/decision_layer/registrations/` and 30 in `studies/decision_layer/logs/`, 63 in all. Each holds the UTC time
of the freeze, the SHA-256 of the frozen text and the SHA-256 of every file of `scripts/` and `src/` as
it stood at that moment. Before publication every frozen text equalled the value in its record. The
texts and the code were edited for publication on 2026-10-23, in wording and paths only. Each edited
copy of a frozen text carries a publication note, and the check scripts that compare a text with its
record report a difference on these copies ([study README](../studies/decision_layer/README.md)).

### Frozen texts

44 of the 63 frozen texts were edited for publication; the other 19 are published as frozen, byte
for byte.

| file | freeze record | SHA-256 as registered | SHA-256 as published |
|---|---|---|---|
| `studies/decision_layer/registrations/R10_abcd_reward_only.md` | `registrations/R10_abcd_reward_only.frozen.json` | `89a1cdc193495812864e9d684d49744ad04d354b7d875808e38619b08a2ae4be` | `0c423a352f72d22815184367625902ef96a3305d4b102a4746d24ce5f4fa07f1` |
| `studies/decision_layer/registrations/R10c_abcd_linucb.md` | `registrations/R10c_abcd_linucb.frozen.json` | `5d2561a67c29766b3115f291a7a7aa362dbe904ced51066a087218b28bc6efde` | `36854d7c4573cfd65be9e7a5a62af545fbced9634e64f037b7d8ad5dcf799bfb` |
| `studies/decision_layer/registrations/R14_abcd_neural_bandit.md` | `registrations/R14_abcd_neural_bandit.frozen.json` | `08536c978d2f8fbb4ecc0d88e806d38d2a0def7ae89661db792a6357c5014bc3` | `4b967ca4367f67860d9c691bf27a7e88712e720ea11a2638d7db4f5a61b4063e` |
| `studies/decision_layer/registrations/R15_bandit_replay.md` | `registrations/R15_bandit_replay.frozen.json` | `7f7a8b808fe369462f24ba1c0332c3eff36c09cbb01044b3f7da8e8f9c6bbfef` | `0ea85cea9b46bf39ea1b047d00c87995e07eac218ac5108c5fc7eb686936e5ac` |
| `studies/decision_layer/registrations/R16b_flat_matched_size.md` | `registrations/R16b_flat_matched_size.frozen.json` | `c8114b22d4dcc55dc62c726461a437bf969b5f04669d47452c22632b385ed89e` | `13b290b512346ae79553afc83ef71cedf2603f0fd221d190731dff5978aa80d2` |
| `studies/decision_layer/registrations/R17_bandit_closed_loop.md` | `registrations/R17_bandit_closed_loop.frozen.json` | `7c2a3a5ea279acae319e4817e8d305ec7898421348ba467e12b979d6255b7cde` | `615248512bd2de8f26cbe6fb90e9fc3c866e9bdbc9b087e954a5336f4a67730e` |
| `studies/decision_layer/registrations/R18_star_replication.md` | `registrations/R18_star_replication.frozen.json` | `207f4202e9c27450e42d2d23de23b264184213dfbd526191bfda9d2989efa43f` | `4b743bc94b31a84ac37271a95cad01fb4faa67ffdd00fbce8f10f07e35e6e9c1` |
| `studies/decision_layer/registrations/R19_no_network_agent.md` | `registrations/R19_no_network_agent.frozen.json` | `b743819a784f2dfed0ec2ee80f0d3f552a055329d11393f929174584b5375af8` | `3b001ae846e36d165f137a6ef6ae29c1f64f3e19d14c55ca496b221d599ba137` |
| `studies/decision_layer/registrations/R20_customer_only_closed_loop.md` | `registrations/R20_customer_only_closed_loop.frozen.json` | `a4fc1418cfe68cf0be06081cb8d7d33700b21d56578fb58fc50ed4c42772501b` | `f37de6b487ec57d9896ede9dc1bf961276c9017870980f50947a90ff8e82d280` |
| `studies/decision_layer/registrations/R21_c4_table_agent.md` | `registrations/R21_c4_table_agent.frozen.json` | `bc4a2d1e3d751726f5e112c6257f689c69a427278c330a8f35857192369ca2f2` | `6dea347f0f03d6b498924f11bc52e772e5fe00452a0f0d92ced4e5df35fb3be2` |
| `studies/decision_layer/registrations/R22_tuned_s_star.md` | `registrations/R22_tuned_s_star.frozen.json` | `de7794c5407d3b29f91c8f7b6f3dec8d029a38b8db8379be97a91c9fd563fa39` | `1341faadbece8c9f4f26c0111ca81e62cb93ca253bb355cc422fcc11f71fe325` |
| `studies/decision_layer/registrations/R23_tuned_s_abcd.md` | `registrations/R23_tuned_s_abcd.frozen.json` | `0d54d187441c7eb3e196f6f3c864a11bfa7fe56dc845fb2321cfa18d419b42c2` | `6c909b095c5cf5cf0b8563262bf5aedd089fa4a545c9c5973ac469a04d470f45` |
| `studies/decision_layer/registrations/R24_multiwoz_random_negatives.md` | `registrations/R24_multiwoz_random_negatives.frozen.json` | `4d675dde55a158603fd0b6f9cb1dc780dac2f2fd5539564a695b121153991e20` | `a53ee9310fc6d665af00de541a24f95e5fb7b955e652e2b35df10a547f04c3f2` |
| `studies/decision_layer/registrations/R9_abcd_rules_over_sequences.md` | `registrations/R9_abcd_rules_over_sequences.frozen.json` | `ed0efe8edb882f79d8671acdd033c1f7b369b27bbdc2c91ea02a3b254c8eddba` | `40c6f1a912a518982600cd127908ba57d60ba6b8e66bf81c3f50c0ba8b8422bd` |
| `studies/decision_layer/registrations/R9b_memory_clean_control.md` | `registrations/R9b_memory_clean_control.frozen.json` | `15124997bb8615681708702561f227c73e589dac6a4821e916014e4b76b1f3bf` | `b70935fc7d2d51f832062bc790c73d47bdc44b68b949f0ac6560f8167636874f` |
| `studies/decision_layer/logs/plan_e60_static.md` | `logs/plan_e60_static.frozen.json` | `d0a4eb820533ebedfa2fdfe9eda14006a49d6d2d785afb096ac23123f7b45dcf` | `11528c20df4656e1d48793f40f3fe6d9614e208419d2bfbdf74379f4cdd39cd8` |
| `studies/decision_layer/logs/plan_e70_verify.md` | `logs/plan_e70_verify.frozen.json` | `271eabf8177cb4331198762cfad902e0d9fa327228267c72530faaf2fcc939a8` | `92fa4b3b63d511aa87d2f0ddb466a2e29c47330ccfd0e5a79e1e2b4dd82cf93b` |
| `studies/decision_layer/logs/plan_e71_local.md` | `logs/plan_e71_local.frozen.json` | `76fac7b0b6940c5cf4e81a06607999a201a8befbfbf8629ef4aa8c370481b5e9` | `0aec263742c2c3edf3ca570c6c2f0826af5183b4969bb2a1df6cc82620c90a5a` |
| `studies/decision_layer/logs/plan_e71_v2_amendment.md` | `logs/plan_e71_v2_amendment.frozen.json` | `3353e99779d5845ae3fb673b5d9318f0d94e75e654ab1c8fc91dca6ca93fd132` | `4a2a2cf246a8e1a92956f38ade8d8e4c47743a4ab900399d219ad416cbabc2d7` |
| `studies/decision_layer/logs/plan_e71_v2_note2_2026-10-19.md` | `logs/plan_e71_v2_note2_2026-10-19.frozen.json` | `c46bfe3833464d49605ff04f70fe0d46e541a32b2ebdcb91051096b2ce7645df` | `00253377b4880c50707103fdf92f9199db07348da2d9bb5bf027fae408639a43` |
| `studies/decision_layer/logs/plan_e72_count_table.md` | `logs/plan_e72_count_table.frozen.json` | `1f800fad11299f264ec45089993e2ab13dbb506a77d2e9ef8ac23365217344ce` | `d1c077d3ff8b0c5329f7cabe8af18ae96785ab5d507e91d4f915570cba30de65` |
| `studies/decision_layer/logs/plan_e72d_guards_c4.md` | `logs/plan_e72d_guards_c4.frozen.json` | `a235720e55a981331853a21b1a388ca0967c01da7ce5a80c31bfa2063c95d91c` | `a80ea46c1244efea99a5b02fc5052bf078eaac75c8b24e907bf4d8420b32bbfc` |
| `studies/decision_layer/logs/plan_e72e_guards_c4_s16.md` | `logs/plan_e72e_guards_c4_s16.frozen.json` | `e591b9895dced866c75ff3e85b4991c41cdf1e4165096574f814c9fc4011a964` | `89486a06f4f401f0a522406b703be57cfbec8e2ece4a14f038e45fa1d116c6f9` |
| `studies/decision_layer/logs/plan_e73_replies_star.md` | `logs/plan_e73_replies_star.frozen.json` | `4f7582f359def1759f9235b84bca72586c56d786f016b799fe00f96ac96ef965` | `6032e6c49552b4dc3dd714f4d6847fdfc5c8ca626bbb43703871ab6e77fcec36` |
| `studies/decision_layer/logs/plan_e74_s_sweep.md` | `logs/plan_e74_s_sweep.frozen.json` | `2a183a80e71541a5bdc3dce4ee83b901457336770204b823b0d414a64bc32270` | `4316d31347836a7b62da383b51a07e2226e92087012eac1609300428c24dd763` |
| `studies/decision_layer/logs/plan_e74cd.md` | `logs/plan_e74cd.frozen.json` | `283ce101ca26b6fdd815fb8dd7ee1a494fbf6e358f0cbe20d375f693736acc42` | `42518ce13b4c70635996c8120338a8d81fd79f3ec8be3fdb211ea80186df3e7a` |
| `studies/decision_layer/logs/plan_e75_abcd_s.md` | `logs/plan_e75_abcd_s.frozen.json` | `c967031010c4fbdfe56c07766ca78fd4673987f714a616a04e4348e6d1891c5d` | `4911939bb98fc84d648051da63d1544abcb75d06deeb445b4ed727863e851e60` |
| `studies/decision_layer/logs/plan_e76_wordedit_c4.md` | `logs/plan_e76_wordedit_c4.frozen.json` | `5ad065a53ae1ddb26a22ac4491c62d896cdb63cfafefda105086bbcc38773bbd` | `2ee4b975e2d6c52e1607ff3848170ae05d56acf1cdc459bfd63e13fcbf46c570` |
| `studies/decision_layer/logs/plan_e77_multiwoz_dev.md` | `logs/plan_e77_multiwoz_dev.frozen.json` | `6410ca234d603ac73bce5ea72915087c060cf591d3939af52e3a3665f10d9cbf` | `d6ee0cec1d27aa3575f3bc943dc25c39af2b0850b8d7c07b6baf780010f10ab6` |
| `studies/decision_layer/logs/plan_e78_2d_counts.md` | `logs/plan_e78_2d_counts.frozen.json` | `d80a9c1bc2fd134ab90c0cda6325749c66ff3c0535f07f0797c7d8c14011637f` | `6788049ddcb358495bb16c033b16b483b128bb9e707217f9acdcb0d9a764e506` |
| `studies/decision_layer/logs/plan_e79_c4_prototypes.md` | `logs/plan_e79_c4_prototypes.frozen.json` | `a040045d765a41a336b1b8c6bc8ac6073d6ec2ed1530565952b4cae58896bc4f` | `3b55dca08da0d3f19b0ade827ed55bc1eb40f960f5bac798fc99923c1f6e297e` |
| `studies/decision_layer/logs/plan_e80_2d_recent.md` | `logs/plan_e80_2d_recent.frozen.json` | `d045ad4a951a791da77ad0a9a6edfe8f5c4b514fe176d4d5e45c1b93ad4bdb2e` | `6d0c52ce6881c9d14e1530a3d090e13787342a62f5e00038e83745e08421a241` |
| `studies/decision_layer/logs/plan_e81_T_screen.md` | `logs/plan_e81_T_screen.frozen.json` | `c58c502aac6fc0c22e597241e80681678af51b6ac32b18fbf80ccd009e332c16` | `d96bc2ed2c2e0e53b3aaaaa2732eff5330bdcccc4d46632ee5991326c927a641` |
| `studies/decision_layer/logs/plan_e81_note2_2026-10-21.md` | `logs/plan_e81_note2_2026-10-21.frozen.json` | `82103b3296c74ce186de07b3dd231ccb6abbf60e4f43525309cdd50b401212e8` | `1696d79c2fa9cae0d3c3e602eca33b322d9e63e3dad5fbebdd0242e26bce1285` |
| `studies/decision_layer/logs/plan_e81_note_2026-10-21.md` | `logs/plan_e81_note_2026-10-21.frozen.json` | `690b7b69a26faf81428a546918c073e49445ae0372279ac25acef2f1c5be730a` | `a35d0deca8dda127199d123d3a6d5ddd2a6b7822520b5e007a34547dc947b781` |
| `studies/decision_layer/logs/plan_e82_r5_guards.md` | `logs/plan_e82_r5_guards.frozen.json` | `1e22b912e2135c72a555ab085c9702e378d1afdf3e3286ef933edac539163104` | `12562aee8f50b7cf9962ebdb04f3c8db7bc8e262a7df9f9a0a57116a85f1ef0a` |
| `studies/decision_layer/logs/plan_e82_refined_table_agent.md` | `logs/plan_e82_refined_table_agent.frozen.json` | `1dfea7622161cb8b5db1db93b9990180c3b6ba2f25a928808c8b7388a58a577a` | `2daf28439b4e9d5b881a78e367ef3961b4c1595e1058f438b0b00f6c6e178039` |
| `studies/decision_layer/logs/plan_e83_learning_dynamics.md` | `logs/plan_e83_learning_dynamics.frozen.json` | `5ddfe717b15a629db9b126939f2c981fab41cff2408888e69b04b5f2f4aa5d8e` | `1fcc260a006d368e486379c1b58058f07912d7307a79dd8f001403739f2e4a4a` |
| `studies/decision_layer/logs/plan_e83_note_2026-10-21.md` | `logs/plan_e83_note_2026-10-21.frozen.json` | `b78b0018ec22ffd3d4ca45541a0b8b25d826291fe0fadfb14af6f4d373919b5f` | `279c2f9557fac48b19d22d78c57fd1bb7c346efb9176465e9dbb77f23084596c` |
| `studies/decision_layer/logs/plan_e84_negatives_fair.md` | `logs/plan_e84_negatives_fair.frozen.json` | `50bf9d348a7188ba3290e742c7c6dd70c1d5e3056a15a6fc1e9fa4a264382d4f` | `33c875b7dd11edf28fcc321fbced3bbec1ae441468f8b2e61d255b1aa193daa5` |
| `studies/decision_layer/logs/plan_e84_note_2026-10-22.md` | `logs/plan_e84_note_2026-10-22.frozen.json` | `72ede09035bc6f59638344ef079967efef99c45e883e8633affddd12a8ec7b8a` | `a59875d53db3b6ead0f178bda952cbbfca8da2e6a2803d7905f6a29721c3c13d` |
| `studies/decision_layer/logs/plan_e85_s_under_random.md` | `logs/plan_e85_s_under_random.frozen.json` | `df0e05079e24cc29a0c71e201f2b7a7620fc77fe6377cee16182fae1c00ae1b1` | `46fad4c63d3afba4d619546564c3368600ceceffb8c9ef6149243d271bab357e` |
| `studies/decision_layer/logs/plan_e86_clause_drop.md` | `logs/plan_e86_clause_drop.frozen.json` | `1d4258bdc24e41dcd70b02328413cbebfbe94a25fe28a85492c983918505a300` | `25b795cf6ad91cb5d1aeac6526818f882f7c32ac4b72d54b268341c44e5bc489` |
| `studies/decision_layer/logs/plan_e87_receipts.md` | `logs/plan_e87_receipts.frozen.json` | `39313a383c97d293a25590b5e82a8d9320b0ac5e83ae0a7ac073b498f48d91fa` | `a9574bc1b654afcf36b6e477d4f1bee6e9a9f6c399adec86481953955a66d695` |

### Code

The code changed between freezes as the study went on, so one file can have different values in
different records. The table gives, for each file of `scripts/` and `src/` that was edited for
publication, its value just before publication and the records that hold that value. The edits are
comments and docstrings, two absolute path constants that are now derived from the location of the file
(`scripts/count_levels_diag.py`, `scripts/e80_selftest.py`), and the constant `SAMPLE_SHA` of
`scripts/e71_k35_sample.py`, which pins the sample file in the next table.

| file | SHA-256 before the publication edits | pinned in this form by | SHA-256 as published |
|---|---|---|---|
| `studies/decision_layer/scripts/abcd_episodic.py` | `5617f7e6733fac87b459c20c449f1bba8e6ad52024c7aceb29922f3fddb7189a` | 36 of the 36 records that pin the file (2026-10-17T21:09:33Z to 2026-10-23T01:55:38Z) | `3a032e225db866e056effae90e7589e9145a627d9f56b8892b0ddfef9b145f77` |
| `studies/decision_layer/scripts/c4_saturation.py` | `c7e0fc3528ed0609fbed666df45728b79614dc943d38b3fba8e630cebcddb8f8` | 16 of the 16 records that pin the file (2026-10-20T18:55:37Z to 2026-10-23T01:55:38Z) | `82e0aaea7216eb681876af31220b7acc066befdd81d867ae6300f10dd28d29b7` |
| `studies/decision_layer/scripts/count_levels_diag.py` | `17c16363eeda628ee249757e31dbfa2f33511dcd26bf77227d4298ca28a9c884` | 16 of the 16 records that pin the file (2026-10-20T18:55:37Z to 2026-10-23T01:55:38Z) | `32a8b6fed97928b236d562e0863fd8c786ccdbc3cd2f49bd71b2cfeaf4261b03` |
| `studies/decision_layer/scripts/count_table_coverage.py` | `598e381c6eea25fe398c00650d0ca0d3139aabce76e09f043ffd657ae6b3f641` | 29 of the 29 records that pin the file (2026-10-19T05:05:14Z to 2026-10-23T01:55:38Z) | `064e95b85238ffa6241e07bcc4c75ff020269d53104573baf5834fda0d0e5487` |
| `studies/decision_layer/scripts/demo_agent_c4.py` | `73d76a219b773ab61d7ec883f43e994bc3f1ed1348c5b3e0ec30c7891329c666` | 24 of the 24 records that pin the file (2026-10-19T11:46:21Z to 2026-10-23T01:55:38Z) | `cabdc4fa8afef749c214067c46b0ff2ca8f8a78a80486c21bb7e288ccfccb558` |
| `studies/decision_layer/scripts/e71_k35_sample.py` | `b31a5b3418d46c94767f183ff0b52443d2f73bb79809c03fac3ecde10cc56e55` | 26 of the 26 records that pin the file (2026-10-19T10:19:18Z to 2026-10-23T01:55:38Z) | `44877562ead243254ebf2d9a55872bf1909c69a0e7d3ba5777d4a0dfc47cdfad` |
| `studies/decision_layer/scripts/e77_breakdown.py` | `00aa17dc4a5f93369c92cde79a5378d36888b21742099fd311130f3bd5f5195f` | 17 of the 17 records that pin the file (2026-10-19T21:43:42Z to 2026-10-23T01:55:38Z) | `d0a552263267484f6da78b6f66d751bde752f929337d893d38dfa7617281c82d` |
| `studies/decision_layer/scripts/e78_check.py` | `1305d8945ab9cbbb8edacda39442058ee7c01485b5e55d38676ea1c2a87e73c0` | 16 of the 16 records that pin the file (2026-10-20T18:55:37Z to 2026-10-23T01:55:38Z) | `5ee1deb94b6358612a7aeed2dcf9f6fc1deaff0a3862a6e0c6c653ebdb0517b8` |
| `studies/decision_layer/scripts/e80_check.py` | `43f28b80db8bc6ef49aa7b091ab6d4e6a48c97c3d3d7beeac738daf07efb2a14` | 14 of the 14 records that pin the file (2026-10-20T20:53:40Z to 2026-10-23T01:55:38Z) | `ab95b4d4846cdbb59f75a5796e9f9885adb541dcf40c4b3527a591975efa096e` |
| `studies/decision_layer/scripts/e80_selftest.py` | `bd93a4fa9187d29a844106e10246c0e128584cdfd1a5a97c230918340ca14a3c` | 14 of the 14 records that pin the file (2026-10-20T20:53:40Z to 2026-10-23T01:55:38Z) | `3bebe172be7b7eadbef47c656ffd1b338d6dfe0997698197e9fc02d806a0db5f` |
| `studies/decision_layer/scripts/e82_gate_epochs.py` | `5c63849238b0c039f889490c77dbd3883d1fe5b2404fff5fad34e99f0736fe25` | 12 of the 12 records that pin the file (2026-10-21T04:32:36Z to 2026-10-23T01:55:38Z) | `5c6b01f442dedb989f92e4bafd0c6061afc31a0c9cb5dc94069e49519022eb06` |
| `studies/decision_layer/scripts/e87_readability.py` | `f156fcd10f798bac66a305f9b91773f8d48eb6df5c06cb21369ffcc61479b3a2` | 1 of the 1 records that pin the file (2026-10-23T01:55:38Z) | `53f3ce169d8124013b98f9e94776dee8464538860cba70002f4d49fa22d3750d` |
| `studies/decision_layer/scripts/lr_c_check.py` | `f84c494aa12c35f4d18819bb006b3fa47559d74077e68c3cf78d8e6596de3c48` | 34 of the 34 records that pin the file (2026-10-18T20:23:31Z to 2026-10-23T01:55:38Z) | `3160f37dfdddfe1b5043bada24d35acb764f5a8230a0a471e75e36cabc33f788` |
| `studies/decision_layer/scripts/lr_refit_posthoc.py` | `571223f162bbc6ac9840cda5a54830f435f87e0df4639067f3870d389a7e4c97` | 33 of the 33 records that pin the file (2026-10-18T23:23:30Z to 2026-10-23T01:55:38Z) | `1c096c946f44e2c700297a90f3d7e2ee24697e71c941fafb76a35fb1b81b20c0` |
| `studies/decision_layer/scripts/master_queue.sh` | `166cf53faa8b1e34123aeab45440cb95b3b4a715170987e0f290e205fc754f35` | 36 of the 63 records that pin the file (2026-10-17T21:09:33Z to 2026-10-23T01:55:38Z) | `9b38e4fd47c7898ce716a907f6c73f277202c66e7d98d708d9d5c413fbbafad9` |
| `studies/decision_layer/scripts/multiwoz_prepare.py` | `e1c7299ff99fcde4832834185ca6207c0c0905b4092bb151cdc3d66dd11f9398` | 17 of the 17 records that pin the file (2026-10-19T21:43:42Z to 2026-10-23T01:55:38Z) | `716c1ab6951a21b7ba3ab55ae0b6c3624fddfccdc44bbda6c5bc19b6c2ab1917` |
| `studies/decision_layer/scripts/newtable_run.py` | `f98ce98e3b8477d844e3549985cd9827a6b91d612a3b20095663eda7c7e32819` | 13 of the 13 records that pin the file (2026-10-21T01:31:09Z to 2026-10-23T01:55:38Z) | `f8c820ade4ffdbdb958b8cc5db800fed1f8dce829c131920fb2c1c0175db5b18` |
| `studies/decision_layer/scripts/p9_size.py` | `43e12a3136985d636e6782f6d68c6d10394a3d6e1756443600a145c26d5a434b` | 16 of the 16 records that pin the file (2026-10-20T18:55:37Z to 2026-10-23T01:55:38Z) | `96097617ec9644adf3d3876ac4776881546281b617511cb651f4f6ffbd79ed30` |
| `studies/decision_layer/scripts/r25_check.py` | `059b43756955b1f1c97a9bc46fa513c879ee82ecf1dd7e8c75725b3c19739574` | 12 of the 12 records that pin the file (2026-10-21T04:32:36Z to 2026-10-23T01:55:38Z) | `4cbd7caf169488c4a8e2b6d1affb1e66fc2ec664f317e3cb0660cce7b3056070` |
| `studies/decision_layer/scripts/report_twolevel.py` | `34e7c051414dc14afa7a1747493062fb8d98f811f00dbdc889ac3560379bc027` | 12 of the 12 records that pin the file (2026-10-21T04:32:36Z to 2026-10-23T01:55:38Z) | `e4f1202364d4087f29f73b6f2a4900e31c3ca6a1df4acfb5232b597ed8b5d94a` |
| `studies/decision_layer/scripts/star_explore.py` | `b403db2f25d2dd56b4af91461f18b43b67bf5771d6855e9048a617ba3870ff93` | 36 of the 36 records that pin the file (2026-10-17T21:09:33Z to 2026-10-23T01:55:38Z) | `d748ac85a06bfbe39c170f8b5b5fedefe39af628777224a102b168a9392a8874` |
| `studies/decision_layer/scripts/star_prepare.py` | `359845561307fdb91d81a35f75e4d7f53e9f25371af723cce7cf9871b251d4b7` | 37 of the 37 records that pin the file (2026-10-17T14:26:43Z to 2026-10-23T01:55:38Z) | `a4b4f3c0c98455a4600f5c7e83efa5d3042de36c7cb794b6e7916965591255b9` |
| `studies/decision_layer/src/dl/abcd.py` | `a66c0c633ca227d24ec1f4b9ffcc97cf3b149c6fd6fc6e60919b9f9ed149cf63` | 34 of the 49 records that pin the file (2026-10-18T20:23:31Z to 2026-10-23T01:55:38Z) | `d8cf795ac26b1f0589ebfa4177f7196a6110e3ddcf9d43ad1c111bf6b60b96dc` |
| `studies/decision_layer/src/dl/count_bits.py` | `c9434a1b141d7a3bfca491606b8c024c900c449972f147f895fc8f21e2f42324` | 16 of the 16 records that pin the file (2026-10-20T18:55:37Z to 2026-10-23T01:55:38Z) | `15e999330d046c4d6c5e2f6cd71b1f3a9e6f3ad28efe8638c9d7c1792b540f45` |
| `studies/decision_layer/src/dl/count_text.py` | `5f201106593c065f03a8f1bb843ae62ba08940422f682e0a76cdf064e9882d7e` | 29 of the 29 records that pin the file (2026-10-19T05:05:14Z to 2026-10-23T01:55:38Z) | `c19275bd282859bb0a6845964bab28af84b043af7d0f5922e9de05e38fc686c7` |
| `studies/decision_layer/src/dl/static_text.py` | `ec89f28c7f96c4b21d0704588e5730fc909beaaf4391ce57c736cc4266c66c39` | 36 of the 36 records that pin the file (2026-10-17T21:09:33Z to 2026-10-23T01:55:38Z) | `e7037dc6ac570487cc23dd84125166b10fb1677fa48edd88d44dab0d675b5a2a` |

### Other recorded values

Dev plans, dated notes and the results page also record the hashes of some files that have no freeze
record of their own: driver scripts, a sample file and a draft registration. A value given as a prefix
is the first eight characters of the SHA-256, as the text records it.

| file | where the value is recorded | SHA-256 as recorded | SHA-256 as published | note |
|---|---|---|---|---|
| `studies/decision_layer/logs/e81_driver.sh` | `logs/plan_e81_note_2026-10-21.md` | `73e44676d4dba0eb135905d00e858eff31beb6b683a2780ca8dedf5156d97ac7` | `265c3b64740b8cc2ab72a7a0666d50d35f2b36f49d2c236aa8696f316965caa0` | a comment reworded; paths derived from the location of the file |
| `studies/decision_layer/logs/e81_chain.sh` | `docs/results.md` (prefix) | `039c01c2` | `b341b084eab5193e36f14c4eb6613f27e9e5e34af760b438dd74efc0e649b690` | comments reworded; paths derived from the location of the file |
| `studies/decision_layer/logs/e82_driver.sh` | `logs/plan_e82_note2_2026-10-21.md` | `d0d80e4ca8d8c6ae3e409d1662d0ecd3a8dbec308682607c5b99016fad1dd2bd` | `b1f9013ebae4609858a12beb5eead56123e4b38f1e2fbbc2c135c9ddb216b64f` | paths derived from the location of the file; one log string reworded |
| `studies/decision_layer/logs/e83_driver.sh` | `logs/plan_e83_learning_dynamics.md` | `f75ec266da30f90d751ecb4cde4fb27d9b7f636ee0c4e3848a3f02aeccfe320b` | `86987afd40c507b73152a65e6c7a5e351318cca4d7e9cf66e23fa1efcff3af36` | comments reworded; paths derived from the location of the file |
| `studies/decision_layer/logs/R24_chain.sh` | `registrations/R24_multiwoz_random_negatives.md` | `0b5925549ab45cbdbcc8d5e6882a777419ee4e3f7ae36779f29907016e93c159` | `913c0d1ce5b87fa114cdb51ba66bb4f714e77fd450c654c4b597164f1793962a` | paths derived from the location of the file |
| `studies/decision_layer/logs/e71_k35_sample.json` | `logs/plan_e71_v2_note2_2026-10-19.md`; `SAMPLE_SHA` in `scripts/e71_k35_sample.py` held it before publication | `5a42edcf25522aa28a501c4b4d25d72d4725411174a1a73c326a011000da39ff` | `95d3dbbfa08fde2cd2bb4bfedb2017ce1eaa5ca3dca23d6a823a5945306557c6` | the narrative value of the key `what` reworded, every other value unchanged; `SAMPLE_SHA` now holds the published value |
| `studies/decision_layer/src/dl/count_bits.py` | `logs/plan_e78_2d_counts.md` (prefix) and the logs of the e78 runs, which are not distributed | `c9434a1b` | `15e999330d046c4d6c5e2f6cd71b1f3a9e6f3ad28efe8638c9d7c1792b540f45` | comments and docstrings reworded; also in the code table above |
| `studies/decision_layer/registrations/R24_multiwoz_superiority.md` | `registrations/R24_multiwoz_random_negatives.md` and `docs/results.md` (prefix) | `b19d8c4a` | `c36e99f64e22c4f0caa0d45a49719d93388a41388dc6416d4cdbfefc31c21c6f` | the draft that R24 superseded, never frozen; reworded for publication |

## tally

The freeze script of the study writes a record next to each frozen registration: one for the
registration T0 and one for each of its two amendments. Each holds the UTC time of the freeze and the
SHA-256 of the registration and of every file under `src/`; the record of T0 also holds those of the
three test files. Before publication each frozen text equalled the value in its record. T0 was edited for
publication on 2026-10-23 in wording only, and its copy carries a publication note; the two amendments
are published as frozen, byte for byte. Four files under `src/` were edited in comments
and docstrings only. Six further files under `src/` (`src/tally/t0/run.py`, `src/tally/b1/flat.py`,
`src/tally/b1/report.py`, `src/tally/b1/team.py`, `src/tally/b6/run.py` and `src/tally/floor/kn.py`)
changed after the freeze of T0 and before publication; they differ from the record of T0 and equal the
records of the two amendments, as the study README says. The run records, which are not distributed, also hold the SHA-256 of every file under
`src/` at the start of each run.

| file | freeze record | SHA-256 as registered | SHA-256 as published |
|---|---|---|---|
| `studies/tally/registrations/T0.md` | `registrations/T0.frozen.json` | `78d408591d89c24fbc3a92f0b20784ea3f355b882f82be4b3ec14cd6ac7dc409` | `ee3fccda83ba627ad7f531b42ba663d7b001cbe23b11602acb86c32b459dc254` |

| file | SHA-256 before the publication edits | pinned in this form by | SHA-256 as published |
|---|---|---|---|
| `studies/tally/src/tally/b6/config.py` | `e4e4f703bc172cfa2e3b742fe639b55bf7f35f083604fefd2d4ede267a6f6e6a` | `T0.frozen.json`, `T0_amendment_coalesced_settings.frozen.json`, `T0_amendment_flat_settings.frozen.json` | `61acf051ce9a1935a612c2c6a2a0d6b61c6b2aa40a3f459473c8575804a90bc2` |
| `studies/tally/src/tally/floor/codes.py` | `4b8ea46141b1d2730de7a007b9d478fbaca1ca6035afa065a5cfbc0e9cd0e30b` | `T0_amendment_coalesced_settings.frozen.json`, `T0_amendment_flat_settings.frozen.json` | `47cae5999e93e7ab587d630f2506b765a75db4fa575bd4430c1db75d4efea01f` |
| `studies/tally/src/tally/floor/kenlm.py` | `871c8d309f6eb024e58365ede67b74cf321983c1b3d42bcc08013679b4177ae3` | `T0.frozen.json`, `T0_amendment_coalesced_settings.frozen.json`, `T0_amendment_flat_settings.frozen.json` | `25f7585608ae1b787a6b0cc6a8f60b29db5df93a3927a1d0fc71ae4132b7395b` |
| `studies/tally/src/tally/t0/data.py` | `25c66da020ecfd1d416e40f382b6609b22807a96e0f3c08c05024a14e2c08475` | `T0.frozen.json`, `T0_amendment_coalesced_settings.frozen.json`, `T0_amendment_flat_settings.frozen.json` | `4650b3caf3b9c7ed11821ba4e79287d8156d7c0eb577f7acb0722f66ed9543b9` |

## Pins on files that are not in the repository

These values are published as registered. Their targets are run folders, datasets or local copies of
other repositories, none of which is distributed, so the values were not recomputed.

| what is pinned | where the value is recorded | SHA-256 as registered | note |
|---|---|---|---|
| `scripts/train_controller.py` of Clause-Driven-LLM, second recorded state | run record of `20261002T210555Z_m5_setup_torchcpu` | `e999bfb14af47181fd910927c74ed09f5c6b43cff880a6f38300b9b38a0068b2` | `studies/clause_lm/src/clause_driven_llm/train_controller.patch`, applied to the upstream file, gives this state |
| `scripts/train_controller.py` of Clause-Driven-LLM, first recorded state | run records of `20261002T195152Z_m5_setup_sorted`, `20261002T195255Z_m5_gate0_s8_seed42_sorted`, `20261002T205115Z_m5_setup_first_appearance` and `20261002T205134Z_m5_gate0_s8_seed42_first_appearance` | `d6214e313601fcede90e7b04c1dee444412c25141d85272620e03968b6d77e80` | `studies/clause_lm/src/clause_driven_llm/train_controller.state1.patch`, applied to the upstream file, gives this state |
| 17 data files of hierarchical_tm | `studies/hierarchical_tm/frozen_settings_hashes.json`, block of data pins | in that file | datasets, not distributed |
| core and cache files of the count-vote runs | `core_file_sha256` and `cache_meta_sha256` in `studies/tm_language_model/experiments/cvtm/configs/cvtm_a3_H8_*.json` | in those files | files inside run folders |
| dataset files of the pre-registration | `studies/tm_language_model/docs/notes/preregistration-data-hashes*.json` | in those files | datasets, not distributed |
| tables, example files, caches and data files read by the registered runs of decision_layer | the registrations, dev plans and dated notes in `studies/decision_layer/registrations/` and `studies/decision_layer/logs/` | in those files | not distributed |
| data files of each run of tally | `data_hashes.json` in each run folder of tally | in those files | run folders, not distributed |
