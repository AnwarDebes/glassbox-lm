# Plans and registrations of or_group_tm, in the order in which they were fixed

Each plan was fixed (its text and the code hashed, with the time) before the first run it governs. The plans E0 to E4 and E6 were fixed in a working folder outside this repository and are published as copies with a publication note; the registered confirmations that read held-back data were fixed in this folder, and their records (`*.frozen.json`) are next to them.

| file | fixed (UTC) | SHA-256 of the fixed text |
|---|---|---|
| [E0_planted_feasibility.md](E0_planted_feasibility.md) | 2026-11-12T15:53:26Z | `d5f19fb9d86e0bbcd6f806e2213f0f830eac7cbd0cad9d4b003a016218463dfa` |
| [E0b_one_seed_per_example.md](E0b_one_seed_per_example.md) | 2026-11-12T15:56:18Z | `39ed24ad3b000364aaaa22a4eeb433f7c57782c07b1fe83848e9a582f440a4ad` |
| [E1_planted_confirmation.md](E1_planted_confirmation.md) | 2026-11-12T15:58:27Z | `ed616c4dc66cae4458b7a225aec20f4354572af60e2158e30ba4972ae317e5f7` |
| [E2_slide10_design.md](E2_slide10_design.md) | 2026-11-12T16:04:09Z | `64d265c87bf4e39e9463463f9b9c54bcc500fa3dfa6b237d89fe460db0c17167` |
| [E2_note_decay.md](E2_note_decay.md) | 2026-11-12T16:06:22Z | `50fdd5746ca1402e7710528ff243d178c5ab22d6135eb6b45d1cc7f0d5d055b6` |
| [E4_wikipedia_design.md](E4_wikipedia_design.md) | 2026-11-12T16:20:17Z | `89f51bae1fc39c077a792dbd1e1ff912c3c53866b875d90da0bd70d8fa3b98f8` |
| [E6_sequence_demo.md](E6_sequence_demo.md) | 2026-11-12T17:03:46Z | `96d25e1710f7c6798013f981d345df1cd681cb56e64d612d53bdea5ddb03584e` |
| [E3_slide10_text.md](E3_slide10_text.md) | 2026-11-12T17:48:56Z | `f97da6b4b9fd30824e6e1ff6fc168362f99983d05f562a941dad4194b8176c09` |
| [E5_wikipedia_test.md](E5_wikipedia_test.md) | 2026-11-12T21:05:33Z | `3050ac2f6efb7be1c89dd0cfd58e915dcdfdb3ffecb8be95d0ece85ba40869b9` |

E3 was fixed in this folder by `src/freeze.py`, which `E3_job.sh registered` runs; its record, [E3_slide10_text.frozen.json](E3_slide10_text.frozen.json), lies next to it and holds the time, the SHA-256 of the plan and of every file it pins. The text in this folder is the fixed text itself. E6 is exploratory and was fixed in the working folder; its record stays there. E5 was fixed the same way by `E5_job.sh registered`; its record is [E5_wikipedia_test.frozen.json](E5_wikipedia_test.frozen.json). The throwaway plans of the dry runs, `dry/E3_dry.md` and `dry/E5_dry.md`, and their records are not registrations and are not listed.
