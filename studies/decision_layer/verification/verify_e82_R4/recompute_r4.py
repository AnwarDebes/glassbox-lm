# independent recompute of e82's R4 rule (verify-result step 1-2): reads a snapshot of abcd.jsonl, standard library only
import json, sys
snap = sys.argv[1]
rows = {}
for i, line in enumerate(open(snap), 1):
    d = json.loads(line)
    if d.get("arm") == "tm":
        rows.setdefault((d.get("tag"), d.get("seed")), []).append((i, d))
pairs = {"star": ([("e74c_star_Bc4_c16000_s16", 1), ("e78_star_B_c16000_s16", 2)], "e82_star_Bt_c16000_s16"),
         "abcd": ([("e74c_abcd_Bc4_c4000_s16", 1), ("e78_abcd_B_c4000_s16", 2)], "e82_abcd_Bt_c4000_s16"),
         "multiwoz": ([("e77_mwoz_Bc4_c16000_s8", 1), ("e79_mwoz_B_c16000_s8", 2)], "e82_multiwoz_Bt_c16000_s8")}
ok_all = True
for ds, (refs, cand) in pairs.items():
    R, C = [], []
    for t, sd in refs:
        got = rows.get((t, sd), [])
        assert len(got) == 1, (t, sd, len(got))
        i, d = got[0]; R.append(d["acc"])
        print(f"  ref  {t} seed {sd}: acc {d['acc']!r} (line {i}; s {d['cfg']['s']}, clauses {d['cfg']['n_clauses']}, T {d['cfg']['T']}, epochs {d['cfg']['epochs']}, table {d['static_info']['xcust'].get('table')})")
    for sd in (1, 2):
        got = rows.get((cand, sd), [])
        assert len(got) == 1, (cand, sd, len(got))
        i, d = got[0]; C.append(d["acc"])
        print(f"  cand {cand} seed {sd}: acc {d['acc']!r} (line {i}; s {d['cfg']['s']}, clauses {d['cfg']['n_clauses']}, T {d['cfg']['T']}, epochs {d['cfg']['epochs']}, table {d['static_info']['xcust'].get('table')} {d['static_info']['xcust'].get('table_file_sha256','')[:16]}, time {d['time']})")
    diff = 100 * (sum(C) / 2 - sum(R) / 2)
    ok = diff >= -0.5
    ok_all &= ok
    print(f"{ds}: candidate mean {sum(C)/2:.6f}, reference mean {sum(R)/2:.6f}, difference {diff:+.4f} points -> {'holds' if ok else 'FAILS'} (R4: >= -0.5)")
print("R4:", "TAKEN" if ok_all else "NOT taken")
