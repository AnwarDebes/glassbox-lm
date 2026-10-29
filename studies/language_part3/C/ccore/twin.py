"""The count twin of C-core (PLAN_part3.md section 4, "Count twin", review 7): 8 trials, each a
bucketed-EM mixture (on calib) of counted class chains over the same codes the TM reads.

Trials (all scored on valid by the same rule as every arm: valid bpw of the system {F, f, s}):
  T-E order 3, 4, 5 alone; T-E order 4 and 5 + T-B + T-EB; T-E order 5 + T-EB;
  T-ALL (T-E + T-B + T-EB + T-B4 + T-EB2) at T-E order 4 and 5.
With T-ALL at T-E order 5, every code bit the TM sees is in at least one counted chain at full depth
(tests: every_code_bit_is_in_a_counted_chain)."""
import numpy as np

from . import chains, systems

TRIALS = [
    ("TE3", [("TE", 3)]),
    ("TE4", [("TE", 4)]),
    ("TE5", [("TE", 5)]),
    ("TE4+TB+TEB", [("TE", 4), "TB", "TEB"]),
    ("TE5+TB+TEB", [("TE", 5), "TB", "TEB"]),
    ("TE5+TEB", [("TE", 5), "TEB"]),
    ("TALL4", [("TE", 4), "TB", "TEB", "TB4", "TEB2"]),
    ("TALL5", [("TE", 5), "TB", "TEB", "TB4", "TEB2"]),
]


def mname(m):
    return f"TE{m[1]}" if isinstance(m, tuple) else m


def all_members(trials=TRIALS):
    out = []
    for _, ms in trials:
        for m in ms:
            if m not in out:
                out.append(m)
    return out


def build(members, y_train, comp_train, mode="kn"):
    """mode "kn": T-E through tally KN on the train class stream y_train (its targets are its context);
    mode "pair": T-E through PairKN on the true class contexts of the train stream, with the targets
    y_train given separately (used for permuted targets); the contexts then come from comp_train["cls"]."""
    models = {}
    for m in members:
        if isinstance(m, tuple):
            order = m[1]
            if mode == "kn":
                models[mname(m)] = chains.TEModel(y_train, order)
            else:
                ctx = chains.class_contexts(comp_train["cls"], chains.K - 1, order - 1)
                models[mname(m)] = chains.PairKN(ctx, y_train, order)
        else:
            models[m] = chains.ADChain(m, chains.chain_keys(comp_train, chains.SPECS[m]), y_train)
    return models


def dists(models, comp_eval):
    """Class distributions (n, 65) of every member on one split; comp_eval["cls"] is its class stream."""
    out = {}
    for name, md in models.items():
        if isinstance(md, chains.TEModel):
            out[name] = md.dist(comp_eval["cls"])
        elif isinstance(md, chains.PairKN):
            out[name] = md.dist(chains.class_contexts(comp_eval["cls"], chains.K - 1, md.N - 1))
        else:
            out[name] = md.dist(chains.chain_keys(comp_eval, chains.SPECS[name]))
    return out


def trial_class_model(members, D_cal, D_val, y_cal, b_cal, b_val):
    names = [mname(m) for m in members]
    if len(names) == 1:
        return D_cal[names[0]], D_val[names[0]], {"members": names, "weights": None}
    lam_g, lamb, em = systems.fit_class_mixture([D_cal[n] for n in names], y_cal, b_cal)
    PX_cal = systems.mix_classes([D_cal[n] for n in names], lamb, b_cal)
    PX_val = systems.mix_classes([D_val[n] for n in names], lamb, b_val)
    return PX_cal, PX_val, {"members": names, "lam_global": np.round(lam_g, 5).tolist(),
                            "mean_weight_valid": np.round(lamb[b_val].mean(0), 5).tolist(), **em}
