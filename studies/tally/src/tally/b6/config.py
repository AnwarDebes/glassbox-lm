"""B6 settings: members, budget, epochs, folds, arms.

Members follow the published recipes, scaled to the budget of 100 clauses per class per
member. T is scaled with the number of clauses at the ratio of the recipe (Granmo 2023 Table 1
uses the same T/clauses ratio on all three datasets). s, weighting and the literal budget are
copied unchanged.

Fashion-MNIST: Granmo 2023 (arXiv 2309.04801) Table 1 and the CAIR repository scripts.
AG News: the CAIR repository's IMDb team script (word n-grams and char_wb n-grams, 10,000
weighted clauses, T 8000, s 2.0, clause_drop_p 0.75; max_included_literals is printed in the
file names but not passed to the machine, so no literal budget). Three of its four views.
"""
CLAUSES = 100          # per class per member (design decision)
FOLDS = 3              # K of the out-of-fold shadow teams (rule T, sequential boosting)
REFRESH = 1            # team error refreshed every REFRESH epochs
SEEDS = tuple(range(1, 11))

MEMBERS = {
    "fmnist": [
        dict(name="thresh10", view="thresh", patch=(10, 10), weighted=True, t_ratio=0.25, s=10.0,
             max_lit=32, drop=0.0),
        dict(name="thermo3", view="thermo", patch=(3, 3), weighted=True, t_ratio=0.75, s=2.5,
             max_lit=32, drop=0.0),
        dict(name="thermo4", view="thermo", patch=(4, 4), weighted=True, t_ratio=0.75, s=2.5,
             max_lit=32, drop=0.0),
        dict(name="hog", view="hog", patch=None, weighted=False, t_ratio=0.025, s=10.0,
             max_lit=32, drop=0.0),
    ],
    "agnews": [
        dict(name="word1", view="word1", patch=None, weighted=True, t_ratio=0.8, s=2.0,
             max_lit=None, drop=0.75),
        dict(name="word2", view="word2", patch=None, weighted=True, t_ratio=0.8, s=2.0,
             max_lit=None, drop=0.75),
        dict(name="char4", view="char4", patch=None, weighted=True, t_ratio=0.8, s=2.0,
             max_lit=None, drop=0.75),
    ],
}
CLASSES = {"fmnist": 10, "agnews": 4}

# Epochs per member training, the same for every arm. Set before the registration is frozen;
# the published recipes use 100 (Fashion-MNIST) and 40 (IMDb).
EPOCHS = {"fmnist": 30, "agnews": 20}

# One machine with the same total clauses: Fashion-MNIST members do not share an input
# shape (three convolutions over different images and a flat HOG vector), so the single
# machine takes the configuration of one member with M times the clauses; which member is
# fixed by the tuning run (seed 1, validation). AG News: all three views concatenated.
SINGLE = {"fmnist": dict(base="thresh10"), "agnews": dict(base="concat")}

# Clause-fusion head: a TM over the concatenated clause outputs of the frozen blind members.
HEAD = dict(clauses=100, T=100, s=10.0, weighted=True, max_lit=None, drop=0.0)

# Neural gate: the architecture of tmu.composite.gating.neural_gate (Dense 128 relu, Dense 64
# relu, softmax over members, Adam), trained on the calib split.
GATE = dict(hidden=(128, 64), max_iter=200)

ARMS = ("member", "single", "ruleT", "nocomm", "coupled", "seqboost", "fusion", "gate")


# Multiplier on the scaled T per member, fixed by the tuning run before the freeze (1.0 = the
# recipe ratio). The same value is used by every arm.
T_SCALE = {}


def T_of(member, clauses=CLAUSES):
    return member["t_ratio"] * clauses * T_SCALE.get(member["name"], 1.0)


def member_seed(seed, index, copy=0):
    """Paired initialisation: member i of seed s starts from the same state in every arm.
    copy > 0 are the out-of-fold shadow copies."""
    return 1_000_000 * copy + 1000 * seed + index
