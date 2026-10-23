#!/usr/bin/env python
"""e83's checker without the L8 arm (dated note logs/plan_e83_note_2026-10-21.md; added 2026-10-21).

    e83_pick_noL8.py --stage screen|mid|final

The L8 arm (exact literal budget 8) cannot run on MultiWOZ at 4,000 clauses: tm.py raises NotImplementedError for an
exact budget in its chunked update, which every MultiWOZ view-B run at 4k uses (4,000 x 14,498 literals is above
CHUNK_ELEMS). The note drops L8 for that mechanical reason. This file runs the frozen scripts/e83_pick.py unchanged,
with "b8" removed from its ARMS: the screen, the pick rule, P1, C1, C2 and C3 are the plan's, on REF, nrand, nmix50
and drop50. Tested by tests/test_e83_pick_noL8.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e83_pick as P  # noqa: E402

del P.ARMS["b8"]

if __name__ == "__main__":
    P.main()
