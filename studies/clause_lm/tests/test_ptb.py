"""src/ptb.py: the loader refuses the test split without opening it, and reads train and valid as documented.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv-cdllm/bin/python tests/test_ptb.py

Run from studies/clause_lm. CPU only, under a second. Part 1 uses the real data folder of tm_language_model but only
asks for the test split, which the loader must refuse before it opens anything (a trip wire records every opened path
that contains "/test/"; ptb.tokens is replaced by a recorder as well). Part 2 points the loader at toy text files in a
temporary folder.

Asserts:
- load() with "test" among the splits (alone, after train, after valid) exits with the message that names
  final_test.py, and no file under a test folder is opened and no tokens are read;
- path("test") only builds the path;
- on toy files: one <eos> after every sentence, the vocabulary is the sorted set of training tokens, ids are int32
  and decode back to the tokens, and a validation word missing from training stops the load.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toyutil  # noqa: E402

check = toyutil.Checks()


def main():
    with tempfile.TemporaryDirectory(prefix="clm_ptb_") as tmp:
        ptb, _ = toyutil.use_src(Path(tmp))
        real_tokens = ptb.tokens
        calls = []
        ptb.tokens = lambda split: calls.append(split) or real_tokens(split)
        with toyutil.Tripwire("/test/") as tw:
            for splits in (("test",), ("train", "test"), ("valid", "test"), ("train", "valid", "test")):
                try:
                    ptb.load(splits)
                    refused, msg = False, ""
                except SystemExit as e:
                    refused, msg = True, str(e)
                check(refused and "final_test.py" in msg, f"load{splits} is refused ({msg!r})")
            p = ptb.path("test")
        ptb.tokens = real_tokens
        check(tw.hits == [] and calls == [], f"nothing under a test folder was opened, no tokens read ({tw.hits}, {calls})")
        check(p.endswith("/test/text.utf8"), f"path('test') only builds the path ({p})")

        train = [["a", "b", "c"], ["b", "c"], ["c", "a", "d"]]
        valid = [["c", "b"], ["a"]]
        ptb.PTB = str(toyutil.write_ptb(Path(tmp), train, valid))
        vocab, w2i, ids = ptb.load(("train", "valid"))
        check(vocab == ["<eos>", "a", "b", "c", "d"], f"vocabulary is the sorted set of training tokens ({vocab})")
        check(all(v.dtype == np.int32 for v in ids.values()), "ids are int32")
        dec = {s: [vocab[i] for i in ids[s]] for s in ids}
        check(dec["train"] == ["a", "b", "c", "<eos>", "b", "c", "<eos>", "c", "a", "d", "<eos>"],
              "train decodes to its tokens with one <eos> per sentence")
        check(dec["valid"] == ["c", "b", "<eos>", "a", "<eos>"], "valid decodes to its tokens with one <eos> per sentence")
        toyutil.write_ptb(Path(tmp), train, [["a", "zzz"]])
        try:
            ptb.load(("train", "valid"))
            stopped = False
        except AssertionError as e:
            stopped = "out-of-vocabulary" in str(e)
        check(stopped, "a validation word missing from training stops the load")
    check.done()


if __name__ == "__main__":
    main()
