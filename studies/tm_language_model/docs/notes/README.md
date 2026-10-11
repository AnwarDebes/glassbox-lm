# Data hash records

The fifteen files `preregistration-data-hashes*.json` in this folder are the data hash records of the
pre-registration. Each holds the reference hashes of the dataset files of one step: the frozen body of
[docs/13](../13-preregistration.md) (section 4), amendment A4, the exploratory early tests of 2026-09-24
(the seven files with A6 in their name; [docs/12](../12-design-changes.md) F23), amendment A7, and the
confirmatory test files of amendment A9 item 10, of which the TREC record holds a git blob SHA-1 under
amendment A10. The registration, or the record itself, declares each of them write-once: a record is
written one time and not edited afterwards. Code opens these files by path
(`src/tmlm/ssl/downstream.py`, `src/tmlm/ssl/confirm.py` and several experiment scripts) and reads the
hash values and the source fields (repository, revision, file size); no code reads the two reworded
values.

The wording of four of these records was edited for publication on 2026-10-10. In
`preregistration-data-hashes-A6-men.json`, `preregistration-data-hashes-A6-simlex999.json`,
`preregistration-data-hashes-A6-wordnet.json` and `preregistration-data-hashes-A6-ws353.json` the text of the
value of the key `written` was reworded, and in `preregistration-data-hashes-A6-simlex999.json` one rule
citation inside the value of the key `splits`. No key and no hash value was changed in any of the fifteen
files, and the other eleven files are byte identical to the records as they were written. The registered
datasets, hashes, split definitions and dates are unchanged.

The write times inside the records place three of them (`preregistration-data-hashes-A6-agnews.json`,
`-A6-sst2.json` and `-A6-trec.json`) in the run `20260924T151202Z_ssl_transfer_data_exploratory`, and the two
test references (`-A7-agnews-test.json` and `-A7-trec-test.json`) among four runs named
`ssl_r1cf_testhash_confirmatory` of 2026-09-26. All five of these runs have exit status 1 in the
[run index](../../../../docs/run-index.md); a fifth test-hash run of the same evening has exit status 0.

The datasets themselves are not distributed with the repository. The hashes pin files that a reader has to
obtain separately; [docs/data.md](../../../../docs/data.md) at the repository root says where each dataset
comes from.
