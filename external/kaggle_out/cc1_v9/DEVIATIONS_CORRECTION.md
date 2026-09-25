# Correction to `run_record.json` (review 022 C2, RESULTS 87)

`deviations` lists "activation checkpointing of Encoder/crossEncoder". That entry was carried over from the v7 kernel's
list and is **wrong for the trained model**. The production trainer (`run_train_dp.py`) applied only the DataParallel
patch, the freeze of the ten unused parameters and the resume hooks. Evidence: `../cc1_v8/train_session1.log` and
`train_session2.log` each contain `LINCS DataParallel devices: [0, 1]` and `LINCS froze 10 unused parameters` and zero
checkpointing lines; session 1's epochs are full length from epoch 0 (437 s). Checkpointing ran only inside GUARD F's
one-batch memory probe. The run record itself is left byte-for-byte as the kernel wrote it.
