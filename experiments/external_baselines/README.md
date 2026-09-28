# Reproducibility bundles: external methods only

This directory contains two isolated packages for the external-baseline work:

1. [`aime_four_methods_seed0/`](aime_four_methods_seed0/): AIME runners and
   validation/selection/test audit code for AFlow, MaAS, ADAS, and G-Designer.
   Each shell invocation runs its named pair serially on one visible GPU.
2. [`masbench_aflow_maas_seed0/`](masbench_aflow_maas_seed0/): the complete
   frozen 24/24/60 five-axis MASBench seed-0 AFlow+MaAS package, with the two
   missing MaAS helper files recovered and documented.

Neither package includes or runs the RPAS method. The MASBench task instruction
is extracted into a small task-contract module; the RPAS optimizer/search source
file that previously hosted the instruction is intentionally excluded. Generic
Q/E selection helpers and task/evaluation adapters are not the RPAS method.

The AIME package intentionally excludes raw questions/answers, model weights,
embeddings, credentials, predictions, and result outputs. MASBench's five-axis
dataset is included from its frozen, licensed upstream snapshot. See each
package README for upstream pins, adapter disclosures, setup commands, protocol
values, and hardware validation limits.

Both packages are single-seed pilots (`seed=0`). They do not provide a
three-seed mean±standard-deviation result. Tests are local CPU/offline tests;
neither package has been run end-to-end on the target RTX PRO 6000.
