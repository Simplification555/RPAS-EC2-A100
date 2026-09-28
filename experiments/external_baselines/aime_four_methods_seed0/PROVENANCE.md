# Provenance and scope

This folder was assembled from the recovered local experiment wrappers and
upstream repositories identified in `README.md`. It contains baseline adapters
only. There is no RPAS search algorithm, RPAS result, secret, model checkpoint,
or AIME test content in this folder.

The AFlow/MaAS runner maps frozen AIME rows into each upstream method's MATH
input schema and adds local execution-safety, shared scoring, and blind-split
audit code. The ADAS/G-Designer runner similarly adapts task rows, model
transport, and evaluation. These are compatibility adapters, so authors must
describe the actual adapter settings in any paper and must not claim unmodified
upstream benchmark results.

The missing MaAS helper files recovered for the adjacent MASBench package are
project-local compatibility files, not files from the MaAS upstream commit.
Their byte hashes and limitations are recorded in
`../masbench_aflow_maas_seed0/experiments/MAAS_HELPER_PROVENANCE.md`.

The frozen AIME data files are included with the user's explicit publication
authorization. They were copied from the private source repository
`JiangyueAnn/RPAS` at revision
`e12f58823be5f91a32f05f9af4d36e54838ffe59`; `data/frozen_aime_manifest.json`
records each source/split path, SHA-256, and row count. The runner pins the
manifest hash and verifies the canonical validation and test files before
using them. Test data is not opened until the durable D_select candidate lock
exists.

The upstream repositories are fetched rather than vendored. License inspection
of the pinned snapshots found an MIT license in AFlow and Apache-2.0 in ADAS;
no `LICENSE` file was found in the checked MaAS or G-Designer snapshots. This
is a file-presence observation, not a legal conclusion. The two MaAS helper
modules are project-local files whose original Git commit/license metadata was
not available; see the adjacent helper provenance note and confirm rights
before further redistribution.
