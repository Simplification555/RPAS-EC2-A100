# MASBench split provenance

These five frozen-axis splits were taken from the `data/masbench/` directory of
[`JiangyueAnn/RPAS`](https://github.com/JiangyueAnn/RPAS) at commit
`e12f58823be5f91a32f05f9af4d36e54838ffe59`. Each axis contains the canonical
24-row search split, 24-row selection split, 60-row test split, and its source
manifest. The source is Salesforce's `MASBench` dataset; see
[`README.upstream.md`](README.upstream.md) and the
[Salesforce/MASBench dataset card](https://huggingface.co/datasets/Salesforce/MASBench)
for attribution and license metadata (Apache-2.0).
The upstream data license text is included here as `LICENSE-2.0.txt` and
applies to the upstream dataset, not automatically to this package's custom
code.

The root `manifest_search_select.json` is a deliberately D_test-blind run-time
manifest. It includes only search/selection row IDs and raw file hashes. It was
generated from those two split files; it contains no test IDs, labels, content
hashes, or test-size metadata. The per-axis source manifests include test
integrity metadata and are read by the runner only after `selection_frozen.json`
has been written.
