# Packaged binary baseline

This is the frozen primary word TF-IDF/logistic classifier from the binary study,
trained on the shared 4,000 documents. The model is copied byte-for-byte from the
completed fit; image builds perform no training. It was chosen as the primary
lexical reference before test evaluation, rather than selected for deployment by
comparing test scores.

`manifest.json` records the model checksum, runtime versions, study freeze,
temperature and calibration-fitted routing threshold. The API returns the binary
class decision and temperature-scaled probabilities. Its separate `action` skips
only routine predictions above the saved threshold; all other documents go to
review. The threshold targets a 1% empirical review-reference miss rate on the
500-document calibration cohort. It is not a population miss-rate guarantee.

The approximately 4.9 MB model contains a vocabulary and learned coefficients.
It contains no document-body records. The scientific evidence packages still
exclude model weights; this separately documented artifact is for CPU serving.
There are no neural weights or GPU dependencies in the image.

The artifact is committed so a clean clone can build and run offline inference
without private training data or API keys. Dependency installation during the image
build requires network access. To repackage from the preserved local study, run
`python -m decision_models.build_model`; it verifies frozen input hashes and copies
the selected artifact without fitting or opening the test cohort.
