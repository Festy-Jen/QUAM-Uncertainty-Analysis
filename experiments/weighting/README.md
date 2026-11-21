# QUAM trajectory-weighting diagnostic

This folder documents the **100-MNIST-image sensitivity analysis** described in the Practical Work report. The image predictions and loss arrays were generated with the upstream [`ml-jku/quam`](https://github.com/ml-jku/quam) pipeline. The stored `.pkl` files are deliberately excluded from this repository.

`check_quam_weighting.py` is a **reconstructed analysis script** based on the report's stated formula. **It is not the original diagnostic script that produced the historical numbers**, and it has **not been verified against the student's real QUAM MNIST pickle**. It computes per-sample Setting-B KL(reference || sample), then compares uniform aggregation over every stored step/model with training-loss softmax weights using temperature `0.001`. It also calculates a first-ten-steps sensitivity condition and reports the effective number of weighted samples.

From the repository root:

```bash
python experiments/weighting/check_quam_weighting.py --input data/quam_mnist_preds.pkl --n-images 100 --temperature 0.001
```

The reported historical summary was: unweighted mean `0.067677`; weighted-all mean `0.000583`; first-step total normalized weight approximately `0.971512`; effective sample count approximately `10.59`. **Do not assert that this reconstruction reproduces those exact values until the code has actually been run against the same saved file.** Differences may arise from trajectory selection, loss definition, normalization, numerical precision, or the original helper's weighting conventions.

Only unpickle files from trusted sources. The script does not retrain any models or determine a canonical QUAM sampling/weighting procedure. It leaves the main Setting A/B helper functions unchanged.
