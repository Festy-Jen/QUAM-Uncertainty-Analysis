# Cleanup changes and validation scope

This branch improves documentation, plotting, and reproducibility checks while retaining the original uncertainty calculations. Prediction-generation code and saved `.pkl` inputs were not changed.

## Changes made

- `README.md`: installation, provenance, reproducibility, input layout, explicit numerical/interpretation limitations and commands.
- `requirements.txt`: direct dependencies including `pandas` and `seaborn`. Preserved original freeze in `requirements-original-lock.txt`; added `requirements-dev.txt` for tests.
- `analysis_utils.py`: consolidated duplicated pickle loading, tensor-shape validation, conditional softmax, clipping and A/B dispatch. The historical numerical conventions were deliberately preserved.
- `correlation_analysis.py`: CLI options; one pickle load per method; full-precision Pearson/Spearman CSV exports; bold contrast-aware annotations; PNG plus vector PDF; individual matrices suited to report layout; optional pairplot.
- `make_plots.py`: CLI method/dataset/setting, optional axis limits, non-interactive figure saving.
- `py_combined.py`: CLI arguments, same extreme-point plotting logic, PNG and PDF output, neutral comments.
- `run_all.py`: single command to run correlations for requested datasets and optional exploratory figures.
- `helper_functions.py`: provenance and comment cleanup; **the Setting A and B numerical implementation remains unchanged**; fixed the misleading `#retention` comment and noninteractive `plt.show` warning.
- `RUN_LEDGER.csv`: placeholder for **actual** seed, checkpoint, config and command per generated pickle. Entries are marked `UNVERIFIED`, not inferred from source templates.
- `tests/`: mathematical regression tests and figure smoke tests with synthetic probability arrays.

## Validation performed

1. `python -m compileall` completed.
2. `python -m pytest -q` passed all 11 tests.
3. Figure smoke tests generated Pearson/Spearman PDF/PNG/CSV, an individual hexbin and a 2x2 extreme-point plot using *synthetic* data.

**Local prediction-data check:** `python run_all.py --datasets mnist emnist cifar10 --with-individual` completed successfully using the saved prediction files. The regenerated Pearson/Spearman coefficients agreed with the previous figures to their displayed precision. This checks consistency with the earlier analysis; it is not an independent validation of checkpoint provenance or the uncertainty estimators.

## Remaining validation and reproducibility gaps

- `samples.min()<0 or max()>1` does **not** validate probability normalization; retained to preserve previous results.
- Clipping to `1e-14` before computing correlations introduces ties; retained to preserve previous results.
- The original QUAM trajectory-weighting choices are not automatically reproduced by this exploratory code.
- Matching target sequences does not prove image-level alignment.
- `evaluate_missclass` outputs AUROC/AUPR/FPR approximations, not accuracy-retention curves.
- Checkpoint paths, hardware, runtime and seeds cannot be recovered from a generic `.pkl` or config template; complete `RUN_LEDGER.csv` using actual historical run evidence.
- The LLM Colab proof-of-concept is *not* included in this image-analysis codebase; document/submit it separately if required.
- The Overleaf report can use the updated vector heatmaps; report edits are maintained separately from this analysis repository.
