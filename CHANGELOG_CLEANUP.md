# Cleanup changes and validation scope

This is a **submission-preparation branch/copy**, not a claimed scientific rerun. No existing `data/*.pkl` files, original Git history, or student's published figures were modified.

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
- Submission archive: removed `.git/`, caches, unused exploratory plots, prediction files. The independent original archive is unchanged.

## Validation performed

1. `python -m compileall` completed.
2. `python -m pytest -q` passed all synthetic tests.
3. Figure smoke tests generated Pearson/Spearman PDF/PNG/CSV, an individual hexbin and a 2x2 extreme-point plot using *synthetic* data.

**Not performed:** complete end-to-end rerun on the original MNIST/EMNIST/CIFAR-10 prediction files because those local pickle arrays were not in the uploaded repository ZIP. Therefore, absolute Pearson/Spearman values have not been independently reproduced. Run the commands from README against your own `data/` and compare their two-decimal annotations to the previously submitted figures before replacing report figures.

## Caveats still requiring student validation

- `samples.min()<0 or max()>1` does **not** validate probability normalization; retained to preserve previous results.
- Clipping to `1e-14` before computing correlations introduces ties; retained to preserve previous results.
- The original QUAM trajectory-weighting choices are not automatically reproduced by this exploratory code.
- Matching target sequences does not prove image-level alignment.
- `evaluate_missclass` outputs AUROC/AUPR/FPR approximations, not accuracy-retention curves.
- Checkpoint paths, hardware, runtime and seeds cannot be recovered from a generic `.pkl` or config template; complete `RUN_LEDGER.csv` using actual historical run evidence.
- The LLM Colab proof-of-concept is *not* included in this image-analysis codebase; document/submit it separately if required.
- Your editable Overleaf report source, not this source-only archive, must be updated to embed the new vector heatmaps. This cleanup does not alter the PDF already sent to your supervisor.
