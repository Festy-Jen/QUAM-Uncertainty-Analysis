# Exploratory LLM output-head perturbations

This supplement records the Google Colab investigation discussed in the Practical Work report. It used `Qwen/Qwen1.5-4B`, `ttomov/maqa_star` and `ttomov/ambigqa_star`. It is **not** an implementation of the QUAM adversarial search procedure.

The procedure supplies each candidate answer as part of the prompt, selects the ten most probable next tokens, adds independent Gaussian noise (std `0.01`) to corresponding output-head weight rows for `30` samples, and applies Setting-B-style uncertainty formulas to the resulting ten-class predictions. The scores therefore characterize next-token behavior *after the supplied answer*, not probabilities over competing answers.

`output_head_perturbation.ipynb` is an organized reproduction of the original Colab experiment. Its `RUN_INFERENCE=False` default avoids loading or running the 4B model unintentionally. It retains the original `float16` arithmetic by default. The alternative float32 numerical path is **for a new rerun**, not a replacement of the original stored results. Model revision, seed, and Colab package versions were not fully captured for the historical run.

The separately audited historical files yielded:

| Dataset | Questions | Answer variants | All four metrics finite |
|---|---:|---:|---:|
| MAQA* | 592 | 2,669 | 2,666 (99.89%) |
| AmbigQA* | 4,377 | 12,920 | 306 (2.37%) |

Those aggregates are in `saved_results_audit.csv`; raw historical `.pkl` files are **not included**. The original AmbigQA* plots cannot be interpreted as representative whole-dataset correlations because of severe nonfinite outputs. The notebook is syntax-checked but has not been rerun with Qwen during repository preparation.

To inspect saved results in Colab, point `SAVED_RESULTS` in the notebook at a trusted `quam_maqa_results.pkl` or `quam_ambigqa_results.pkl` file. The notebook reports the jointly finite fraction explicitly and plots only that subset. Only unpickle files from trusted sources.
