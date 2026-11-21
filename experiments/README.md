# Practical-work experimental supplements

These files supplement the central image prediction analysis. They are deliberately separate from the main `run_all.py` workflow.

- `experiments/llm/`: an exploratory Qwen output-head perturbation notebook and an audit summary of saved MAQA*/AmbigQA* results. This does **not** implement QUAM.
- `experiments/weighting/`: a transparent independent *reconstruction* of the weighting sensitivity diagnostic. The original script used to obtain the historical results was not available; this new script must be checked against the real QUAM MNIST pickle before it can be described as numerical reproduction.

Do not place original large prediction files, model checkpoints, or Colab outputs in the public source repository.
