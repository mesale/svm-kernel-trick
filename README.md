# Support Vector Machines & the Kernel Trick

Binary classification on the **two-moons** dataset. A **linear SVM** baseline is compared with **polynomial** and **RBF** kernels, and a dedicated section demonstrates *why* the kernel trick works (and what it costs).

| Kernel | Tuned parameters (5-fold stratified CV, balanced accuracy) | CV bal. acc. | Test accuracy | Test bal. acc. | Test F1 |
|---|---|---|---|---|---|
| Linear (baseline) | `C=1` | 0.8646 | 0.8917 | 0.8917 | 0.8943 |
| Polynomial | `C=10, degree=3, gamma=1, coef0=1` | 0.9458 | 0.9250 | 0.9250 | 0.9256 |
| **RBF (selected)** | `C=10, gamma=1` | **0.9500** | **0.9333** | **0.9333** | **0.9333** |

Linear stalls at ≈0.865 because the classes are not linearly separable; both nonlinear kernels gain ≈8 points. RBF and degree-3 polynomial are statistically tied (difference ≈0.4 pt vs. fold SD ≈1.5–2.4 pt), so RBF is chosen on reasoning (see `DOCUMENTATION.md`).

## Repository layout

```
svm-kernel-trick/
├── svm_kernel_trick.ipynb   # main notebook (run top to bottom)
├── src/svm_utils.py         # decision-boundary plot + gamma heuristics
├── artifacts/results.json   # headline numbers written by the notebook
├── DOCUMENTATION.md         # short report + technical documentation
├── requirements.txt
└── README.md
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook svm_kernel_trick.ipynb              # Kernel > Restart & Run All
```

Runtime is roughly 1 minute on a laptop (most of it is the timing benchmark in §11). Everything is seeded with `RANDOM_STATE = 42`, so results are reproducible. Tested with scikit-learn 1.9.1.

## What the notebook covers

1. **Data** – `make_moons(n=600, noise=0.25)`, visualised, with a note on why no straight line separates the classes; noise sweep.
2. **Split & scaling** – stratified 80/20 split; `StandardScaler` lives *inside* a `Pipeline`, so it is re-fitted on each training fold only (no leakage). The test set is used only in §10.
3. **Models** – linear (`C`), polynomial (`C` × `degree` ∈ {2, 3} × `gamma`), RBF (`C` × `gamma`), all with `GridSearchCV` on the same folds. Extra sweeps: degree 1/2/3/5 and gamma 0.01/0.1/1/10.
4. **Metrics** – accuracy, balanced accuracy, F1, classification report, confusion matrices.
5. **Visualisation** – decision boundaries, margins (±1) and circled support vectors for every kernel.
6. **Kernel trick study** – numerical proof that `K(x,z) = φ(x)·φ(z)`, identical decision functions for kernel vs explicit feature map, feature-count explosion table, timing benchmark and limitations.
7. **Model selection** – RBF, with justification; robustness-to-noise experiment.

See `DOCUMENTATION.md` for the report and details.
