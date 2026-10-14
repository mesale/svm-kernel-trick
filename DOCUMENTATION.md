# Documentation SVM & the Kernel Trick

This file has two parts: **Part A** is the short project report (≈2 pages); **Part B** is technical documentation of the code.

---

## Part A Short report

### 1. Dataset and approach
I used `make_moons(n_samples=600, noise=0.25, random_state=42)`: 600 points, 2 features, two interleaving half-circles, perfectly balanced (300/300). The crescents interlock, so every straight line cuts through one of them a linear model has a built-in error floor. The two features also have different spreads (std ratio ≈1.6), which matters because SVM kernels depend on distances.

The data was split 80/20 with stratification (480 train / 120 test). A `Pipeline(StandardScaler → SVC)` is used everywhere, so scaling statistics are computed on the training portion of each CV fold only; the test set is touched once, at the end. Parameters were chosen by **5-fold stratified cross-validation on the training set**, scored with **balanced accuracy**; the same folds were used for all models.

### 2. Models and parameters tested

| Kernel | Grid | Selected |
|---|---|---|
| Linear | `C` ∈ {0.1, 1, 10, 100} | `C=1` |
| Polynomial (`coef0=1`) | `C` ∈ {0.1, 1, 10, 100} × `degree` ∈ {2, 3} × `gamma` ∈ {scale, 0.1, 1} | `C=10, degree=3, gamma=1` |
| RBF | `C` ∈ {0.1, 1, 10, 100} × `gamma` ∈ {0.01, 0.1, 1, 10} | `C=10, gamma=1` |

Extra sweeps with `C` re-tuned each time: polynomial degree 1, 2, 3, 5 and RBF gamma 0.01, 0.1, 1, 10.

### 3. Main results

| Kernel | CV bal. acc. (±SD) | Test acc. | Test bal. acc. | Test F1 | Support vectors |
|---|---|---|---|---|---|
| Linear | 0.8646 ± 0.024 | 0.8917 | 0.8917 | 0.8943 | 164 (34%) |
| Polynomial d=3 | 0.9458 ± 0.017 | 0.9250 | 0.9250 | 0.9256 | 83 (17%) |
| RBF | 0.9500 ± 0.015 | 0.9333 | 0.9333 | 0.9333 | 89 (19%) |

* Linear is capped near 0.865 whatever `C` is a structural limit, not a tuning problem. The nonlinear kernels gain ≈8 points.
* Degree sweep (CV): d=1 0.865, d=2 0.867, d=3 0.946, d=5 0.944 → degree 3 is the smallest degree on the plateau.
* Gamma sweep (CV): 0.01 → 0.863 (underfit, almost straight boundary), 0.1 → 0.910, **1 → 0.950**, 10 → 0.942 (wigglier boundary, needs `C=0.1`). Best `C` moves down as gamma grows, so the two were tuned jointly.
* The selected RBF model's test confusion matrix is `[[56, 4], [4, 56]]` 4 errors per class, consistent with the label noise from `noise=0.25`.
* The linear model's test score (0.892) is *higher* than its CV score (0.865); with only 120 test rows this is sampling noise, so CV is the more trustworthy estimate.
* Noise robustness (5 random datasets per level): the nonlinear kernels stay ahead of linear at every noise level, converging as noise approaches 0.4 and the classes genuinely overlap.

### 4. The Kernel Trick
**Idea.** A linear SVM only ever uses inner products `xᵢ·xⱼ` between training points. Replacing them with `K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)` gives a linear SVM in feature space φ a nonlinear boundary in the original space while φ is never computed.

**Demonstration (notebook §11).** For degree 2, `γ=1`, `coef0=1`, `K(x,z) = (x·z + 1)²` equals the inner product of the explicit 6-D map `φ(x) = [1, √2·x₁, √2·x₂, x₁², x₂², √2·x₁x₂]`: the two 480×480 matrices differ by ≈1e-14. An SVM trained on φ(x) and an SVM with `kernel="poly"` produce the same decision function (difference ≈1e-12) and 0/480 different predictions. It is an exact identity, not an approximation.

**When it helps.** The explicit space grows combinatorially: with 10 input features a degree-5 map has 3,002 dimensions, and 50 features at degree 8 would need ≈1.9 billion. The kernel matrix stays n×n. For RBF the feature space is infinite-dimensional, so the kernel is the only route.

**Honest finding on cost.** With only 2 input features the explicit map is tiny, so explicit features + a linear SVM was actually *faster* than the kernel route in my benchmark (kernel ≈2× slower at degree 3 and >100× slower at degree 8 for n=3200; the degree-8 case is most likely a poor-conditioning effect of large kernel values). So the kernel trick is not "always faster"; its advantage is that its cost does not depend on the width of the implicit space.

**Limitations.** The kernel matrix is O(n²) in memory (≈80 GB dense for 100k rows) and training scales between O(n²) and O(n³), so very large datasets need `LinearSVC`/SGD or kernel approximations (Nyström, random Fourier features). Results also depend strongly on `C`, `gamma` and `degree`: too small a gamma underfits, too large overfits, and large degrees make the solver slow.

### 5. Selected model and justification
**RBF kernel, `C=10`, `gamma=1`** (test accuracy / balanced accuracy / F1 = 0.933). RBF and polynomial d=3 are tied within noise (0.4 pt CV gap vs fold SD ≈1.5–2.4 pt), so I chose on other grounds: (i) no degree to choose one length scale `gamma` plus `C`; (ii) a local kernel, so the boundary is bounded and well behaved away from the data the polynomial boundary visibly extends in an unconstrained straight-ish arm at the left edge of the plot, a sign of poor extrapolation; (iii) it was best on CV and on test, with the lowest CV variance of the three at the best setting; and (iv) it needs far fewer support vectors than the linear model (89 vs 164). The test set played no part in the choice.

### 6. What I learned
* A model's ceiling is set by the data's geometry: no `C` rescues a linear SVM on interlocking moons.
* The kernel trick is an exact algebraic identity that lets the SVM work in rich (even infinite) feature spaces at a cost set by the number of samples.
* `gamma` and `C` interact; they must be tuned together, with scaling inside the cross-validation pipeline to avoid leakage.
* Don't over-read small score gaps: differences below the fold SD are not evidence of a better model.
* Textbook claims such as "the kernel trick is faster" depend on dimensionality; I only believed it after measuring it.

---

## Part B Technical documentation

### Files
| File | Purpose |
|---|---|
| `svm_kernel_trick.ipynb` | Full experiment, sections 1–12, run top-to-bottom |
| `src/svm_utils.py` | Helper functions (below) |
| `artifacts/results.json` | Headline numbers written by the last cell |

### `src/svm_utils.py`
* `plot_decision_boundary(model, X, y, title="", ax=None)` takes a fitted `Pipeline` ending in a step named `"svc"` and the training data in original units; draws predicted regions, the boundary (decision function = 0), dashed margin lines (±1) and circled support vectors (found via `svc.support_`).
* `median_pairwise_distance(X)` median Euclidean distance between distinct rows.
* `suggested_gamma(X)` `1 / median distance`; a starting value for RBF `gamma`, not a replacement for CV.

### Notebook sections
| § | Content |
|---|---|
| 1–2 | Imports; data generation, visualisation, structure, noise sweep |
| 3–4 | Stratified split; scaling demo and leakage explanation |
| 5 | Baseline linear SVM with cross-validation |
| 6 | `GridSearchCV` for linear, polynomial, RBF; sensitivity to `C` |
| 7 | Decision boundaries & margins; degree sweep; gamma sweep; gamma heuristic |
| 8 | Support-vector counts per kernel |
| 9 | Robustness to noise |
| 10 | Model selection, test metrics, confusion matrices |
| 11 | Kernel trick: φ-vs-kernel check, feature growth, timing, limitations |
| 12 | Conclusions; `results.json` export |

### Design decisions
* **Scaler inside the `Pipeline`** prevents leakage during CV and at prediction time.
* **Balanced accuracy for model selection** robust to class imbalance (irrelevant here, but safe); accuracy and F1 are also reported.
* **Same `StratifiedKFold(5, shuffle=True, random_state=42)` for every model** comparisons are paired.
* **`coef0=1` for the polynomial kernel** makes it include all lower-degree terms (otherwise degree-d homogeneous only).
* **Grid edges** selected `C=10` is interior to {0.1…100}; gamma=1 is interior to {0.01…10}.

### Known limitations
* Single 80/20 split; with 120 test points one error ≈ 0.8 pt, so test scores differ from CV by noise.
* The robustness experiment (§9) reuses hyperparameters tuned at `noise=0.25`; it is not re-tuned per noise level.
* Timing numbers depend on the machine and are not exact; read them as orders of magnitude.

### Reproducing
`pip install -r requirements.txt`, then restart and run all cells in `svm_kernel_trick.ipynb`. Seed: `RANDOM_STATE = 42`.
