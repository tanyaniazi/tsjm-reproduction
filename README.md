# TSJM: Two-Stage Joint Modeling Reproduction (Python)

A Python reproduction and practical walkthrough of the **Two-Stage Joint Modeling (TSJM)** methodology for multiple longitudinal markers and time-to-event outcomes, based on the paper:

> **Baghfalaki, T., et al. (2026).** *A two-stage joint modeling approach for multiple longitudinal markers and time-to-event data.* Statistical Methods in Medical Research.

---

## 📌 Overview

Full multivariate joint models (MMJM) become computationally prohibitive or fail to converge when modeling many longitudinal biomarkers simultaneously. Traditional two-stage approaches (Standard Two-Stage, STS) fit linear mixed models separately in Stage 1, but suffer from **informative dropout bias** because they ignore the event process.

**TSJM solves this problem:**
1. **Stage 1 (One-Marker Joint Models):** Fits a separate bivariate joint model for each individual marker alongside the survival endpoint. This corrects the predicted longitudinal trajectories for dropout bias while maintaining computational tractability.
2. **Stage 2 (Multivariate Cox Model):** Feeds the denoised predicted trajectories ($\hat{\eta}_{ik}(t)$) into a multivariate Cox proportional hazards model as time-dependent covariates.
3. **Dynamic Prediction:** Computes landmark-based individual risk probabilities over time ($\pi_i(s, \Delta t)$).

---

## 📂 Repository Structure

```text
├── TSJM-Reproduction.ipynb    # Main step-by-step reproduction notebook
├── tsjm_core.py              # Python implementation of Stage 1 & 2 estimation routines
├── dataLong.csv              # Longitudinal marker dataset (simulated benchmarks)
├── dataSurv.csv              # Survival endpoint dataset (id, survtime, death)
├── TSJM-main/                # Original R package source code by Baghfalaki et al.
├── paper_extract.txt         # Plain text extracted from paper for reference
└── baghfalaki-et-al-2026...  # Original published paper PDF
```

---

## 🚀 Notebook Pipeline (`TSJM-Reproduction.ipynb`)

| Step | Section | Description |
| :---: | :--- | :--- |
| **1** | **Data Setup** | Separate longitudinal & survival tables, subject-level train/validation split. |
| **2** | **Model & Skeletons** | Matrix builders (`build_XZ`), true trajectory formulation ($\eta = X\beta + Zb$), and hazard specifications. |
| **3** | **Stage 1 (Single Marker)** | Fits one-marker joint model ($Y_1$ + survival) to verify parameter convergence. |
| **4** | **Stage 1 (All Markers)** | Fits models for $Y_1, \dots, Y_4$ and extracts clean predicted trajectories $\hat{\eta}_{ik}(t)$. |
| **5** | **Stage 2 (Cox Model)** | Multi-marker survival model using time-dependent $\hat{\eta}_{ik}(t)$ and baseline covariates. |
| **6** | **Rubin's Rules** | Accounting for Stage 1 parameter uncertainty in Stage 2 standard errors. |
| **7** | **Dynamic Prediction** | Patient-specific landmark risk forecasting (Algorithm 1 from paper). |
| **8** | **Validation Metrics** | Time-dependent AUC and Brier Score evaluation across landmark times $s$. |

---

## 📊 Data Variables

The provided demonstration dataset uses the simulated benchmark data from the authors' R package:

- **`id`**: Unique patient identifier.
- **`obstime`**: Visit / observation time for longitudinal measurements.
- **`x1`, `x2`**: Baseline covariates (e.g., treatment group, standardized baseline age).
- **`Y1` – `Y4`**: Repeated, noisy biomarker measurements.
- **`survtime`**: Time to event or right-censoring.
- **`death`**: Event indicator ($1 = \text{event occurred}$, $0 = \text{censored}$).

---

## 🛠️ Requirements & Installation

Python 3.9+ with standard scientific libraries:

```bash
pip install numpy scipy pandas matplotlib lifelines
```

Clone the repository and open the notebook:

```bash
git clone https://github.com/<your-username>/tsjm-reproduction.git
cd tsjm-reproduction
jupyter notebook TSJM-Reproduction.ipynb
```

---

## 📖 Citation

```bibtex
@article{baghfalaki2026twostage,
  title={A two-stage joint modeling approach for multiple longitudinal markers and time-to-event data},
  author={Baghfalaki, Taban and others},
  journal={Statistical Methods in Medical Research},
  year={2026}
}
```
