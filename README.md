# Two-Stage Joint Modeling (TSJM) — Python Reproduction & Theory Guide

> **A comprehensive theoretical and practical walkthrough of Two-Stage Joint Modeling for multi-marker longitudinal trajectories and time-to-event outcomes.**  
> Based on the foundational methodology of **Baghfalaki et al. (2026)**, this repository contains both deep conceptual lecture notes and an end-to-end Python reproduction of the algorithm, dynamic risk prediction, and validation pipelines.

---

## 📑 Table of Contents
1. [Core Concepts & Theoretical Foundations](#-core-concepts--theoretical-foundations)
   - [The Clinical Motivation & Challenges](#1-the-clinical-motivation--challenges)
   - [Survival Analysis & The Cox Proportional Hazards Model](#2-survival-analysis--the-cox-model)
   - [Linear Mixed Models (Longitudinal Submodel)](#3-linear-mixed-effects-models-lmm)
   - [The Standard Two-Stage Flaw: Informative Dropout Bias](#4-the-standard-two-stage-flaw-informative-dropout-bias)
   - [The TSJM Breakthrough: Stage 1 Single-Marker Joint Models](#5-the-tsjm-breakthrough-stage-1-single-marker-joint-models)
   - [Uncertainty Propagation: Multiple Imputation & Rubin's Rules](#6-uncertainty-propagation--rubins-rules)
   - [Dynamic Prediction (Algorithm 1 vs. Algorithm 2)](#7-dynamic-prediction-individualized-risk-over-time)
2. [Code Reproduction & Architecture](#-code-reproduction--architecture)
   - [Repository Structure](#repository-structure)
   - [Walkthrough of Steps in the Notebook](#step-by-step-notebook-pipeline)
   - [Data Dictionary](#data-variables)
3. [Setup & Quickstart](#-setup--quickstart)
4. [References](#-references)

---

## 🧠 Core Concepts & Theoretical Foundations

### 1. The Clinical Motivation & Challenges
In modern clinical trials and longitudinal cohort studies (e.g., liver disease, dementia, cardiovascular health), doctors repeatedly collect multiple continuous biomarkers ($Y_1, Y_2, \dots, Y_K$) at irregular hospital visits alongside tracking the time until a clinical milestone occurs (e.g., death, relapse, diagnosis).

**The Two Fundamental Challenges:**
1. **Measurement Error & Noise:** Laboratory blood tests and clinical exams have high day-to-day noise ($\varepsilon$). Raw observations fluctuate wildly around the true biological state.
2. **Intermittent / Irregular Visits:** Patients do not show up every day. A survival model needs to evaluate risk continuously at every event time $t$, but we only have sparse, discrete snapshots.

Standard Cox models fail here because feeding raw, intermittent, noisy marker values directly into survival analysis causes severe **attenuation bias** (regression dilution).

---

### 2. Survival Analysis & The Cox Model
A survival model predicts the instantaneous rate of experiencing an event, quantified by the **hazard function**:

$$h(t \mid X) = h_0(t) \exp(X \beta)$$

- **$h_0(t)$ (Baseline Hazard):** The nonparametric underlying risk over time when all covariates are zero. It cancels out when taking hazard ratios between patients.
- **$\exp(\beta_j)$ (Hazard Ratio):**
  - $\beta_j > 0 \implies \text{HR} > 1$ (increases hazard / risk).
  - $\beta_j < 0 \implies \text{HR} < 1$ (protective / decreases hazard).

#### Parameter Estimation: Cox Partial Likelihood & Newton–Raphson
The baseline hazard $h_0(t)$ is left unspecified. The coefficients $\beta$ are estimated by maximizing Cox's **log partial likelihood**:

$$\ell(\beta) = \sum_{i=1}^{K} \left[ X_i \beta - \ln \sum_{j \in \mathcal{R}_i} \exp(X_j \beta) \right]$$

where $\mathcal{R}_i$ is the **risk set** (the set of patients alive and under observation just prior to time $t_i$). The model finds the optimum using the **Newton–Raphson algorithm**:

$$\beta_{\text{new}} = \beta_{\text{old}} + \mathcal{I}(\beta_{\text{old}})^{-1} \mathcal{U}(\beta_{\text{old}})$$

- $\mathcal{U}(\beta)$ = Score function (first derivative / gradient).
- $\mathcal{I}(\beta)$ = Information matrix (negative second derivative / curvature).
- The baseline cumulative hazard is then recovered using the **Breslow estimator**:
  $$h_0(t_i) = \frac{d_i}{\sum_{j \in \mathcal{R}_i} \exp(X_j \beta)}$$

---

### 3. Linear Mixed-Effects Models (LMM)
To model how a patient's health markers evolve continuously over time, we use a linear mixed model for marker $k$ of patient $i$ at visit time $s$:

$$Y_{ik}(s) = \eta_{ik}(s \mid \boldsymbol{\beta}_k, \mathbf{b}_{ik}) + \varepsilon_{ik}$$

$$\eta_{ik}(s) = \mathbf{x}_{ik}(s)^\top \boldsymbol{\beta}_k + \mathbf{z}_{ik}(s)^\top \mathbf{b}_{ik}$$

- **$\eta_{ik}(s)$:** The **true, error-free biological trajectory** of the marker at time $s$.
- **$\boldsymbol{\beta}_k$ (Fixed Effects):** Population-average parameters shared by all individuals (e.g., baseline intercept and average slope over time).
- **$\mathbf{b}_{ik}$ (Random Effects):** Patient-specific deviations from the population average ($\mathbf{b}_i \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$). Each patient gets their own personalized baseline ($b_0$) and progression rate ($b_1$).
- **$\varepsilon_{ik} \sim \mathcal{N}(0, \sigma_k^2)$:** Random Gaussian measurement noise.

---

### 4. The Standard Two-Stage Flaw: Informative Dropout Bias
A naïve two-stage strategy (Standard Two-Stage, STS) operates in sequence:
1. **Stage 1:** Fit $K$ separate mixed models on the longitudinal data alone to predict health trajectories $\hat{\eta}_{ik}(t)$.
2. **Stage 2:** Plug $\hat{\eta}_{ik}(t)$ into a Cox model as time-dependent covariates to predict survival.

#### The Fatal Flaw:
Stage 1 is fitted in complete isolation from the survival data. It has **no awareness that the sickest patients experienced the event and dropped out of the study early**.
Because patients with deteriorating trajectories drop out, the remaining observed visits are dominated by healthier survivors. A plain mixed model fits an **overly optimistic trajectory**, leading to biased estimates in Stage 2.

---

### 5. The TSJM Breakthrough: Stage 1 Single-Marker Joint Models
Full multivariate joint models (MMJM) attempt to estimate all $K$ markers and survival at the same time, but when $K \ge 4$ or $10+$, the random-effects covariance matrix blows up and becomes computationally intractable.

**TSJM's Solution:**  
Instead of $K$ separate mixed models, TSJM fits **$K$ separate one-marker joint models** in Stage 1:

$$\text{Marker } k + \text{Survival Outcome } (T_i, \delta_i) \quad \text{fitted jointly for each } k \in \{1, \dots, K\}$$

#### Letting Survival Data "Talk" to the Trajectory
Under the Bayesian formulation:

$$P(\boldsymbol{\theta}, \mathbf{b}_i \mid \text{Data}) \propto \underbrace{P(\text{Longitudinal } Y_{ik} \mid \mathbf{b}_i)}_{\text{How well trajectory fits lab points}} \times \underbrace{P(\text{Survival } T_i, \delta_i \mid \mathbf{b}_i)}_{\text{How well hazard explains death time}} \times P(\mathbf{b}_i)$$

- If a patient dies at $t = 1.5$ years, a candidate trajectory guessing a slow, healthy slope will yield a survival likelihood near zero.
- The model automatically discards unrealistic "healthy" slopes and selects a steep, deteriorating slope that properly aligns with their early death.
- **Result:** Trajectories are corrected for dropout bias at a fraction of the computational cost of a full multivariate model.

---

### 6. Uncertainty Propagation & Rubin's Rules
A major pitfall of two-stage models is **"letting upstream models lie to downstream models"** by treating estimated trajectories $\hat{\eta}$ as known, rigid facts. Because $\hat{\eta}$ comes from an estimated posterior distribution, it carries inherent estimation doubt.

To prevent artificial confidence intervals and inflated false-positive rates, TSJM applies **Multiple Imputation and Rubin's Rules**:

1. **Draw $M$ Imputations:** Sample $M$ realizations of random effects $\mathbf{b}_{ik}^{(m)}$ from their posterior distribution (e.g., $M = 100$).
2. **Fit $M$ Cox Models:** Run the Stage 2 Cox model separately for each imputation, obtaining estimates $\hat{\boldsymbol{\alpha}}^{(m)}$ and variances $W_m$.
3. **Pool with Rubin's Formula:**
   - **Pooled Estimate:**
     $$\bar{\boldsymbol{\alpha}} = \frac{1}{M} \sum_{m=1}^{M} \hat{\boldsymbol{\alpha}}^{(m)}$$
   - **Within-Imputation Variance ($\bar{W}$):** Average of internal variances across runs.
   - **Between-Imputation Variance ($B$):** How much the estimates bounced around across the $M$ draws:
     $$B = \frac{1}{M - 1} \sum_{m=1}^{M} (\hat{\boldsymbol{\alpha}}^{(m)} - \bar{\boldsymbol{\alpha}})^2$$
   - **Total Honest Variance ($T$):**
     $$T = \bar{W} + \left(1 + \frac{1}{M}\right) B$$
   - **Honest Standard Error:**
     $$\text{SE} = \sqrt{T} \implies 95\% \text{ CI} = \bar{\boldsymbol{\alpha}} \pm 1.96 \times \text{SE}$$

---

### 7. Dynamic Prediction: Individualized Risk Over Time
Standard survival analysis only calculates baseline risk at enrollment ($t = 0$). **Dynamic Prediction** computes real-time conditional survival probabilities for an active patient at landmark time $s$ using all history up to $s$:

$$\pi_i(s, \Delta t) = P(T_i > s + \Delta t \mid T_i > s, \mathcal{Y}_i(s))$$

Using the fundamental law of survival analysis connecting cumulative hazard $\Delta H$ to survival:

$$P(\text{Survival}) = \exp(-\Delta H) \implies \text{Risk of Event} = 1 - \exp(-\Delta H)$$

#### Two Computation Algorithms:
- **Algorithm 1 (First-Order / Plug-in):**
  Plugs the posterior mean $\hat{\mathbf{b}}_i$ directly into the cumulative hazard formula. Extremely fast and clinically accurate for point predictions.
- **Algorithm 2 (Monte Carlo Simulation):**
  Draws $L$ samples (e.g., $L = 500$) from both parameter uncertainty and random effects distributions, producing full empirical distribution curves and **95% prediction intervals (error bars)**.

---

## 💻 Code Reproduction & Architecture

### Repository Structure
```text
├── TSJM-Reproduction.ipynb    # Main step-by-step reproduction notebook
├── tsjm_core.py              # Core Python routines for Stage 1 & 2 estimation
├── dataLong.csv              # Longitudinal marker dataset (simulated benchmarks)
├── dataSurv.csv              # Survival endpoint dataset (id, survtime, death)
├── mynotes.pdf               # Reference handwritten lecture/study notes (8 pages)
└── README.md                 # Theoretical guide and repository documentation
```

### Step-by-Step Notebook Pipeline

| Step | Focus | Purpose & Code Role |
| :---: | :--- | :--- |
| **1** | **Data Setup** | Separates longitudinal visits and survival endpoints; executes subject-level train/validation split. |
| **2** | **Model & Skeletons** | Implements matrix builders (`build_XZ`), true linear predictor ($\eta = X\beta + Zb$), and hazard linear predictors. |
| **3** | **Stage 1 (Single Marker)** | Fits a joint model for $Y_1$ + survival to test convergence and parameter recovery. |
| **4** | **Stage 1 (All Markers)** | Loops Stage 1 across $Y_1, \dots, Y_K$ to extract denoised trajectories $\hat{\eta}_{ik}(t)$ for every patient. |
| **5** | **Stage 2 (Cox Model)** | Fits multi-marker Cox proportional hazards model using time-dependent $\hat{\eta}$ and baseline variables. |
| **6** | **Rubin's Rules** | Pools variance across posterior draws to produce honest standard errors and confidence intervals. |
| **7** | **Dynamic Prediction** | Implements Algorithm 1 to calculate real-time conditional failure probabilities $\pi_i(s, \Delta t)$. |
| **8** | **Validation Metrics** | Computes landmark time-dependent AUC (discrimination) and Brier Score (calibration). |

### Data Variables

- **`id`**: Unique patient identifier (clusters repeated measurements).
- **`obstime`**: Time of hospital visit / observation.
- **`x1`**: Binary baseline covariate ($\sim \text{Bernoulli}(0.5)$, e.g., treatment arm or sex).
- **`x2`**: Continuous baseline covariate ($\sim \mathcal{N}(0, 0.5^2)$, e.g., standardized age).
- **`Y1` – `Y4`**: Repeated continuous biomarker values measured with noise.
- **`survtime`**: Time of event occurrence or right-censoring.
- **`death`**: Censoring indicator ($1 = \text{event occurred}$, $0 = \text{censored}$).

---

## 🚀 Setup & Quickstart

### Prerequisites
Python 3.9+ with standard scientific libraries:
```bash
pip install numpy scipy pandas matplotlib lifelines
```

### Run the Reproduction
```bash
git clone https://github.com/tanyaniazi/tsjm-reproduction.git
cd tsjm-reproduction
jupyter notebook TSJM-Reproduction.ipynb
```

---

## 📖 References

- **Baghfalaki, T., et al. (2026).** *A two-stage joint modeling approach for multiple longitudinal markers and time-to-event data.* Statistical Methods in Medical Research.
- **Rubin, D. B. (1987).** *Multiple Imputation for Nonresponse in Surveys.* John Wiley & Sons.
- **Cox, D. R. (1972).** *Regression models and life-tables (with discussion).* Journal of the Royal Statistical Society: Series B.
