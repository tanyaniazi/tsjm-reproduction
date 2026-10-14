"""
Core helpers for a pedagogical TSJM reproduction (Baghfalaki et al.).

Stage 1 here is a *shared-parameter one-marker joint model* fit by joint MAP
(random effects treated as parameters, Gaussian prior). This mirrors the paper's
Stage-1 idea (marker + survival together) without needing JAGS. Stage 2 uses a
Cox model with time-dependent current-value predictions (lifelines).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


# ---------------------------------------------------------------------------
# Design / bookkeeping
# ---------------------------------------------------------------------------

def subject_index_maps(ids: np.ndarray):
    """Map raw ids -> 0..n-1 contiguous indices."""
    uniq = pd.unique(ids)
    id_to_i = {int(u): i for i, u in enumerate(uniq)}
    idx = np.array([id_to_i[int(u)] for u in ids], dtype=int)
    return uniq.astype(int), id_to_i, idx


def piecewise_cuts(times: np.ndarray, n_internal: int = 4) -> np.ndarray:
    """Internal cutpoints like the R package (quantiles of observed times)."""
    qs = np.linspace(0.2, 0.8, n_internal)
    return np.unique(np.quantile(times, qs))


def build_linear_design(obstime: np.ndarray):
    """X=Z=[1,t] for linear random intercept + slope."""
    t = np.asarray(obstime, float)
    X = np.column_stack([np.ones(len(t)), t])
    return X, X.copy()


# ---------------------------------------------------------------------------
# Survival pieces: λ(t)=h(interval)*exp(A+B t), closed-form cumulative hazard
# ---------------------------------------------------------------------------

def _interval_hazards(t: float, cuts: np.ndarray, h: np.ndarray) -> float:
    """Piecewise-constant baseline hazard value at time t."""
    # cuts are internal; J = len(cuts)+1 pieces
    edges = np.concatenate([[0.0], cuts, [np.inf]])
    for j in range(len(h)):
        if edges[j] < t <= edges[j + 1] or (j == 0 and t == 0):
            return h[j]
    return h[-1]


def cumhaz_piecewise_exp(T: float, cuts: np.ndarray, h: np.ndarray, A: float, B: float) -> float:
    """Λ(T)=∫_0^T h(u) exp(A+B u) du with piecewise-constant h."""
    if T <= 0:
        return 0.0
    edges = np.concatenate([[0.0], cuts, [T]])
    # only up to T
    edges = edges[edges <= T]
    if edges[-1] < T:
        edges = np.append(edges, T)
    total = 0.0
    for j in range(len(edges) - 1):
        lo, hi = edges[j], edges[j + 1]
        if hi <= lo:
            continue
        # which piece? use midpoint
        hj = _interval_hazards(0.5 * (lo + hi), cuts, h)
        if abs(B) < 1e-10:
            total += hj * np.exp(A) * (hi - lo)
        else:
            total += hj * np.exp(A) * (np.exp(B * hi) - np.exp(B * lo)) / B
    return float(total)


def hazard_at(T: float, cuts: np.ndarray, h: np.ndarray, A: float, B: float) -> float:
    return float(_interval_hazards(T, cuts, h) * np.exp(A + B * T))


# ---------------------------------------------------------------------------
# Stage 1: one-marker joint MAP
# ---------------------------------------------------------------------------

def pack_params(beta, log_sigma, log_sd_b, rho_z, gamma, alpha, log_h, b):
    # rho via tanh for |rho|<1; D = cov of (b0,b1)
    return np.concatenate([
        np.asarray(beta, float),
        [log_sigma, log_sd_b[0], log_sd_b[1], rho_z],
        np.asarray(gamma, float),
        [alpha],
        np.asarray(log_h, float),
        np.asarray(b, float).ravel(),
    ])


def unpack_params(theta, n_subj, n_gamma, n_h):
    i = 0
    beta = theta[i:i + 2]; i += 2
    log_sigma = theta[i]; i += 1
    log_sd0 = theta[i]; i += 1
    log_sd1 = theta[i]; i += 1
    rho_z = theta[i]; i += 1
    gamma = theta[i:i + n_gamma]; i += n_gamma
    alpha = theta[i]; i += 1
    log_h = theta[i:i + n_h]; i += n_h
    b = theta[i:].reshape(n_subj, 2)
    sd0, sd1 = np.exp(log_sd0), np.exp(log_sd1)
    rho = np.tanh(rho_z)
    D = np.array([[sd0**2, rho * sd0 * sd1], [rho * sd0 * sd1, sd1**2]])
    return {
        "beta": beta,
        "sigma": np.exp(log_sigma),
        "D": D,
        "gamma": gamma,
        "alpha": float(alpha),
        "h": np.exp(log_h),
        "b": b,
    }


def neg_joint_loglik(theta, *, y, X, Z, id_idx, Time, Death, XS, cuts, n_subj):
    """Negative joint log-likelihood / posterior (Gaussian priors on b_i)."""
    n_gamma = XS.shape[1]
    n_h = len(cuts) + 1
    p = unpack_params(theta, n_subj, n_gamma, n_h)
    beta, sigma, D, gamma, alpha, h, b = (
        p["beta"], p["sigma"], p["D"], p["gamma"], p["alpha"], p["h"], p["b"]
    )

    # longitudinal
    mu = X @ beta + np.sum(Z * b[id_idx], axis=1)
    resid = y - mu
    ll_y = -0.5 * len(y) * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(resid**2) / sigma**2

    # random-effects prior (vectorized)
    try:
        sign, logdet = np.linalg.slogdet(D)
        if sign <= 0:
            return 1e12
        Dinv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        return 1e12
    quad = np.einsum("ij,jk,ik->i", b, Dinv, b)
    ll_b = -0.5 * (n_subj * (2 * np.log(2 * np.pi) + logdet) + np.sum(quad))

    # survival (current-value association)
    A = XS @ gamma + alpha * (beta[0] + b[:, 0])
    B = alpha * (beta[1] + b[:, 1])
    ll_s = 0.0
    for i in range(n_subj):
        Ti = float(Time[i])
        Lam = cumhaz_piecewise_exp(Ti, cuts, h, float(A[i]), float(B[i]))
        if Death[i]:
            haz = hazard_at(Ti, cuts, h, float(A[i]), float(B[i]))
            if haz <= 0:
                return 1e12
            ll_s += np.log(haz) - Lam
        else:
            ll_s += -Lam

    ll_prior = -0.5 * (
        np.sum(beta**2) / 100
        + np.sum(gamma**2) / 100
        + alpha**2 / 100
        + np.sum(np.log(h) ** 2) / 10
    )
    return -(ll_y + ll_b + ll_s + ll_prior)


def fit_one_marker_jm(
    data_long: pd.DataFrame,
    data_surv: pd.DataFrame,
    marker: str = "Y1",
    obstime: str = "obstime",
    covars=("x1", "x2"),
    maxiter: int = 150,
    verbose: bool = False,
):
    long = data_long[["id", obstime, marker]].dropna().copy()
    surv = data_surv[["id", "survtime", "death", *covars]].copy()
    common = np.intersect1d(long["id"].unique(), surv["id"].unique())
    long = long[long["id"].isin(common)].sort_values(["id", obstime])
    surv = surv[surv["id"].isin(common)].sort_values("id")

    uniq, _, id_idx = subject_index_maps(long["id"].to_numpy())
    surv = surv.set_index("id").loc[uniq].reset_index()

    y = long[marker].to_numpy(float)
    X, Z = build_linear_design(long[obstime].to_numpy(float))
    Time = surv["survtime"].to_numpy(float)
    Death = surv["death"].to_numpy(int)
    XS = surv[list(covars)].to_numpy(float)
    n_subj = len(uniq)
    cuts = piecewise_cuts(Time, 4)
    n_h = len(cuts) + 1
    n_gamma = XS.shape[1]

    beta0, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta0
    sigma0 = float(max(resid.std(ddof=1), 0.1))
    b0 = np.zeros((n_subj, 2))
    ids_long = long["id"].to_numpy()
    for i, uid in enumerate(uniq):
        m = ids_long == uid
        b0[i, 0] = float(np.mean(y[m] - X[m] @ beta0))

    theta0 = pack_params(
        beta0, np.log(sigma0), np.log([0.8, 0.5]), 0.0,
        np.zeros(n_gamma), 0.0, np.log(np.full(n_h, 0.5)), b0,
    )

    def objective(theta):
        return neg_joint_loglik(
            theta, y=y, X=X, Z=Z, id_idx=id_idx, Time=Time, Death=Death,
            XS=XS, cuts=cuts, n_subj=n_subj,
        )

    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": max(20000, maxiter * 40), "ftol": 1e-6},
    )
    est = unpack_params(res.x, n_subj, n_gamma, n_h)
    # η at observed survival time for each subject (current value)
    eta_Ti = est["beta"][0] + est["b"][:, 0] + (est["beta"][1] + est["b"][:, 1]) * Time

    out = {
        "marker": marker,
        "ids": uniq,
        "beta": est["beta"],
        "sigma": est["sigma"],
        "D": est["D"],
        "gamma_stage1": est["gamma"],  # one-marker survival coeffs (nuisance for TSJM glue)
        "alpha_stage1": est["alpha"],
        "h": est["h"],
        "cuts": cuts,
        "b": est["b"],
        "Time": Time,
        "Death": Death,
        "XS": XS,
        "covars": list(covars),
        "eta_at_Ti": eta_Ti,
        "success": bool(res.success),
        "nll": float(res.fun),
        "message": res.message,
        "nit": int(res.nit) if hasattr(res, "nit") else None,
    }
    if verbose:
        print(f"[{marker}] success={out['success']} nll={out['nll']:.2f} beta={out['beta']} alpha={out['alpha_stage1']:.3f}")
    return out


def eta_path(beta, b_i, times):
    """η_i(t) for linear trend given subject random effects."""
    t = np.asarray(times, float)
    return (beta[0] + b_i[0]) + (beta[1] + b_i[1]) * t


# ---------------------------------------------------------------------------
# Stage 2: Cox with time-dependent current values
# ---------------------------------------------------------------------------

def build_td_cox_frame(stage1_fits, data_surv, data_long, obstime="obstime", covars=("x1", "x2")):
    """
    Build a start-stop dataset for CoxTimeVaryingFitter.
    Grid = each subject's observation times in long data, plus event time.
    Covariates: baseline covars + eta_k at interval start for each marker.
    """
    markers = [f["marker"] for f in stage1_fits]
    id_to_fit = []
    for f in stage1_fits:
        id_to_fit.append({int(i): k for k, i in enumerate(f["ids"])})

    rows = []
    surv = data_surv.set_index("id")
    for uid, srow in surv.iterrows():
        uid = int(uid)
        # skip if missing any stage1 fit
        if any(uid not in m for m in id_to_fit):
            continue
        T = float(srow["survtime"])
        d = int(srow["death"])
        visits = (
            data_long.loc[data_long["id"] == uid, obstime]
            .drop_duplicates()
            .sort_values()
            .to_numpy(float)
        )
        visits = visits[visits < T]
        grid = np.unique(np.concatenate([visits, [T]]))
        if len(grid) == 1:
            grid = np.array([0.0, T]) if T > 0 else np.array([0.0, 1e-6])
        if grid[0] > 0:
            grid = np.concatenate([[0.0], grid])

        for a, b in zip(grid[:-1], grid[1:]):
            if b <= a:
                continue
            etas = []
            for f, mp in zip(stage1_fits, id_to_fit):
                bi = f["b"][mp[uid]]
                etas.append(float(eta_path(f["beta"], bi, [a])[0]))
            rec = {
                "id": uid,
                "start": float(a),
                "stop": float(b),
                "event": int(d if np.isclose(b, T) else 0),
            }
            for c in covars:
                rec[c] = float(srow[c])
            for mname, e in zip(markers, etas):
                rec[f"eta_{mname}"] = e
            rows.append(rec)
    return pd.DataFrame(rows)


def fit_td_cox(td_df, covars=("x1", "x2"), markers=("Y1", "Y2", "Y3", "Y4")):
    from lifelines import CoxTimeVaryingFitter

    formula_cols = list(covars) + [f"eta_{m}" for m in markers]
    ctv = CoxTimeVaryingFitter(penalizer=0.01)
    ctv.fit(
        td_df,
        id_col="id",
        event_col="event",
        start_col="start",
        stop_col="stop",
        show_progress=False,
    )
    return ctv, formula_cols


# ---------------------------------------------------------------------------
# Dynamic prediction (Algorithm 1, first-order)
# ---------------------------------------------------------------------------

def survival_given_b(t, beta_list, b_list, alpha, gamma, omega, cuts, h0_cox_baseline_cumhaz):
    """
    Generic placeholder — for Cox Stage 2, S(t|b) uses Breslow-type cumhaz.
    Here we implement PH survival using a Breslow baseline cumulative hazard
    function handle: H0(t), with LP = γ'ω + Σ α_k η_k(t_ref) approximated by
    current-value path integrated numerically.
    """
    raise NotImplementedError


def predict_risk_first_order(
    uid,
    s,
    w,
    stage1_fits,
    cox_model,
    data_surv,
    covars=("x1", "x2"),
    n_grid: int = 40,
):
    """
    Algorithm 1 style: use b̂ from Stage 1, Stage-2 coeffs from Cox,
    approximate S(s+w)/S(s) with cumulative hazard of time-varying LP.

    Uses Breslow baseline from lifelines CoxTimeVaryingFitter if available;
    otherwise falls back to a numerical PH approximation with constant baseline 1
    (relative risks still ordered correctly for discrimination demos).
    """
    uid = int(uid)
    srow = data_surv.set_index("id").loc[uid]
    omega = srow[list(covars)].to_numpy(float)
    params = cox_model.params_

    def lp_at(t):
        eta = []
        for f in stage1_fits:
            mp = {int(i): k for k, i in enumerate(f["ids"])}
            if uid not in mp:
                return np.nan
            bi = f["b"][mp[uid]]
            eta.append(float(eta_path(f["beta"], bi, [t])[0]))
        # build covariate vector matching model columns where possible
        val = 0.0
        for c, o in zip(covars, omega):
            if c in params.index:
                val += float(params[c]) * o
        for f, e in zip(stage1_fits, eta):
            name = f"eta_{f['marker']}"
            if name in params.index:
                val += float(params[name]) * e
            elif f["marker"] in params.index:
                val += float(params[f["marker"]]) * e
        return val

    # Numerical cumulative hazard ratio using baseline from cumulative_hazard_
    # If unavailable, use exp(lp) increments (assumes H0(t)=t).
    def cumhaz(t0, t1):
        if t1 <= t0:
            return 0.0
        grid = np.linspace(t0, t1, n_grid)
        # try Breslow baseline derivative from lifelines
        try:
            base = cox_model.baseline_cumulative_hazard_
            # columns may vary; take first
            col = base.columns[0]
            H = np.interp(grid, base.index.to_numpy(float), base[col].to_numpy(float))
            dH = np.diff(H, prepend=H[0])
            dH[0] = 0.0
        except Exception:
            dH = np.diff(grid, prepend=grid[0])
            dH[0] = 0.0
        lp = np.array([lp_at(tt) for tt in grid])
        return float(np.sum(np.exp(lp) * dH))

    # survival conditional on being at risk at s:
    # S(s+w)/S(s) = exp(-(Λ(s+w)-Λ(s))) under PH with time-varying cov computed along path
    dLam = cumhaz(s, s + w)
    surv_cond = np.exp(-dLam)
    risk = 1.0 - surv_cond
    return {"id": uid, "s": s, "w": w, "risk": float(risk), "surv_cond": float(surv_cond)}


# ---------------------------------------------------------------------------
# Simple landmark AUC / Brier (ignore IPCW for pedagogy; report as naive)
# ---------------------------------------------------------------------------

def landmark_auc_brier(risks, times, events, s, w):
    """
    Among subjects with T>s:
      case: event in (s, s+w]
      control: T > s+w
    Censored inside window dropped (naive, not IPCW).
    """
    risks = np.asarray(risks, float)
    times = np.asarray(times, float)
    events = np.asarray(events, int)
    at_risk = times > s
    case = at_risk & (events == 1) & (times <= s + w)
    control = times > (s + w)
    # drop at-risk censored before s+w
    cens_in_window = at_risk & (events == 0) & (times <= s + w)
    keep = (case | control) & (~cens_in_window)
    # redefine keep properly
    keep = case | control
    r = risks[keep]
    y = case[keep].astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {"AUC": np.nan, "BS": np.nan, "n": int(keep.sum())}

    # AUC Mann-Whitney
    r1 = r[y == 1]
    r0 = r[y == 0]
    auc = np.mean([np.mean(a > r0) + 0.5 * np.mean(a == r0) for a in r1])
    bs = float(np.mean((r - y) ** 2))
    return {"AUC": float(auc), "BS": bs, "n": int(keep.sum()), "n_case": int(y.sum()), "n_ctrl": int((1 - y).sum())}
