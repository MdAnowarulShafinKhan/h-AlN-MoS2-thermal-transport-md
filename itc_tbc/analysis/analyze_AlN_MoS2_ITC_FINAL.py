#!/usr/bin/env python3
"""
FINAL publication-grade analysis for h-AlN/h-MoS2 interfacial thermal conductance.

Designed for AlN_MoS2_ITC_ITR_FINAL_CORRECTED.in (and backward-compatible with earlier output formats).

PRIMARY METHOD
--------------
Controlled two-temperature preparation followed by completely thermostat-free
NVE relaxation.  The primary conductance estimator is the energy-integral form

    dE_hot/dt = -A G DeltaT

with positive transferred energies

    Q_hot(t)  = E_hot(t_ref)  - E_hot(t)
    Q_cold(t) = E_cold(t) - E_cold(t_ref)
    Q_sym(t)  = 0.5 * (Q_hot + Q_cold)

and

    Q_sym = A G integral(DeltaT dt) + b.

The slope therefore gives G without assuming a 2D layer thickness or heat
capacity. R = 1/G.

ANALYSIS PRINCIPLES
-------------------
* The LAMMPS output is not time-averaged.
* Quantitative regression uses NON-OVERLAPPING block averages (default 0.50 ps),
  not an overlapping moving average.
* The late-time noise-dominated region is excluded automatically when DeltaT
  reaches 0.25 of its initial value, following established MoS2 AEMD practice.
* A 1 ps default fit start excludes the immediate thermostat-removal transient.
* Hot-side, cold-side, symmetric-energy, exponential-decay, NVE-energy,
  interface-gap, LJ-energy, block-size, fit-start, and fit-end sensitivities
  are all reported.
* Multi-seed statistics are calculated from independent per-run G and R values.
  ALL-RUN statistics are primary; QC-PASS summaries are secondary diagnostics.
* An ensemble-averaged energy-integral fit is also reported as a validation
  estimator; it does not replace independent-run uncertainty.
* Publication figures are written by default as 1200-dpi PNG plus vector PDF.

Dependencies: Python >=3.9, NumPy, Matplotlib. No SciPy/Pandas required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------- constants ----------------------------------
EV_TO_J = 1.602176634e-19
PS_TO_S = 1.0e-12
A2_TO_M2 = 1.0e-20
EV_PER_PS_TO_W = EV_TO_J / PS_TO_S


# ----------------------------- plot formatting ------------------------------
def set_publication_style() -> None:
    """Journal-friendly plotting defaults with safe font fallbacks."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, stem: Path, dpi: int, formats: Sequence[str]) -> List[str]:
    paths: List[str] = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().strip().lstrip(".")
        if not fmt:
            continue
        p = stem.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
            kwargs["dpi"] = dpi
        fig.savefig(p, **kwargs)
        paths.append(str(p))
    plt.close(fig)
    return paths


# ------------------------------ basic maths ---------------------------------
def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    return out


def linear_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) <= 0:
        raise ValueError("Linear fit requires >=3 points and nonzero x range.")
    m, b = np.polyfit(x, y, 1)
    pred = m * x + b
    resid = y - pred
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(x)))
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    slope_se = float(np.sqrt((ss_res / max(len(x) - 2, 1)) / sxx)) if sxx > 0 else float("nan")
    return {
        "slope": float(m),
        "intercept": float(b),
        "pred": pred,
        "resid": resid,
        "r2": r2,
        "rmse": rmse,
        "slope_se": slope_se,
    }


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = np.mean(values[idx], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_prep_file(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Read optional prep_RUNID.dat produced by the final LAMMPS input."""
    if not path.exists():
        return None
    a = np.loadtxt(path, comments="#")
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] < 7:
        return None
    out = {
        "step": a[:, 0],
        "time_ps": a[:, 1],
        "T_mos2": a[:, 2],
        "T_aln": a[:, 3],
        "E_langevin_mos2": a[:, 4],
        "E_langevin_aln": a[:, 5],
        "gap_A": a[:, 6],
        # New publication-grade LAMMPS output records the actual weighted targets.
        # Older 7-column preparation files remain fully readable.
        "T_mos_target": a[:, 7] if a.shape[1] >= 9 else np.full(len(a), np.nan),
        "T_aln_target": a[:, 8] if a.shape[1] >= 9 else np.full(len(a), np.nan),
    }
    out["time_ps"] = out["time_ps"] - out["time_ps"][0]
    return out


# ------------------------------- I/O -----------------------------------------
FINAL_COLUMNS = [
    "step", "time_ps", "T_mos2", "T_aln", "T_all",
    "E_mos2", "E_aln", "E_total", "PE_total", "KE_total", "E_LJ",
    "area_A2", "gap_A", "rms_supper_A", "rms_aln_A",
    "E_partition_error", "natoms", "T0_target", "DT0_target", "CHI",
    "LJMODE_code", "HOT_code", "seed",
]

# Current self-describing output: the original 23 columns plus actual preparation
# targets and layer atom counts. Keeping FINAL_COLUMNS separately preserves
# backward compatibility with the already-completed seed01 trajectory.
FINAL_V2_COLUMNS = FINAL_COLUMNS + [
    "T_mos_target", "T_aln_target", "N_mos2", "N_aln",
]

NEW_COLUMNS = [
    "step", "time_ps", "T_mos2", "T_aln", "T_all",
    "E_mos2", "E_aln", "E_total", "PE_total", "KE_total", "E_LJ",
    "area_A2", "gap_A", "rms_supper_A", "rms_aln_A",
    "E_partition_error", "natoms",
]

OLD_COLUMNS = [
    "step", "time_ps", "T_mos2", "T_aln", "E_mos2", "E_aln",
    "E_total", "area_A2", "gap_A", "rms_supper_A", "rms_aln_A",
]


def read_relax_file(filename: Path) -> Dict[str, np.ndarray]:
    data = np.loadtxt(filename, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] >= len(FINAL_V2_COLUMNS):
        cols = FINAL_V2_COLUMNS
        data = data[:, :len(cols)]
    elif data.shape[1] >= len(FINAL_COLUMNS):
        cols = FINAL_COLUMNS
        data = data[:, :len(cols)]
    elif data.shape[1] >= len(NEW_COLUMNS):
        cols = NEW_COLUMNS
        data = data[:, :len(cols)]
    elif data.shape[1] >= len(OLD_COLUMNS):
        cols = OLD_COLUMNS
        data = data[:, :len(cols)]
    else:
        raise ValueError(
            f"{filename}: expected at least {len(OLD_COLUMNS)} columns; found {data.shape[1]}."
        )

    out = {name: data[:, i].astype(float) for i, name in enumerate(cols)}
    # Backward-compatible missing quantities become NaN arrays.
    n = len(data)
    for name in FINAL_V2_COLUMNS:
        if name not in out:
            out[name] = np.full(n, np.nan, dtype=float)

    # Remove rows with nonfinite mandatory quantities, sort, and drop duplicate times.
    mandatory = np.isfinite(out["time_ps"]) & np.isfinite(out["T_mos2"]) & np.isfinite(out["T_aln"])
    mandatory &= np.isfinite(out["E_mos2"]) & np.isfinite(out["E_aln"]) & np.isfinite(out["E_total"])
    if np.count_nonzero(mandatory) < 20:
        raise ValueError(f"{filename}: too few finite mandatory data rows.")
    for k in out:
        out[k] = out[k][mandatory]

    order = np.argsort(out["time_ps"])
    for k in out:
        out[k] = out[k][order]
    _, unique_idx = np.unique(out["time_ps"], return_index=True)
    unique_idx = np.sort(unique_idx)
    for k in out:
        out[k] = out[k][unique_idx]

    out["time_ps"] = out["time_ps"] - out["time_ps"][0]
    if len(out["time_ps"]) < 20 or np.any(np.diff(out["time_ps"]) <= 0):
        raise ValueError(f"{filename}: time data are not strictly increasing after cleanup.")
    return out


def block_average(d: Dict[str, np.ndarray], block_ps: float) -> Dict[str, np.ndarray]:
    """Non-overlapping block average of all trajectory columns."""
    t = d["time_ps"]
    dt = float(np.median(np.diff(t)))
    if block_ps <= dt * 1.01:
        return {k: np.array(v, copy=True) for k, v in d.items()}
    nper = max(1, int(round(block_ps / dt)))
    blocks: Dict[str, List[float]] = {k: [] for k in d}
    n = len(t)
    for start in range(0, n, nper):
        stop = min(start + nper, n)
        # Drop only a very short final remainder.
        if stop - start < max(2, nper // 2):
            break
        sl = slice(start, stop)
        for k, arr in d.items():
            vals = arr[sl]
            finite = vals[np.isfinite(vals)]
            blocks[k].append(float(np.mean(finite)) if len(finite) else float("nan"))
    out = {k: np.asarray(v, dtype=float) for k, v in blocks.items()}
    out["time_ps"] = out["time_ps"] - out["time_ps"][0]
    return out


def write_dict_rows_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_processed_csv(path: Path, p: Dict[str, np.ndarray]) -> None:
    keys = [
        "time_ps", "T_mos2", "T_aln", "T_all", "T_hot", "T_cold",
        "T_layer_midpoint", "T_atom_weighted_layers", "DeltaT_K",
        "Integral_DeltaT_Kps", "Q_hot_eV", "Q_cold_eV", "Q_sym_eV",
        "E_total_eV", "E_LJ_eV", "gap_A", "fit_mask",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        n = len(p["time_ps"])
        for i in range(n):
            w.writerow([
                p["time_ps"][i], p["T_mos2"][i], p["T_aln"][i], p["T_all"][i],
                p["T_hot"][i], p["T_cold"][i], p["T_layer_midpoint"][i],
                p["T_atom_weighted_layers"][i], p["delta"][i], p["integral"][i],
                p["Qhot"][i], p["Qcold"][i], p["Qsym"][i],
                p["E_total"][i], p["E_LJ"][i], p["gap_A"][i], int(p["fit_mask"][i]),
            ])


# --------------------------- fit window/estimator ----------------------------
def determine_hot(raw: Dict[str, np.ndarray], hot_arg: str, early_ps: float = 0.5) -> str:
    if hot_arg != "auto":
        return hot_arg
    t = raw["time_ps"]
    m = t <= min(early_ps, t[-1])
    dd = float(np.median(raw["T_mos2"][m] - raw["T_aln"][m]))
    return "mos2" if dd >= 0 else "aln"


def choose_fit_window(
    t: np.ndarray,
    delta: np.ndarray,
    fit_start_ps: float,
    fit_end_ps: Optional[float],
    end_fraction: float,
    delta0_window_ps: float,
    sustain_points: int,
    min_fit_points: int,
) -> Tuple[np.ndarray, float, float, bool]:
    early = t <= min(t[-1], max(delta0_window_ps, t[1] if len(t) > 1 else 0.0))
    if not np.any(early):
        early = np.arange(len(t)) < min(5, len(t))
    delta0 = float(np.median(delta[early]))
    if not np.isfinite(delta0) or delta0 <= 0:
        raise RuntimeError(f"Initial hot-cold DeltaT is not positive: {delta0}")

    threshold = end_fraction * delta0
    threshold_reached = False
    if fit_end_ps is not None:
        end = float(min(fit_end_ps, t[-1]))
        threshold_reached = bool(np.any(delta[(t <= end) & (t > fit_start_ps)] <= threshold))
    else:
        candidates = np.where((t > fit_start_ps) & (delta <= threshold))[0]
        end = float(t[-1])
        sustain_points = max(1, int(sustain_points))
        for idx in candidates:
            j = min(idx + sustain_points, len(t))
            if j - idx >= sustain_points and np.all(delta[idx:j] <= threshold):
                end = float(t[idx])
                threshold_reached = True
                break

    mask = (t >= fit_start_ps) & (t <= end) & np.isfinite(delta) & (delta > 0)
    if np.count_nonzero(mask) < min_fit_points:
        raise RuntimeError(
            f"Only {np.count_nonzero(mask)} fit points remain. Increase production time, "
            "decrease block size, or adjust fit-window settings."
        )
    return mask, delta0, end, threshold_reached


def slope_to_G(m_eV_per_Kps: float, area_A2: float) -> float:
    area_m2 = area_A2 * A2_TO_M2
    return m_eV_per_Kps * EV_PER_PS_TO_W / area_m2


@dataclass
class Estimate:
    result: Dict[str, object]
    processed: Dict[str, np.ndarray]
    fit_sym: Dict[str, object]
    fit_hot: Dict[str, object]
    fit_cold: Dict[str, object]
    exp_fit: Dict[str, object]


def estimate_from_blocked(
    blocked: Dict[str, np.ndarray],
    hot: str,
    args,
    fit_start_override: Optional[float] = None,
) -> Estimate:
    t = blocked["time_ps"]
    if hot == "mos2":
        Th, Tc = blocked["T_mos2"], blocked["T_aln"]
        Eh, Ec = blocked["E_mos2"], blocked["E_aln"]
        hot_name, cold_name = "MoS2", "h-AlN"
    elif hot == "aln":
        Th, Tc = blocked["T_aln"], blocked["T_mos2"]
        Eh, Ec = blocked["E_aln"], blocked["E_mos2"]
        hot_name, cold_name = "h-AlN", "MoS2"
    else:
        raise ValueError("hot must be mos2 or aln")

    delta = Th - Tc
    fit_start = args.fit_start_ps if fit_start_override is None else fit_start_override
    fit_mask, delta0, fit_end, threshold_reached = choose_fit_window(
        t, delta, fit_start, args.fit_end_ps, args.end_fraction,
        args.delta0_window_ps, args.threshold_sustain_points, args.min_fit_points,
    )

    integ = cumulative_trapezoid(delta, t)
    Qhot = Eh[0] - Eh
    Qcold = Ec - Ec[0]
    Qsym = 0.5 * (Qhot + Qcold)

    area_A2 = float(np.nanmedian(blocked["area_A2"][fit_mask]))
    if not np.isfinite(area_A2) or area_A2 <= 0:
        raise RuntimeError("Invalid interface area in trajectory.")

    x = integ[fit_mask]
    fit_sym = linear_fit(x, Qsym[fit_mask])
    fit_hot = linear_fit(x, Qhot[fit_mask])
    fit_cold = linear_fit(x, Qcold[fit_mask])

    G = slope_to_G(float(fit_sym["slope"]), area_A2)
    G_se_ols = abs(slope_to_G(float(fit_sym["slope_se"]), area_A2))
    G_hot = slope_to_G(float(fit_hot["slope"]), area_A2)
    G_cold = slope_to_G(float(fit_cold["slope"]), area_A2)
    R = 1.0 / G if G > 0 else float("nan")

    # Secondary exponential diagnostic: DeltaT = A exp(-t/tau).
    tf = t[fit_mask]
    df = delta[fit_mask]
    exp_fit = linear_fit(tf, np.log(df))
    mexp = float(exp_fit["slope"])
    tau_ps = -1.0 / mexp if mexp < 0 else float("nan")
    pred_delta = np.exp(float(exp_fit["intercept"]) + mexp * tf)
    ss_res = float(np.sum((df - pred_delta) ** 2))
    ss_tot = float(np.sum((df - np.mean(df)) ** 2))
    r2_exp = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    exp_fit["pred_delta"] = pred_delta
    exp_fit["r2_delta"] = r2_exp

    # NVE energy diagnostics.
    Et = blocked["E_total"]
    Efit = Et[fit_mask]
    efit = linear_fit(t[fit_mask], Efit)
    energy_drift_eV_per_ns = float(efit["slope"]) * 1000.0
    natoms = float(np.nanmedian(blocked["natoms"])) if np.any(np.isfinite(blocked["natoms"])) else float("nan")
    KE_mean = float(np.nanmean(blocked["KE_total"][fit_mask])) if np.any(np.isfinite(blocked["KE_total"][fit_mask])) else float("nan")
    energy_drift_pct_KE_per_ns = (
        abs(energy_drift_eV_per_ns) / abs(KE_mean) * 100.0
        if np.isfinite(KE_mean) and abs(KE_mean) > 0 else float("nan")
    )
    energy_pp_meV_atom = (
        float(np.ptp(Efit)) / natoms * 1000.0
        if np.isfinite(natoms) and natoms > 0 else float("nan")
    )

    # Geometry and interface-energy diagnostics.
    gap = blocked["gap_A"]
    gap_fit = gap[fit_mask]
    gapfit = linear_fit(t[fit_mask], gap_fit)
    gap_mean = float(np.nanmean(gap_fit))
    gap_std = float(np.nanstd(gap_fit, ddof=1)) if len(gap_fit) > 1 else 0.0
    gap_drift_A_per_100ps = float(gapfit["slope"]) * 100.0

    E_LJ = blocked["E_LJ"]
    E_LJ_fit = E_LJ[fit_mask]
    E_LJ_mean = float(np.nanmean(E_LJ_fit)) if np.any(np.isfinite(E_LJ_fit)) else float("nan")
    E_LJ_std = float(np.nanstd(E_LJ_fit, ddof=1)) if np.count_nonzero(np.isfinite(E_LJ_fit)) > 1 else float("nan")

    part = blocked["E_partition_error"]
    part_max = float(np.nanmax(np.abs(part[fit_mask]))) if np.any(np.isfinite(part[fit_mask])) else float("nan")

    # Temperature diagnostics.  The arithmetic midpoint 0.5*(Th+Tc) is useful
    # descriptively, but it is NOT the correct nominal-temperature QC variable for
    # unequal finite layers.  Use LAMMPS T_all for the system-temperature check.
    Tinterface = 0.5 * (Th + Tc)
    Tmean_fit = float(np.mean(Tinterface[fit_mask]))
    Tall = blocked["T_all"]
    Tsystem_mean_fit = (
        float(np.nanmean(Tall[fit_mask]))
        if np.any(np.isfinite(Tall[fit_mask])) else Tmean_fit
    )

    Nmos_meta = float(np.nanmedian(blocked["N_mos2"])) if np.any(np.isfinite(blocked["N_mos2"])) else float("nan")
    Naln_meta = float(np.nanmedian(blocked["N_aln"])) if np.any(np.isfinite(blocked["N_aln"])) else float("nan")
    if np.isfinite(Nmos_meta) and np.isfinite(Naln_meta) and (Nmos_meta + Naln_meta) > 0:
        Tweighted_layers = (Nmos_meta * blocked["T_mos2"] + Naln_meta * blocked["T_aln"]) / (Nmos_meta + Naln_meta)
        Tweighted_mean_fit = float(np.nanmean(Tweighted_layers[fit_mask]))
    else:
        Tweighted_layers = np.full_like(Tinterface, np.nan, dtype=float)
        Tweighted_mean_fit = float("nan")

    side_disagreement = (
        abs(G_hot - G_cold) / max(abs(G), 1e-30) * 100.0
        if np.isfinite(G_hot) and np.isfinite(G_cold) else float("nan")
    )

    result: Dict[str, object] = {
        "hot_layer": hot_name,
        "cold_layer": cold_name,
        "fit_start_ps": float(fit_start),
        "fit_end_ps": float(fit_end),
        "fit_duration_ps": float(fit_end - fit_start),
        "DeltaT0_K": float(delta0),
        "DeltaT_end_K": float(np.median(delta[fit_mask][-min(3, np.count_nonzero(fit_mask)):])),
        "end_fraction_threshold_reached": bool(threshold_reached),
        # mean_interface_T_K is retained for backward-compatible output naming and
        # means the arithmetic layer-temperature midpoint, not the system average.
        "mean_interface_T_K": Tmean_fit,
        "mean_system_T_K": Tsystem_mean_fit,
        "mean_atom_weighted_layer_T_K": Tweighted_mean_fit,
        "T0_target_K": float(np.nanmedian(blocked["T0_target"])) if np.any(np.isfinite(blocked["T0_target"])) else float("nan"),
        "DT0_target_K": float(np.nanmedian(blocked["DT0_target"])) if np.any(np.isfinite(blocked["DT0_target"])) else float("nan"),
        "CHI": float(np.nanmedian(blocked["CHI"])) if np.any(np.isfinite(blocked["CHI"])) else float("nan"),
        "LJMODE": ("LB" if float(np.nanmedian(blocked["LJMODE_code"])) < 0.5 else "GEOM") if np.any(np.isfinite(blocked["LJMODE_code"])) else "UNKNOWN",
        "seed": int(round(float(np.nanmedian(blocked["seed"])))) if np.any(np.isfinite(blocked["seed"])) else -1,
        "T_mos_target_K": float(np.nanmedian(blocked["T_mos_target"])) if np.any(np.isfinite(blocked["T_mos_target"])) else float("nan"),
        "T_aln_target_K": float(np.nanmedian(blocked["T_aln_target"])) if np.any(np.isfinite(blocked["T_aln_target"])) else float("nan"),
        "N_mos2": Nmos_meta,
        "N_aln": Naln_meta,
        "area_A2": area_A2,
        "area_nm2": area_A2 / 100.0,
        "G_sym_W_m2K": G,
        "G_sym_MW_m2K": G / 1.0e6,
        "G_sym_OLS_SE_MW_m2K": G_se_ols / 1.0e6,
        "R_sym_m2K_W": R,
        "R_sym_x1e8_m2K_W": R * 1.0e8 if np.isfinite(R) else float("nan"),
        "R2_energy_sym": float(fit_sym["r2"]),
        "energy_fit_RMSE_eV": float(fit_sym["rmse"]),
        "G_hot_MW_m2K": G_hot / 1.0e6,
        "R2_hot": float(fit_hot["r2"]),
        "G_cold_MW_m2K": G_cold / 1.0e6,
        "R2_cold": float(fit_cold["r2"]),
        "side_fit_disagreement_pct": side_disagreement,
        "tau_exp_ps": tau_ps,
        "R2_exp_DeltaT": r2_exp,
        "R2_log_DeltaT": float(exp_fit["r2"]),
        "energy_drift_eV_per_ns": energy_drift_eV_per_ns,
        "energy_drift_pct_KE_per_ns": energy_drift_pct_KE_per_ns,
        "energy_peak_to_peak_meV_per_atom": energy_pp_meV_atom,
        "gap_mean_A": gap_mean,
        "gap_std_A": gap_std,
        "gap_drift_A_per_100ps": gap_drift_A_per_100ps,
        "rms_upperS_mean_A": float(np.nanmean(blocked["rms_supper_A"][fit_mask])),
        "rms_AlN_mean_A": float(np.nanmean(blocked["rms_aln_A"][fit_mask])),
        "E_LJ_mean_eV": E_LJ_mean,
        "E_LJ_std_eV": E_LJ_std,
        "partition_error_max_abs_eV": part_max,
        "natoms": natoms,
    }

    processed = {
        "time_ps": t,
        "T_mos2": blocked["T_mos2"],
        "T_aln": blocked["T_aln"],
        "T_all": Tall,
        "T_hot": Th,
        "T_cold": Tc,
        "T_layer_midpoint": Tinterface,
        "T_atom_weighted_layers": Tweighted_layers,
        "delta": delta,
        "integral": integ,
        "Qhot": Qhot,
        "Qcold": Qcold,
        "Qsym": Qsym,
        "E_total": Et,
        "E_LJ": E_LJ,
        "gap_A": gap,
        "fit_mask": fit_mask,
        "pred_delta_fit": pred_delta,
        "fit_time": tf,
    }
    return Estimate(result, processed, fit_sym, fit_hot, fit_cold, exp_fit)


# ------------------------------ sensitivities --------------------------------
def sensitivity_analysis(raw: Dict[str, np.ndarray], hot: str, args) -> List[Dict[str, object]]:
    """Run transparent per-trajectory sensitivity checks.

    The primary result is not replaced by any sensitivity result.  These rows are
    diagnostic only.  In addition to block width and fit start, the updated
    workflow explicitly tests the automatic fit-end fraction because the early
    transient can have a systematically different apparent slope.
    """
    rows: List[Dict[str, object]] = []
    for block_ps in args.sensitivity_blocks:
        try:
            blocked = block_average(raw, block_ps)
            est = estimate_from_blocked(blocked, hot, args)
            rows.append({
                "sensitivity_type": "block_ps",
                "value": block_ps,
                "G_MW_m2K": est.result["G_sym_MW_m2K"],
                "R_x1e8_m2K_W": est.result["R_sym_x1e8_m2K_W"],
                "R2": est.result["R2_energy_sym"],
                "fit_end_ps": est.result["fit_end_ps"],
            })
        except Exception as exc:
            rows.append({"sensitivity_type": "block_ps", "value": block_ps, "error": str(exc)})

    blocked_primary = block_average(raw, args.block_ps)
    for start_ps in args.sensitivity_fit_starts:
        try:
            est = estimate_from_blocked(blocked_primary, hot, args, fit_start_override=start_ps)
            rows.append({
                "sensitivity_type": "fit_start_ps",
                "value": start_ps,
                "G_MW_m2K": est.result["G_sym_MW_m2K"],
                "R_x1e8_m2K_W": est.result["R_sym_x1e8_m2K_W"],
                "R2": est.result["R2_energy_sym"],
                "fit_end_ps": est.result["fit_end_ps"],
            })
        except Exception as exc:
            rows.append({"sensitivity_type": "fit_start_ps", "value": start_ps, "error": str(exc)})

    # Fit-end-fraction sensitivity is meaningful only when an explicit --fit-end-ps
    # was not supplied.  Use a cloned argparse Namespace so the primary settings
    # remain untouched.
    if args.fit_end_ps is None:
        for end_fraction in args.sensitivity_end_fractions:
            try:
                test_args = argparse.Namespace(**vars(args))
                test_args.end_fraction = float(end_fraction)
                est = estimate_from_blocked(blocked_primary, hot, test_args)
                rows.append({
                    "sensitivity_type": "end_fraction",
                    "value": float(end_fraction),
                    "G_MW_m2K": est.result["G_sym_MW_m2K"],
                    "R_x1e8_m2K_W": est.result["R_sym_x1e8_m2K_W"],
                    "R2": est.result["R2_energy_sym"],
                    "fit_end_ps": est.result["fit_end_ps"],
                    "threshold_reached": est.result["end_fraction_threshold_reached"],
                })
            except Exception as exc:
                rows.append({"sensitivity_type": "end_fraction", "value": float(end_fraction), "error": str(exc)})
    return rows


def relative_span_pct(values: Iterable[float]) -> float:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(a) < 2 or abs(np.mean(a)) < 1e-30:
        return float("nan")
    return float((np.max(a) - np.min(a)) / abs(np.mean(a)) * 100.0)


# ------------------------------ quality control ------------------------------
def add_quality_flags(result: Dict[str, object], sensitivity: List[Dict[str, object]], args) -> None:
    flags: List[str] = []
    advisory: List[str] = []
    G = float(result["G_sym_MW_m2K"])
    if not np.isfinite(G) or G <= 0:
        flags.append("NONPOSITIVE_G")
    if float(result["R2_energy_sym"]) < args.min_r2:
        flags.append(f"ENERGY_FIT_R2<{args.min_r2:g}")
    if np.isfinite(float(result["side_fit_disagreement_pct"])) and float(result["side_fit_disagreement_pct"]) > args.max_side_disagreement_pct:
        flags.append("HOT_COLD_FITS_DISAGREE")
    if float(result["fit_duration_ps"]) < args.min_fit_duration_ps:
        flags.append("SHORT_FIT_WINDOW")
    if not bool(result.get("end_fraction_threshold_reached", False)) and args.fit_end_ps is None:
        flags.append("END_FRACTION_NOT_REACHED")
    if np.isfinite(float(result["gap_drift_A_per_100ps"])) and abs(float(result["gap_drift_A_per_100ps"])) > args.max_gap_drift_A_per_100ps:
        flags.append("GAP_DRIFT")
    if np.isfinite(float(result["energy_drift_pct_KE_per_ns"])) and float(result["energy_drift_pct_KE_per_ns"]) > args.max_energy_drift_pct_KE_per_ns:
        flags.append("NVE_ENERGY_DRIFT")
    if np.isfinite(float(result["partition_error_max_abs_eV"])) and float(result["partition_error_max_abs_eV"]) > args.max_partition_error_eV:
        flags.append("ENERGY_PARTITION_ERROR")

    target_T = float(result.get("T0_target_K", float("nan")))
    if not np.isfinite(target_T):
        target_T = args.nominal_temp
    # For unequal finite layers, compare the global kinetic temperature (T_all)
    # to T0.  Do not compare the arithmetic midpoint of the two layer temperatures.
    temp_for_qc = float(result.get("mean_system_T_K", float("nan")))
    if not np.isfinite(temp_for_qc):
        temp_for_qc = float(result.get("mean_atom_weighted_layer_T_K", float("nan")))
    if not np.isfinite(temp_for_qc):
        temp_for_qc = float(result["mean_interface_T_K"])
    temp_dev = abs(temp_for_qc - target_T)
    result["mean_system_T_deviation_K"] = temp_dev
    if temp_dev > args.max_mean_temp_deviation_K:
        advisory.append("MEAN_SYSTEM_T_OFF_NOMINAL")

    target_dT = float(result.get("DT0_target_K", float("nan")))
    if np.isfinite(target_dT) and target_dT > 0:
        dT_dev_pct = abs(float(result["DeltaT0_K"]) - target_dT) / target_dT * 100.0
        result["DeltaT0_deviation_pct"] = dT_dev_pct
        if dT_dev_pct > args.max_deltaT0_deviation_pct:
            advisory.append("INITIAL_DELTAT_OFF_TARGET")
    else:
        result["DeltaT0_deviation_pct"] = float("nan")

    block_G = [float(r["G_MW_m2K"]) for r in sensitivity if r.get("sensitivity_type") == "block_ps" and "G_MW_m2K" in r]
    start_G = [float(r["G_MW_m2K"]) for r in sensitivity if r.get("sensitivity_type") == "fit_start_ps" and "G_MW_m2K" in r]
    end_G = [float(r["G_MW_m2K"]) for r in sensitivity if r.get("sensitivity_type") == "end_fraction" and "G_MW_m2K" in r]
    block_span = relative_span_pct(block_G)
    start_span = relative_span_pct(start_G)
    end_span = relative_span_pct(end_G)
    result["block_sensitivity_G_span_pct"] = block_span
    result["fit_start_sensitivity_G_span_pct"] = start_span
    result["fit_end_sensitivity_G_span_pct"] = end_span
    if np.isfinite(block_span) and block_span > args.max_sensitivity_span_pct:
        advisory.append("BLOCK_SIZE_SENSITIVITY")
    if np.isfinite(start_span) and start_span > args.max_sensitivity_span_pct:
        advisory.append("FIT_START_SENSITIVITY")

    result["quality_flags"] = ";".join(flags) if flags else "PASS"
    result["advisory_flags"] = ";".join(advisory) if advisory else "NONE"
    result["qc_pass"] = bool(not flags)


# ------------------------------- per-run plots -------------------------------
def plot_run_figures(
    filename: Path,
    raw: Dict[str, np.ndarray],
    blocked: Dict[str, np.ndarray],
    est: Estimate,
    sensitivity: List[Dict[str, object]],
    outdir: Path,
    args,
) -> List[str]:
    p = est.processed
    r = est.result
    figures: List[str] = []
    stem = filename.stem

    # Fig S0: optional controlled preparation trace. This is generated only if
    # prep_RUNID.dat is present beside the production trajectory.
    runid = stem[len("itc_relax_"):] if stem.startswith("itc_relax_") else stem
    prep = read_prep_file(filename.parent / f"prep_{runid}.dat")
    if prep is not None:
        fig, ax = plt.subplots(1, 2, figsize=(7.08, 2.55))
        ax[0].plot(prep["time_ps"], prep["T_mos2"], label="MoS$_2$")
        ax[0].plot(prep["time_ps"], prep["T_aln"], label="h-AlN")
        # Plot the ACTUAL weighted preparation targets when available.  Do not
        # assume T0 +/- DeltaT/2 for unequal finite layers.
        if np.any(np.isfinite(prep["T_mos_target"])):
            ax[0].axhline(float(np.nanmedian(prep["T_mos_target"])), linestyle=":", lw=0.8, label="MoS$_2$ target")
        if np.any(np.isfinite(prep["T_aln_target"])):
            ax[0].axhline(float(np.nanmedian(prep["T_aln_target"])), linestyle=":", lw=0.8, label="h-AlN target")
        ax[0].set_xlabel("Preparation time (ps)")
        ax[0].set_ylabel("Temperature (K)")
        ax[0].set_title("(a) Controlled two-temperature preparation")
        ax[0].legend(loc="best")
        ax[1].plot(prep["time_ps"], prep["gap_A"])
        ax[1].set_xlabel("Preparation time (ps)")
        ax[1].set_ylabel("Mean interface gap (Å)")
        ax[1].set_title("(b) Interface-spacing stability")
        fig.tight_layout()
        figures += save_figure(fig, outdir / f"{stem}_FigS0_preparation", args.dpi, args.formats)

    # Fig 1: temperatures and DeltaT.
    fig, ax = plt.subplots(1, 2, figsize=(7.08, 2.75))
    l1, = ax[0].plot(raw["time_ps"], raw["T_mos2"], lw=0.45, alpha=0.22, label="MoS$_2$ raw")
    l2, = ax[0].plot(raw["time_ps"], raw["T_aln"], lw=0.45, alpha=0.22, label="h-AlN raw")
    ax[0].plot(blocked["time_ps"], blocked["T_mos2"], lw=1.1, color=l1.get_color(), label=f"MoS$_2$ {args.block_ps:g} ps blocks")
    ax[0].plot(blocked["time_ps"], blocked["T_aln"], lw=1.1, color=l2.get_color(), label=f"h-AlN {args.block_ps:g} ps blocks")
    ax[0].axvspan(float(r["fit_start_ps"]), float(r["fit_end_ps"]), alpha=0.10)
    ax[0].set_xlabel("Time (ps)")
    ax[0].set_ylabel("Temperature (K)")
    ax[0].set_title("(a) Layer temperatures")
    ax[0].legend(ncol=2, loc="best")

    ax[1].plot(p["time_ps"], p["delta"], label=r"$\Delta T$")
    tf = p["fit_time"]
    pred = np.asarray(est.exp_fit["pred_delta"], dtype=float)
    ax[1].plot(tf, pred, linestyle="--", label=rf"exponential diagnostic, $\tau$={float(r['tau_exp_ps']):.2f} ps")
    ax[1].axhline(args.end_fraction * float(r["DeltaT0_K"]), linestyle=":", lw=0.9, label=rf"{args.end_fraction:g}$\Delta T_0$")
    ax[1].axvspan(float(r["fit_start_ps"]), float(r["fit_end_ps"]), alpha=0.10)
    ax[1].set_xlabel("Time (ps)")
    ax[1].set_ylabel(r"$\Delta T=T_{hot}-T_{cold}$ (K)")
    ax[1].set_title(f"(b) Temperature difference, $R^2$={float(r['R2_exp_DeltaT']):.4f}")
    ax[1].legend(loc="best")
    fig.tight_layout()
    figures += save_figure(fig, outdir / f"{stem}_Fig1_thermal_relaxation", args.dpi, args.formats)

    # Fig 2: primary energy-integral fit plus residuals (linearity diagnostic).
    fig, ax = plt.subplots(2, 1, figsize=(3.54, 4.45), gridspec_kw={"height_ratios": [3.0, 1.0]}, sharex=True)
    ax[0].plot(p["integral"], p["Qhot"], lw=0.65, alpha=0.55, label="$Q_{hot}$")
    ax[0].plot(p["integral"], p["Qcold"], lw=0.65, alpha=0.55, label="$Q_{cold}$")
    ax[0].plot(p["integral"], p["Qsym"], lw=1.15, label="$Q_{sym}$")
    mask = p["fit_mask"]
    xfit = p["integral"][mask]
    yfit = p["Qsym"][mask]
    xx = np.linspace(float(np.min(xfit)), float(np.max(xfit)), 300)
    yy = float(est.fit_sym["slope"]) * xx + float(est.fit_sym["intercept"])
    ax[0].plot(xx, yy, linestyle="--", label=rf"fit, $R^2$={float(r['R2_energy_sym']):.5f}")
    ax[0].set_ylabel("Transferred energy (eV)")
    ax[0].set_title(rf"$G$={float(r['G_sym_MW_m2K']):.3f} MW m$^{{-2}}$ K$^{{-1}}$")
    ax[0].legend(loc="best")
    resid = yfit - (float(est.fit_sym["slope"])*xfit + float(est.fit_sym["intercept"]))
    ax[1].plot(xfit, resid, marker="o", markersize=2.0, linestyle="none")
    ax[1].axhline(0.0, linewidth=0.8)
    ax[1].set_xlabel(r"$\int \Delta T\,dt$ (K ps)")
    ax[1].set_ylabel("Residual (eV)")
    fig.tight_layout()
    figures += save_figure(fig, outdir / f"{stem}_Fig2_energy_integral", args.dpi, args.formats)

    # Fig S1: NVE and structure stability.
    fig, ax = plt.subplots(1, 3, figsize=(7.08, 2.45))
    nat = float(r["natoms"])
    if np.isfinite(nat) and nat > 0:
        dE = (p["E_total"] - p["E_total"][0]) / nat * 1000.0
        ylabel = r"$[E_{tot}-E_{tot}(0)]/N$ (meV atom$^{-1}$)"
    else:
        dE = p["E_total"] - p["E_total"][0]
        ylabel = r"$E_{tot}-E_{tot}(0)$ (eV)"
    ax[0].plot(p["time_ps"], dE)
    ax[0].axvspan(float(r["fit_start_ps"]), float(r["fit_end_ps"]), alpha=0.10)
    ax[0].set_xlabel("Time (ps)")
    ax[0].set_ylabel(ylabel)
    ax[0].set_title("(a) NVE energy")

    ax[1].plot(p["time_ps"], p["gap_A"])
    ax[1].axvspan(float(r["fit_start_ps"]), float(r["fit_end_ps"]), alpha=0.10)
    ax[1].set_xlabel("Time (ps)")
    ax[1].set_ylabel("Mean interface gap (Å)")
    ax[1].set_title("(b) Interface spacing")

    if np.any(np.isfinite(p["E_LJ"])):
        ax[2].plot(p["time_ps"], p["E_LJ"] - p["E_LJ"][0])
        ax[2].set_ylabel(r"$E_{LJ}-E_{LJ}(0)$ (eV)")
    else:
        ax[2].text(0.5, 0.5, "LJ-energy column unavailable", ha="center", va="center", transform=ax[2].transAxes)
        ax[2].set_ylabel("Interfacial LJ energy")
    ax[2].axvspan(float(r["fit_start_ps"]), float(r["fit_end_ps"]), alpha=0.10)
    ax[2].set_xlabel("Time (ps)")
    ax[2].set_title("(c) Interfacial LJ energy")
    fig.tight_layout()
    figures += save_figure(fig, outdir / f"{stem}_FigS1_stability", args.dpi, args.formats)

    # Fig S2: block-size, fit-start, and fit-end sensitivity.
    fig, ax = plt.subplots(1, 3, figsize=(7.08, 2.55))
    rb = [x for x in sensitivity if x.get("sensitivity_type") == "block_ps" and "G_MW_m2K" in x]
    rs = [x for x in sensitivity if x.get("sensitivity_type") == "fit_start_ps" and "G_MW_m2K" in x]
    re_ = [x for x in sensitivity if x.get("sensitivity_type") == "end_fraction" and "G_MW_m2K" in x]
    if rb:
        ax[0].plot([float(x["value"]) for x in rb], [float(x["G_MW_m2K"]) for x in rb], marker="o")
    ax[0].axhline(float(r["G_sym_MW_m2K"]), linestyle=":", lw=0.9)
    ax[0].set_xlabel("Non-overlapping block width (ps)")
    ax[0].set_ylabel(r"$G$ (MW m$^{-2}$ K$^{-1}$)")
    ax[0].set_title("(a) Block-size sensitivity")
    if rs:
        ax[1].plot([float(x["value"]) for x in rs], [float(x["G_MW_m2K"]) for x in rs], marker="o")
    ax[1].axhline(float(r["G_sym_MW_m2K"]), linestyle=":", lw=0.9)
    ax[1].set_xlabel("Fit start (ps)")
    ax[1].set_ylabel(r"$G$ (MW m$^{-2}$ K$^{-1}$)")
    ax[1].set_title("(b) Fit-start sensitivity")
    if re_:
        re_ = sorted(re_, key=lambda x: float(x["value"]))
        ax[2].plot([float(x["value"]) for x in re_], [float(x["G_MW_m2K"]) for x in re_], marker="o")
        ax[2].axvline(float(args.end_fraction), linestyle=":", lw=0.8)
    else:
        ax[2].text(0.5, 0.5, "manual fit end\n(no fraction scan)", ha="center", va="center", transform=ax[2].transAxes)
    ax[2].axhline(float(r["G_sym_MW_m2K"]), linestyle=":", lw=0.9)
    ax[2].set_xlabel(r"Fit end ($\Delta T/\Delta T_0$)")
    ax[2].set_ylabel(r"$G$ (MW m$^{-2}$ K$^{-1}$)")
    ax[2].set_title("(c) Fit-end sensitivity")
    fig.tight_layout()
    figures += save_figure(fig, outdir / f"{stem}_FigS2_fit_sensitivity", args.dpi, args.formats)
    return figures


# ------------------------------ structure audit ------------------------------
def parse_lammps_data(path: Path) -> Dict[str, object]:
    """Audit the supplied orthogonal atomic-style 14-type LAMMPS data file."""
    lines = path.read_text(errors="replace").splitlines()
    xlo = xhi = ylo = yhi = zlo = zhi = None
    natoms_header = None
    for line in lines[:100]:
        s = line.strip()
        m = re.match(r"^(\d+)\s+atoms\b", s)
        if m:
            natoms_header = int(m.group(1))
        if "xlo xhi" in line:
            xlo, xhi = map(float, line.split()[:2])
        elif "ylo yhi" in line:
            ylo, yhi = map(float, line.split()[:2])
        elif "zlo zhi" in line:
            zlo, zhi = map(float, line.split()[:2])
    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError(f"Could not parse orthogonal box bounds from {path}")

    try:
        start = next(i for i, s in enumerate(lines) if s.strip().startswith("Atoms")) + 1
    except StopIteration as exc:
        raise ValueError(f"No Atoms section found in {path}") from exc

    rows = []
    started = False
    for line in lines[start:]:
        s0 = line.strip()
        if not s0:
            if started:
                continue
            continue
        # Stop at a new alphabetic section header once atom rows have started.
        if started and re.match(r"^[A-Za-z]", s0):
            break
        s = s0.split()
        if len(s) < 5:
            continue
        try:
            rows.append((int(s[0]), int(s[1]), float(s[2]), float(s[3]), float(s[4])))
            started = True
        except ValueError:
            if started and re.match(r"^[A-Za-z]", s0):
                break
            continue
    if not rows:
        raise ValueError(f"No atomic coordinates parsed from {path}")

    a = np.array(rows, dtype=float)
    typ = a[:, 1].astype(int)
    xyz = a[:, 2:5]
    Lx, Ly, Lz = xhi-xlo, yhi-ylo, zhi-zlo

    def min_periodic_distance(mask1, mask2):
        q1, q2 = xyz[mask1], xyz[mask2]
        best = np.inf
        for i in range(0, len(q1), 64):
            r1 = q1[i:i+64]
            dd = q2[None, :, :] - r1[:, None, :]
            dd[:, :, 0] -= np.rint(dd[:, :, 0] / Lx) * Lx
            dd[:, :, 1] -= np.rint(dd[:, :, 1] / Ly) * Ly
            rr2 = np.sum(dd*dd, axis=2)
            best = min(best, float(np.sqrt(np.min(rr2))))
        return best

    def nearest_inplane(mask):
        q = xyz[mask]
        r = q[0]
        dd = q-r
        dd[:, 0] -= np.rint(dd[:, 0]/Lx)*Lx
        dd[:, 1] -= np.rint(dd[:, 1]/Ly)*Ly
        rr = np.hypot(dd[:, 0], dd[:, 1])
        rr = rr[rr > 1e-7]
        return float(np.min(rr))

    upper = np.isin(typ, [1, 4, 7, 10])
    lower = np.isin(typ, [3, 6, 9, 12])
    mo = np.isin(typ, [2, 5, 8, 11])
    aln = np.isin(typ, [13, 14])
    al = typ == 13

    z_upper = float(np.mean(xyz[upper, 2]))
    z_mo = float(np.mean(xyz[mo, 2]))
    z_lower = float(np.mean(xyz[lower, 2]))
    z_aln = float(np.mean(xyz[aln, 2]))
    a_mos = nearest_inplane(mo)
    a_aln = nearest_inplane(al)

    return {
        "file": str(path),
        "atoms_header": natoms_header,
        "atoms_parsed": len(rows),
        "Lx_A": Lx,
        "Ly_A": Ly,
        "Lz_A": Lz,
        "area_A2": Lx*Ly,
        "area_nm2": Lx*Ly/100.0,
        "MoS2_lowerS_z_A": z_lower,
        "MoS2_Mo_z_A": z_mo,
        "MoS2_upperS_z_A": z_upper,
        "AlN_plane_z_A": z_aln,
        "surface_gap_A": z_aln-z_upper,
        "MoS2_S_to_S_thickness_A": z_upper-z_lower,
        "Mo_plane_to_AlN_plane_A": z_aln-z_mo,
        "a_MoS2_A": a_mos,
        "a_AlN_A": a_aln,
        "a_ratio_AlN_to_MoS2": a_aln/a_mos,
        "nearest_AlN_to_upperS_A": min_periodic_distance(aln, upper),
    }


# ------------------------------ run analysis ---------------------------------
def analyze_one(filename: Path, args, outdir: Path) -> Tuple[Dict[str, object], Dict[str, np.ndarray], List[Dict[str, object]]]:
    raw = read_relax_file(filename)
    hot = determine_hot(raw, args.hot)
    blocked = block_average(raw, args.block_ps)
    est = estimate_from_blocked(blocked, hot, args)
    sensitivity = sensitivity_analysis(raw, hot, args)

    r = est.result
    r.update({
        "file": str(filename),
        "run_name": filename.stem,
        "n_raw_points": len(raw["time_ps"]),
        "n_block_points": len(blocked["time_ps"]),
        "raw_output_dt_ps": float(np.median(np.diff(raw["time_ps"]))),
        "block_ps": args.block_ps,
        "nominal_temperature_K": args.nominal_temp,
        "end_fraction": args.end_fraction,
    })
    add_quality_flags(r, sensitivity, args)

    run_dir = outdir / filename.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    write_processed_csv(run_dir / f"{filename.stem}_processed_blocks.csv", est.processed)
    write_dict_rows_csv(run_dir / f"{filename.stem}_sensitivity.csv", sensitivity)
    figs = plot_run_figures(filename, raw, blocked, est, sensitivity, run_dir, args)
    r["figure_files"] = "|".join(figs)
    return r, est.processed, sensitivity


# ----------------------------- ensemble analysis -----------------------------
def interpolate_ensemble(processed_runs: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a common grid plus individual and mean-SEM DeltaT trajectories."""
    if not processed_runs:
        return np.array([]), np.array([[]]), np.array([])
    dt = max(float(np.median(np.diff(p["time_ps"]))) for p in processed_runs)
    tmax = min(float(p["time_ps"][-1]) for p in processed_runs)
    if tmax <= 0 or dt <= 0:
        return np.array([]), np.array([[]]), np.array([])
    grid = np.arange(0.0, tmax + 0.5*dt, dt)
    arr = np.vstack([np.interp(grid, p["time_ps"], p["delta"]) for p in processed_runs])
    sem = np.std(arr, axis=0, ddof=1)/math.sqrt(len(arr)) if len(arr) > 1 else np.full(len(grid), np.nan)
    return grid, arr, sem


def _lag1_autocorr(values: np.ndarray) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 4 or np.std(a[:-1]) <= 0 or np.std(a[1:]) <= 0:
        return float("nan")
    return float(np.corrcoef(a[:-1], a[1:])[0, 1])


def _ensemble_condition_issues(rows: Sequence[Dict[str, object]]) -> List[str]:
    """Detect mixed physical conditions before averaging trajectories.

    The ensemble-average energy-integral fit is meaningful only for independent
    realizations of the same physical condition.  Per-run outputs are still
    retained if mixed inputs are supplied; only this ensemble validation fit is
    skipped.
    """
    if not rows:
        return ["no runs"]
    issues: List[str] = []
    directions = {f"{r.get('hot_layer','?')}->{r.get('cold_layer','?')}" for r in rows}
    if len(directions) > 1:
        issues.append("mixed heat-flow directions")
    models = {str(r.get("LJMODE", "UNKNOWN")) for r in rows}
    if len(models) > 1:
        issues.append("mixed LJMODE settings")

    def finite_unique(key: str, ndigits: int = 8) -> set:
        vals = []
        for r in rows:
            try:
                v = float(r.get(key, float("nan")))
            except Exception:
                continue
            if np.isfinite(v):
                vals.append(round(v, ndigits))
        return set(vals)

    checks = [
        ("area_A2", "interface areas", 6),
        ("T0_target_K", "target temperatures", 6),
        ("DT0_target_K", "target DeltaT values", 6),
        ("CHI", "CHI values", 8),
    ]
    for key, label, ndigits in checks:
        if len(finite_unique(key, ndigits)) > 1:
            issues.append(f"mixed {label}")
    return issues


def ensemble_energy_integral_fit(
    rows: Sequence[Dict[str, object]],
    processed_runs: Sequence[Dict[str, np.ndarray]],
    args,
    label: str,
    end_fraction_override: Optional[float] = None,
) -> Tuple[Dict[str, object], Optional[Dict[str, np.ndarray]]]:
    """Fit the energy-integral relation after ensemble averaging.

    This is a VALIDATION estimator: it reveals the underlying mean relaxation and
    suppresses coherent finite-trajectory fluctuations.  Statistical uncertainty
    must still come from the distribution of independent per-run G/R estimates;
    the high R^2 of an ensemble-averaged curve must not be treated as an
    uncertainty estimate.
    """
    summary: Dict[str, object] = {
        "subset": label,
        "n_runs": len(rows),
        "role": "validation_only_independent_run_statistics_remain_primary",
    }
    if not rows or not processed_runs or len(rows) != len(processed_runs):
        summary["status"] = "SKIPPED"
        summary["compatibility_issues"] = "missing or mismatched run data"
        return summary, None

    issues = _ensemble_condition_issues(rows)
    if issues:
        summary["status"] = "SKIPPED_MIXED_CONDITIONS"
        summary["compatibility_issues"] = ";".join(issues)
        return summary, None

    dt = max(float(np.median(np.diff(p["time_ps"]))) for p in processed_runs)
    tmax = min(float(p["time_ps"][-1]) for p in processed_runs)
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(tmax) or tmax <= 0:
        summary["status"] = "SKIPPED_INVALID_TIME_GRID"
        return summary, None
    grid = np.arange(0.0, tmax + 0.5*dt, dt)

    def stack(key: str) -> np.ndarray:
        return np.vstack([np.interp(grid, p["time_ps"], p[key]) for p in processed_runs])

    delta_stack = stack("delta")
    qhot_stack = stack("Qhot")
    qcold_stack = stack("Qcold")
    qsym_stack = stack("Qsym")
    tall_stack = stack("T_all")

    delta_mean = np.mean(delta_stack, axis=0)
    qhot_mean = np.mean(qhot_stack, axis=0)
    qcold_mean = np.mean(qcold_stack, axis=0)
    qsym_mean = np.mean(qsym_stack, axis=0)
    tall_mean = np.nanmean(tall_stack, axis=0)
    delta_sem = np.std(delta_stack, axis=0, ddof=1)/math.sqrt(len(rows)) if len(rows) > 1 else np.full(len(grid), np.nan)
    qsym_sem = np.std(qsym_stack, axis=0, ddof=1)/math.sqrt(len(rows)) if len(rows) > 1 else np.full(len(grid), np.nan)

    end_fraction = args.end_fraction if end_fraction_override is None else float(end_fraction_override)
    fit_mask, delta0, fit_end, threshold_reached = choose_fit_window(
        grid, delta_mean, args.fit_start_ps, args.fit_end_ps, end_fraction,
        args.delta0_window_ps, args.threshold_sustain_points, args.min_fit_points,
    )
    integral = cumulative_trapezoid(delta_mean, grid)
    x = integral[fit_mask]
    fit_sym = linear_fit(x, qsym_mean[fit_mask])
    fit_hot = linear_fit(x, qhot_mean[fit_mask])
    fit_cold = linear_fit(x, qcold_mean[fit_mask])

    area_A2 = float(np.nanmedian([float(r["area_A2"]) for r in rows]))
    G = slope_to_G(float(fit_sym["slope"]), area_A2)
    G_hot = slope_to_G(float(fit_hot["slope"]), area_A2)
    G_cold = slope_to_G(float(fit_cold["slope"]), area_A2)
    R = 1.0/G if G > 0 else float("nan")
    side_disagreement = abs(G_hot-G_cold)/max(abs(G), 1e-30)*100.0
    resid_lag1 = _lag1_autocorr(np.asarray(fit_sym["resid"], dtype=float))

    fit_pred_full = np.full(len(grid), np.nan, dtype=float)
    fit_pred_full[fit_mask] = float(fit_sym["slope"])*integral[fit_mask] + float(fit_sym["intercept"])

    summary.update({
        "status": "OK",
        "compatibility_issues": "NONE",
        "fit_start_ps": float(args.fit_start_ps),
        "fit_end_ps": float(fit_end),
        "fit_duration_ps": float(fit_end-args.fit_start_ps),
        "end_fraction": float(end_fraction),
        "end_fraction_threshold_reached": bool(threshold_reached),
        "DeltaT0_K": float(delta0),
        "mean_system_T_K": float(np.nanmean(tall_mean[fit_mask])),
        "area_A2": area_A2,
        "G_sym_W_m2K": G,
        "G_sym_MW_m2K": G/1.0e6,
        "R_sym_m2K_W": R,
        "R_sym_x1e8_m2K_W": R*1.0e8 if np.isfinite(R) else float("nan"),
        "R2_energy_sym": float(fit_sym["r2"]),
        "energy_fit_RMSE_eV": float(fit_sym["rmse"]),
        "energy_fit_residual_lag1": resid_lag1,
        "G_hot_MW_m2K": G_hot/1.0e6,
        "G_cold_MW_m2K": G_cold/1.0e6,
        "side_fit_disagreement_pct": side_disagreement,
    })

    series = {
        "time_ps": grid,
        "delta_mean": delta_mean,
        "delta_sem": delta_sem,
        "integral": integral,
        "Qhot_mean": qhot_mean,
        "Qcold_mean": qcold_mean,
        "Qsym_mean": qsym_mean,
        "Qsym_sem": qsym_sem,
        "T_all_mean": tall_mean,
        "fit_mask": fit_mask,
        "fit_pred": fit_pred_full,
    }
    return summary, series


def ensemble_end_fraction_sensitivity(
    rows: Sequence[Dict[str, object]],
    processed_runs: Sequence[Dict[str, np.ndarray]],
    args,
    label: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if args.fit_end_ps is not None:
        return out
    for fraction in args.sensitivity_end_fractions:
        try:
            summary, _ = ensemble_energy_integral_fit(rows, processed_runs, args, label, end_fraction_override=float(fraction))
            row = dict(summary)
            row["sensitivity_type"] = "ensemble_end_fraction"
            row["value"] = float(fraction)
            out.append(row)
        except Exception as exc:
            out.append({
                "subset": label,
                "sensitivity_type": "ensemble_end_fraction",
                "value": float(fraction),
                "status": "ERROR",
                "error": str(exc),
            })
    return out


def write_ensemble_series_csv(path: Path, series: Optional[Dict[str, np.ndarray]]) -> None:
    if series is None:
        return
    keys = [
        "time_ps", "DeltaT_mean_K", "DeltaT_SEM_K", "Integral_DeltaT_Kps",
        "Q_hot_mean_eV", "Q_cold_mean_eV", "Q_sym_mean_eV", "Q_sym_SEM_eV",
        "T_all_mean_K", "fit_mask", "Q_sym_fit_eV",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        n = len(series["time_ps"])
        for i in range(n):
            w.writerow([
                series["time_ps"][i], series["delta_mean"][i], series["delta_sem"][i],
                series["integral"][i], series["Qhot_mean"][i], series["Qcold_mean"][i],
                series["Qsym_mean"][i], series["Qsym_sem"][i], series["T_all_mean"][i],
                int(series["fit_mask"][i]), series["fit_pred"][i],
            ])


def ensemble_summary(rows: Sequence[Dict[str, object]], label: str, args) -> Dict[str, object]:
    """Primary statistical summary from independent trajectory estimates."""
    G = np.array([float(r["G_sym_MW_m2K"]) for r in rows if np.isfinite(float(r["G_sym_MW_m2K"])) and float(r["G_sym_MW_m2K"]) > 0])
    R = np.array([float(r["R_sym_m2K_W"]) for r in rows if np.isfinite(float(r["R_sym_m2K_W"])) and float(r["R_sym_m2K_W"]) > 0])
    out: Dict[str, object] = {
        "subset": label,
        "n_runs": len(G),
        "primary_statistical_estimator": "mean_of_independent_run_fits",
        "ensemble_average_fit_role": "validation_only",
    }
    if len(G):
        g_lo, g_hi = bootstrap_mean_ci(G, args.bootstrap_samples, args.bootstrap_seed)
        r_lo, r_hi = bootstrap_mean_ci(R, args.bootstrap_samples, args.bootstrap_seed + 1)
        directions = sorted({f"{r.get('hot_layer','?')}->{r.get('cold_layer','?')}" for r in rows})
        models = sorted({str(r.get("LJMODE", "UNKNOWN")) for r in rows})
        chis = np.array([float(r.get("CHI", float("nan"))) for r in rows], dtype=float)
        ttargets = np.array([float(r.get("T0_target_K", float("nan"))) for r in rows], dtype=float)
        dttargets = np.array([float(r.get("DT0_target_K", float("nan"))) for r in rows], dtype=float)
        actual_dt0 = np.array([float(r.get("DeltaT0_K", float("nan"))) for r in rows], dtype=float)
        out.update({
            "direction": ";".join(directions),
            "LJMODE": ";".join(models),
            "CHI": float(np.nanmedian(chis)) if np.any(np.isfinite(chis)) else float("nan"),
            "T0_target_K": float(np.nanmedian(ttargets)) if np.any(np.isfinite(ttargets)) else float("nan"),
            "DT0_target_K": float(np.nanmedian(dttargets)) if np.any(np.isfinite(dttargets)) else float("nan"),
            "DeltaT0_mean_K": float(np.nanmean(actual_dt0)),
            "DeltaT0_SD_K": float(np.nanstd(actual_dt0, ddof=1)) if len(actual_dt0) > 1 else float("nan"),
            "G_mean_MW_m2K": float(np.mean(G)),
            "G_SD_MW_m2K": float(np.std(G, ddof=1)) if len(G) > 1 else float("nan"),
            "G_SEM_MW_m2K": float(np.std(G, ddof=1)/math.sqrt(len(G))) if len(G) > 1 else float("nan"),
            "G_bootstrap95_low_MW_m2K": g_lo,
            "G_bootstrap95_high_MW_m2K": g_hi,
            "G_CV_pct": float(np.std(G, ddof=1)/np.mean(G)*100.0) if len(G) > 1 else float("nan"),
            "R_mean_m2K_W": float(np.mean(R)),
            "R_SD_m2K_W": float(np.std(R, ddof=1)) if len(R) > 1 else float("nan"),
            "R_SEM_m2K_W": float(np.std(R, ddof=1)/math.sqrt(len(R))) if len(R) > 1 else float("nan"),
            "R_bootstrap95_low_m2K_W": r_lo,
            "R_bootstrap95_high_m2K_W": r_hi,
            "mean_interface_T_K": float(np.mean([float(r["mean_interface_T_K"]) for r in rows])),
            "mean_system_T_K": float(np.mean([float(r["mean_system_T_K"]) for r in rows])),
            "mean_energy_R2": float(np.mean([float(r["R2_energy_sym"]) for r in rows])),
        })
    return out


def merge_validation_fit(summary: Dict[str, object], validation: Dict[str, object]) -> None:
    """Copy the key ensemble-validation metrics into the main summary CSV."""
    summary["ensemble_fit_status"] = validation.get("status", "MISSING")
    if validation.get("status") != "OK":
        summary["ensemble_fit_compatibility_issues"] = validation.get("compatibility_issues", "")
        return
    mapping = {
        "G_sym_MW_m2K": "ensemble_fit_G_MW_m2K",
        "R_sym_m2K_W": "ensemble_fit_R_m2K_W",
        "R_sym_x1e8_m2K_W": "ensemble_fit_R_x1e8_m2K_W",
        "R2_energy_sym": "ensemble_fit_R2",
        "energy_fit_residual_lag1": "ensemble_fit_residual_lag1",
        "fit_end_ps": "ensemble_fit_end_ps",
        "DeltaT0_K": "ensemble_fit_DeltaT0_K",
        "side_fit_disagreement_pct": "ensemble_fit_hot_cold_disagreement_pct",
    }
    for src, dst in mapping.items():
        summary[dst] = validation.get(src, float("nan"))


def plot_ensemble(
    results: List[Dict[str, object]],
    processed_runs: List[Dict[str, np.ndarray]],
    outdir: Path,
    args,
    ensemble_fit: Optional[Dict[str, object]] = None,
    ensemble_series: Optional[Dict[str, np.ndarray]] = None,
    ensemble_end_sensitivity: Optional[List[Dict[str, object]]] = None,
) -> List[str]:
    figs: List[str] = []
    if not results:
        return figs
    grid, arr, sem = interpolate_ensemble(processed_runs)

    # Ensemble DeltaT.
    if len(grid):
        fig, ax = plt.subplots(figsize=(3.54, 3.0))
        for a in arr:
            ax.plot(grid, a, lw=0.55, alpha=0.22)
        mean = np.mean(arr, axis=0)
        ax.plot(grid, mean, lw=1.25, label="ensemble mean")
        if len(arr) > 1:
            ax.fill_between(grid, mean-sem, mean+sem, alpha=0.20, linewidth=0, label="SEM")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel(r"$\Delta T$ (K)")
        ax.set_title("Independent thermal-relaxation trajectories")
        ax.legend(loc="best")
        fig.tight_layout()
        figs += save_figure(fig, outdir / "Fig3_ensemble_DeltaT", args.dpi, args.formats)

    # G and R per independent seed.  ALL runs remain visible; QC is diagnostic.
    x = np.arange(1, len(results)+1)
    G = np.array([float(r["G_sym_MW_m2K"]) for r in results])
    R8 = np.array([float(r["R_sym_x1e8_m2K_W"]) for r in results])
    qc = np.array([bool(r["qc_pass"]) for r in results])
    fig, ax = plt.subplots(1, 2, figsize=(7.08, 2.8))
    ax[0].plot(x, G, marker="o", linestyle="none", label="individual runs")
    if len(G) > 1:
        gm, gs = np.nanmean(G), np.nanstd(G, ddof=1)
        ax[0].axhline(gm, lw=1.1, label="all-run mean")
        ax[0].fill_between([0.5, len(G)+0.5], gm-gs, gm+gs, alpha=0.16, label="mean ± SD")
    for xi, yi, ok in zip(x, G, qc):
        if not ok:
            ax[0].annotate("QC", (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)
    ax[0].set_xlim(0.5, len(G)+0.5)
    ax[0].set_xticks(x)
    ax[0].set_xlabel("Independent run")
    ax[0].set_ylabel(r"$G$ (MW m$^{-2}$ K$^{-1}$)")
    ax[0].set_title("(a) Interfacial thermal conductance")
    ax[0].legend(loc="best")

    ax[1].plot(x, R8, marker="o", linestyle="none", label="individual runs")
    if len(R8) > 1:
        rm, rs = np.nanmean(R8), np.nanstd(R8, ddof=1)
        ax[1].axhline(rm, lw=1.1, label="all-run mean")
        ax[1].fill_between([0.5, len(R8)+0.5], rm-rs, rm+rs, alpha=0.16, label="mean ± SD")
    ax[1].set_xlim(0.5, len(R8)+0.5)
    ax[1].set_xticks(x)
    ax[1].set_xlabel("Independent run")
    ax[1].set_ylabel(r"$R$ ($10^-8$ m$^2$ K W$^-1$)")
    ax[1].set_title("(b) Interfacial thermal resistance")
    ax[1].legend(loc="best")
    fig.tight_layout()
    figs += save_figure(fig, outdir / "Fig4_ensemble_G_R", args.dpi, args.formats)

    # QC/SI ensemble figure.
    fig, ax = plt.subplots(2, 2, figsize=(7.08, 4.6))
    ax[0, 0].plot(x, [float(r["R2_energy_sym"]) for r in results], marker="o")
    ax[0, 0].axhline(args.min_r2, linestyle=":", lw=0.9)
    ax[0, 0].set_xlabel("Independent run")
    ax[0, 0].set_ylabel(r"Energy-fit $R^2$")
    ax[0, 0].set_title("(a) Fit quality")

    ax[0, 1].plot(x, [float(r["side_fit_disagreement_pct"]) for r in results], marker="o")
    ax[0, 1].axhline(args.max_side_disagreement_pct, linestyle=":", lw=0.9)
    ax[0, 1].set_xlabel("Independent run")
    ax[0, 1].set_ylabel("Hot/cold disagreement (%)")
    ax[0, 1].set_title("(b) Energy-side agreement")

    ax[1, 0].plot(x, [float(r["energy_drift_pct_KE_per_ns"]) for r in results], marker="o")
    ax[1, 0].axhline(args.max_energy_drift_pct_KE_per_ns, linestyle=":", lw=0.9)
    ax[1, 0].set_xlabel("Independent run")
    ax[1, 0].set_ylabel("NVE drift (% of KE / ns)")
    ax[1, 0].set_title("(c) Energy conservation")

    ax[1, 1].plot(x, [float(r["gap_drift_A_per_100ps"]) for r in results], marker="o")
    ax[1, 1].axhline(args.max_gap_drift_A_per_100ps, linestyle=":", lw=0.9)
    ax[1, 1].axhline(-args.max_gap_drift_A_per_100ps, linestyle=":", lw=0.9)
    ax[1, 1].set_xlabel("Independent run")
    ax[1, 1].set_ylabel(r"Gap drift (Å / 100 ps)")
    ax[1, 1].set_title("(d) Structural stability")
    fig.tight_layout()
    figs += save_figure(fig, outdir / "FigS3_ensemble_QC", args.dpi, args.formats)

    # New: ensemble-averaged energy-integral validation with residuals.
    if ensemble_fit is not None and ensemble_fit.get("status") == "OK" and ensemble_series is not None:
        s = ensemble_series
        mask = s["fit_mask"]
        fig, ax = plt.subplots(2, 1, figsize=(3.54, 4.45), gridspec_kw={"height_ratios": [3.0, 1.0]}, sharex=True)
        ax[0].plot(s["integral"], s["Qhot_mean"], lw=0.65, alpha=0.55, label=r"$\langle Q_{hot}\rangle$")
        ax[0].plot(s["integral"], s["Qcold_mean"], lw=0.65, alpha=0.55, label=r"$\langle Q_{cold}\rangle$")
        ax[0].plot(s["integral"], s["Qsym_mean"], lw=1.15, label=r"$\langle Q_{sym}\rangle$")
        if np.any(np.isfinite(s["Qsym_sem"])):
            ax[0].fill_between(s["integral"], s["Qsym_mean"]-s["Qsym_sem"], s["Qsym_mean"]+s["Qsym_sem"], alpha=0.14, linewidth=0, label="SEM")
        ax[0].plot(s["integral"][mask], s["fit_pred"][mask], linestyle="--", label=rf"ensemble fit, $R^2$={float(ensemble_fit['R2_energy_sym']):.5f}")
        ax[0].set_ylabel("Mean transferred energy (eV)")
        ax[0].set_title(rf"Ensemble validation: $G$={float(ensemble_fit['G_sym_MW_m2K']):.3f} MW m$^{-2}$ K$^{-1}$")
        ax[0].legend(loc="best")
        resid = s["Qsym_mean"][mask] - s["fit_pred"][mask]
        ax[1].plot(s["integral"][mask], resid, marker="o", markersize=2.0, linestyle="none")
        ax[1].axhline(0.0, linewidth=0.8)
        ax[1].set_xlabel(r"$\int \langle\Delta T\rangle\,dt$ (K ps)")
        ax[1].set_ylabel("Residual (eV)")
        fig.tight_layout()
        figs += save_figure(fig, outdir / "Fig5_ensemble_energy_integral", args.dpi, args.formats)

    # New: ensemble fit-end sensitivity.
    good_end = [r for r in (ensemble_end_sensitivity or []) if r.get("status") == "OK" and "G_sym_MW_m2K" in r]
    if good_end:
        good_end = sorted(good_end, key=lambda r: float(r["value"]))
        fractions = np.array([float(r["value"]) for r in good_end])
        gvals = np.array([float(r["G_sym_MW_m2K"]) for r in good_end])
        r2vals = np.array([float(r["R2_energy_sym"]) for r in good_end])
        fig, ax = plt.subplots(1, 2, figsize=(7.08, 2.55))
        ax[0].plot(fractions, gvals, marker="o")
        ax[0].axvline(float(args.end_fraction), linestyle=":", lw=0.8)
        ax[0].set_xlabel(r"Fit end ($\Delta T/\Delta T_0$)")
        ax[0].set_ylabel(r"Ensemble $G$ (MW m$^{-2}$ K$^{-1}$)")
        ax[0].set_title("(a) Ensemble fit-end sensitivity")
        ax[1].plot(fractions, r2vals, marker="o")
        ax[1].axvline(float(args.end_fraction), linestyle=":", lw=0.8)
        ax[1].set_xlabel(r"Fit end ($\Delta T/\Delta T_0$)")
        ax[1].set_ylabel(r"Ensemble energy-fit $R^2$")
        ax[1].set_title("(b) Linearity")
        fig.tight_layout()
        figs += save_figure(fig, outdir / "FigS4_ensemble_fit_end_sensitivity", args.dpi, args.formats)

    # Ready-to-use main summary.  Do not select only QC-pass runs: the independent
    # all-run distribution is the primary statistical result.  Panel (b) uses the
    # ensemble-average energy-integral fit when available.
    target = np.nanmean(G)
    rep_idx = int(np.nanargmin(np.abs(G-target)))
    p_rep = processed_runs[rep_idx]
    rr = results[rep_idx]
    fig, ax = plt.subplots(2, 2, figsize=(7.08, 5.4))
    if len(grid):
        mean = np.mean(arr, axis=0)
        ax[0, 0].plot(grid, mean, lw=1.25)
        if len(arr) > 1:
            ax[0, 0].fill_between(grid, mean-sem, mean+sem, alpha=0.20, linewidth=0)
    ax[0, 0].set_xlabel("Time (ps)")
    ax[0, 0].set_ylabel(r"$\Delta T$ (K)")
    ax[0, 0].set_title("(a) Ensemble thermal relaxation")

    if ensemble_fit is not None and ensemble_fit.get("status") == "OK" and ensemble_series is not None:
        s = ensemble_series
        m = s["fit_mask"]
        ax[0, 1].plot(s["integral"], s["Qsym_mean"], lw=1.0)
        ax[0, 1].plot(s["integral"][m], s["fit_pred"][m], linestyle="--")
        ax[0, 1].set_title(f"(b) Ensemble fit, $R^2$={float(ensemble_fit['R2_energy_sym']):.5f}")
        ax[0, 1].set_xlabel(r"$\int \langle\Delta T\rangle\,dt$ (K ps)")
        ax[0, 1].set_ylabel("Mean transferred energy (eV)")
    else:
        m = p_rep["fit_mask"]
        fit = linear_fit(p_rep["integral"][m], p_rep["Qsym"][m])
        ax[0, 1].plot(p_rep["integral"], p_rep["Qsym"], lw=1.0)
        xx = np.linspace(float(np.min(p_rep["integral"][m])), float(np.max(p_rep["integral"][m])), 300)
        ax[0, 1].plot(xx, float(fit["slope"])*xx+float(fit["intercept"]), linestyle="--")
        ax[0, 1].set_xlabel(r"$\int \Delta T\,dt$ (K ps)")
        ax[0, 1].set_ylabel("Transferred energy (eV)")
        ax[0, 1].set_title(f"(b) Representative fit, $R^2$={float(rr['R2_energy_sym']):.5f}")

    ax[1, 0].plot(x, G, marker="o", linestyle="none")
    if len(G) > 1:
        gm, gs = np.mean(G), np.std(G, ddof=1)
        ax[1, 0].axhline(gm)
        ax[1, 0].fill_between([0.5, len(G)+0.5], gm-gs, gm+gs, alpha=0.16)
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xlabel("Independent run")
    ax[1, 0].set_ylabel(r"$G$ (MW m$^{-2}$ K$^{-1}$)")
    ax[1, 0].set_title("(c) Independent-run conductance")

    ax[1, 1].plot(p_rep["time_ps"], p_rep["T_mos2"], label="MoS$_2$")
    ax[1, 1].plot(p_rep["time_ps"], p_rep["T_aln"], label="h-AlN")
    ax[1, 1].axvspan(float(rr["fit_start_ps"]), float(rr["fit_end_ps"]), alpha=0.10)
    ax[1, 1].set_xlabel("Time (ps)")
    ax[1, 1].set_ylabel("Temperature (K)")
    ax[1, 1].set_title("(d) Representative layer temperatures")
    ax[1, 1].legend(loc="best")
    fig.tight_layout()
    figs += save_figure(fig, outdir / "Fig_MAIN_publication_summary", args.dpi, args.formats)
    return figs


# ------------------------------- main ----------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Publication-grade h-AlN/h-MoS2 ITC/TBC analysis with ensemble validation."
    )
    p.add_argument("files", nargs="*", help="itc_relax_*.dat files from AlN_MoS2_ITC_ITR_FINAL_CORRECTED.in")
    p.add_argument("--hot", choices=["auto", "mos2", "aln"], default="auto")
    p.add_argument("--nominal-temp", type=float, default=300.0,
                   help="nominal finite-system reference temperature used in LAMMPS (fallback for QC/reporting)")
    p.add_argument("--block-ps", type=float, default=0.50,
                   help="NON-overlapping block width used for the primary regression")
    p.add_argument("--fit-start-ps", type=float, default=1.0,
                   help="exclude this initial thermostat-removal interval")
    p.add_argument("--fit-end-ps", type=float, default=None,
                   help="manual fit end; default uses --end-fraction")
    p.add_argument("--end-fraction", type=float, default=0.25,
                   help="automatic fit end when DeltaT reaches this fraction of DeltaT0")
    p.add_argument("--delta0-window-ps", type=float, default=0.50,
                   help="early window used to define DeltaT0 robustly")
    p.add_argument("--threshold-sustain-points", type=int, default=3,
                   help="require this many consecutive blocked points below threshold")
    p.add_argument("--min-fit-points", type=int, default=20)

    p.add_argument("--sensitivity-blocks", type=parse_float_list, default=[0.25, 0.50, 1.00],
                   help="comma-separated block widths for sensitivity analysis")
    p.add_argument("--sensitivity-fit-starts", type=parse_float_list, default=[0.0, 0.5, 1.0, 2.0],
                   help="comma-separated fit starts for sensitivity analysis")
    p.add_argument("--sensitivity-end-fractions", type=parse_float_list,
                   default=[0.50, 0.40, 0.30, 0.25, 0.20],
                   help="comma-separated DeltaT/DeltaT0 fractions for fit-end sensitivity")

    # Transparent QC thresholds. These flags are diagnostic; all-run statistics are always retained.
    p.add_argument("--min-r2", type=float, default=0.98)
    p.add_argument("--max-side-disagreement-pct", type=float, default=10.0)
    p.add_argument("--min-fit-duration-ps", type=float, default=5.0)
    p.add_argument("--max-gap-drift-A-per-100ps", type=float, default=0.10)
    p.add_argument("--max-energy-drift-pct-KE-per-ns", type=float, default=1.0)
    p.add_argument("--max-partition-error-eV", type=float, default=1.0e-5)
    p.add_argument("--max-mean-temp-deviation-K", type=float, default=15.0)
    p.add_argument("--max-deltaT0-deviation-pct", type=float, default=15.0)
    p.add_argument("--max-sensitivity-span-pct", type=float, default=10.0)

    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=20260831)
    p.add_argument("--dpi", type=int, default=1200,
                   help="raster figure resolution; PDF remains vector")
    p.add_argument("--formats", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   default=["png", "pdf"], help="comma-separated figure formats")
    p.add_argument("--outdir", default="ITC_publication_results")
    p.add_argument("--structure-data", default=None,
                   help="optional original LAMMPS data file for geometry audit")
    return p


def main() -> int:
    set_publication_style()
    args = build_parser().parse_args()
    if not args.files and not args.structure_data:
        print("ERROR: provide at least one itc_relax_*.dat file or --structure-data.", file=sys.stderr)
        return 2
    if args.block_ps <= 0:
        print("ERROR: --block-ps must be >0.", file=sys.stderr)
        return 2
    if not (0 < args.end_fraction < 1):
        print("ERROR: --end-fraction must lie between 0 and 1.", file=sys.stderr)
        return 2
    if any((not np.isfinite(f)) or f <= 0 or f >= 1 for f in args.sensitivity_end_fractions):
        print("ERROR: every --sensitivity-end-fractions value must lie strictly between 0 and 1.", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.structure_data:
        audit = parse_lammps_data(Path(args.structure_data))
        with (outdir / "structure_audit.txt").open("w") as f:
            for k, v in audit.items():
                f.write(f"{k}: {v}\n")
        print("\n================ STRUCTURE AUDIT ================")
        for k, v in audit.items():
            print(f"{k}: {v}")

    results: List[Dict[str, object]] = []
    processed_runs: List[Dict[str, np.ndarray]] = []
    for name in args.files:
        filename = Path(name)
        print(f"\nAnalyzing {filename} ...")
        try:
            result, processed, _ = analyze_one(filename, args, outdir)
        except Exception as exc:
            print(f"ERROR analyzing {filename}: {exc}", file=sys.stderr)
            return 1
        results.append(result)
        processed_runs.append(processed)

        print("================ ITC RESULT ================")
        print(f"Direction: {result['hot_layer']} -> {result['cold_layer']}")
        if result.get("seed", -1) != -1:
            print(f"Metadata: seed={result['seed']}, LJMODE={result['LJMODE']}, CHI={result['CHI']}, T0={result['T0_target_K']} K, target DeltaT={result['DT0_target_K']} K")
        print(f"Fit window: {float(result['fit_start_ps']):.3f} to {float(result['fit_end_ps']):.3f} ps")
        print(f"DeltaT0: {float(result['DeltaT0_K']):.3f} K")
        print(f"Mean system temperature (T_all): {float(result['mean_system_T_K']):.3f} K")
        print(f"Arithmetic layer-temperature midpoint: {float(result['mean_interface_T_K']):.3f} K")
        if np.isfinite(float(result.get("T_mos_target_K", float("nan")))):
            print(f"Preparation targets: MoS2={float(result['T_mos_target_K']):.3f} K, h-AlN={float(result['T_aln_target_K']):.3f} K")
        print(f"G: {float(result['G_sym_MW_m2K']):.6g} MW m^-2 K^-1")
        print(f"R: {float(result['R_sym_x1e8_m2K_W']):.6g} x10^-8 m^2 K W^-1")
        print(f"Energy-integral R^2: {float(result['R2_energy_sym']):.6f}")
        print(f"G_hot/G_cold: {float(result['G_hot_MW_m2K']):.6g} / {float(result['G_cold_MW_m2K']):.6g} MW m^-2 K^-1")
        print(f"NVE drift: {float(result['energy_drift_eV_per_ns']):.6g} eV/ns")
        print(f"Mean gap: {float(result['gap_mean_A']):.6f} +/- {float(result['gap_std_A']):.6f} A")
        print(f"QC: {result['quality_flags']} | advisory: {result['advisory_flags']}")

    if results:
        write_dict_rows_csv(outdir / "ITC_individual_runs.csv", results)

        # Primary statistics: ALL independent positive runs. QC-pass-only numbers
        # are retained as a secondary diagnostic and are never substituted silently.
        all_summary = ensemble_summary(results, "ALL_POSITIVE_RUNS", args)
        qc_indices = [i for i, r in enumerate(results) if bool(r["qc_pass"])]
        qc_rows = [results[i] for i in qc_indices]
        qc_processed = [processed_runs[i] for i in qc_indices]
        qc_summary = ensemble_summary(qc_rows, "QC_PASS_RUNS", args)

        all_validation, all_series = ensemble_energy_integral_fit(
            results, processed_runs, args, "ALL_POSITIVE_RUNS"
        )
        if qc_rows:
            qc_validation, qc_series = ensemble_energy_integral_fit(
                qc_rows, qc_processed, args, "QC_PASS_RUNS"
            )
        else:
            qc_validation, qc_series = {"subset": "QC_PASS_RUNS", "n_runs": 0, "status": "SKIPPED_NO_RUNS"}, None

        merge_validation_fit(all_summary, all_validation)
        merge_validation_fit(qc_summary, qc_validation)
        summaries = [all_summary, qc_summary]
        write_dict_rows_csv(outdir / "ITC_ensemble_summary.csv", summaries)
        write_dict_rows_csv(outdir / "ITC_ensemble_energy_integral_summary.csv", [all_validation, qc_validation])
        write_ensemble_series_csv(outdir / "ITC_ensemble_energy_integral_ALL_POSITIVE_RUNS.csv", all_series)
        write_ensemble_series_csv(outdir / "ITC_ensemble_energy_integral_QC_PASS_RUNS.csv", qc_series)

        all_end_sens = ensemble_end_fraction_sensitivity(results, processed_runs, args, "ALL_POSITIVE_RUNS")
        qc_end_sens = ensemble_end_fraction_sensitivity(qc_rows, qc_processed, args, "QC_PASS_RUNS") if qc_rows else []
        write_dict_rows_csv(outdir / "ITC_ensemble_fit_end_sensitivity.csv", all_end_sens + qc_end_sens)

        with (outdir / "ITC_ensemble_summary.txt").open("w") as f:
            for s in summaries:
                f.write(f"[{s['subset']}]\n")
                for k, v in s.items():
                    if k != "subset":
                        f.write(f"{k}: {v}\n")
                f.write("\n")
            f.write("[ENSEMBLE_AVERAGED_ENERGY_INTEGRAL_VALIDATION]\n")
            for k, v in all_validation.items():
                f.write(f"{k}: {v}\n")
            f.write("\nNOTE: ensemble-average fit is a validation estimator; independent-run statistics are primary.\n")

        ensemble_figs = plot_ensemble(
            results, processed_runs, outdir, args,
            ensemble_fit=all_validation,
            ensemble_series=all_series,
            ensemble_end_sensitivity=all_end_sens,
        )

        # Reproducibility manifest.
        manifest = outdir / "analysis_manifest.txt"
        with manifest.open("w") as f:
            f.write("h-AlN/h-MoS2 publication-grade ITC analysis\n")
            f.write("Primary estimator: energy-integral fits of independent thermostat-free NVE trajectories\n")
            f.write("Primary statistics: mean/SD/SEM/bootstrap CI across independent per-run G and R values\n")
            f.write("Ensemble-average energy-integral fit: validation only; not a replacement for independent-run uncertainty\n")
            f.write("Temperature QC: global LAMMPS T_all; arithmetic layer midpoint reported separately\n")
            f.write(f"block_ps={args.block_ps}\n")
            f.write(f"fit_start_ps={args.fit_start_ps}\n")
            f.write(f"fit_end_ps={args.fit_end_ps}\n")
            f.write(f"end_fraction={args.end_fraction}\n")
            f.write(f"sensitivity_end_fractions={','.join(str(x) for x in args.sensitivity_end_fractions)}\n")
            f.write(f"nominal_temp_K={args.nominal_temp}\n")
            f.write(f"dpi={args.dpi}\n")
            f.write(f"formats={','.join(args.formats)}\n")
            f.write(f"bootstrap_samples={args.bootstrap_samples}\n")
            f.write("input_files_and_sha256:\n")
            for name in args.files:
                q = Path(name)
                f.write(f"  {q}: {sha256_file(q)}\n")
            if args.structure_data:
                q = Path(args.structure_data)
                f.write(f"structure_file_sha256: {q}: {sha256_file(q)}\n")
            script_path = Path(__file__).resolve()
            f.write(f"analysis_script_sha256: {script_path}: {sha256_file(script_path)}\n")
            f.write("ensemble_figures:\n")
            for pth in ensemble_figs:
                f.write(f"  {pth}\n")

        print("\n================ ENSEMBLE SUMMARY ================")
        for s in summaries:
            print(f"[{s['subset']}]")
            for k, v in s.items():
                if k != "subset":
                    print(f"  {k}: {v}")

        print("\n[ENSEMBLE-AVERAGED ENERGY-INTEGRAL VALIDATION]")
        if all_validation.get("status") == "OK":
            print(f"  G = {float(all_validation['G_sym_MW_m2K']):.6g} MW m^-2 K^-1")
            print(f"  R = {float(all_validation['R_sym_x1e8_m2K_W']):.6g} x10^-8 m^2 K W^-1")
            print(f"  R2 = {float(all_validation['R2_energy_sym']):.6f}")
            print(f"  fit window = {float(all_validation['fit_start_ps']):.3f} to {float(all_validation['fit_end_ps']):.3f} ps")
            print("  NOTE: this is a validation fit; use independent-run mean/SD/SEM for statistical reporting.")
        else:
            print(f"  status = {all_validation.get('status')}")
            print(f"  issues = {all_validation.get('compatibility_issues', 'unknown')}")

        print(f"\nResults directory: {outdir}")
        print("Publication PNG figures are written at the requested DPI; PDFs are vector graphics.")

        if len(results) < 10:
            print("WARNING: fewer than 10 independent runs were supplied; inspect SD/SEM/CV before treating the ensemble as statistically converged.")
        elif len(results) < 20:
            print("NOTE: 10-19 independent runs supplied. Check cumulative stability of mean/SD/SEM; additional runs may still be useful if scatter remains appreciable.")
        if len(qc_rows) < len(results):
            print("NOTE: one or more runs failed a predefined QC diagnostic. ALL-run statistics remain primary; QC-pass-only summaries are secondary diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
