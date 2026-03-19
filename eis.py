
import io
import re
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.optimize import least_squares, lsq_linear
from scipy.signal import find_peaks
from matplotlib.ticker import FuncFormatter


# =========================================================
# 1. 기본 설정
# =========================================================
st.set_page_config(page_title="EIS Interactive Fitter + Batch Review", layout="wide")
st.title("EIS Interactive / Batch Circuit Fitter")
st.caption("DRT 선분석 기반 초기값 추천, 수동 보정, 배치 리뷰 및 선택형 export")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
E0 = 8.854e-14
EPSILON_R = {"B_1": 2.3, "C_1": 3.5}

if "reviewed_batch_items" not in st.session_state:
    st.session_state["reviewed_batch_items"] = []


# =========================================================
# 2. 이름 변환 유틸
# =========================================================
def normalize_sam_name(name: str) -> str:
    if name is None:
        return None
    n = str(name).strip().upper().replace("-", "_")
    if "ODPA" in n or "B_1" in n or "B-1" in n:
        return "B_1"
    if "BTA" in n or "C_1" in n or "C-1" in n:
        return "C_1"
    return n


def display_sam_name(name: str) -> str:
    n = normalize_sam_name(name)
    if n == "B_1":
        return "B-1"
    if n == "C_1":
        return "C-1"
    return str(name)


def make_safe_key(*parts) -> str:
    raw = "||".join(map(str, parts))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# =========================================================
# 3. 회로 수학
# =========================================================
def Z_R(R, w):
    return np.full_like(w, complex(max(float(R), 1e-30), 0.0), dtype=np.complex128)


def Z_C(C, w):
    return 1.0 / (1j * w * max(float(C), 1e-30))


def Z_CPE(T, P, w):
    t = max(float(T), 1e-30)
    p = np.clip(float(P), 0.0, 1.0)
    return 1.0 / (t * (1j * w) ** p)


def Z_Ws_sigma(sigma, w):
    return max(float(sigma), 1e-30) / np.sqrt(1j * w)


def Z_parallel(Z1, Z2):
    return 1.0 / (1.0 / Z1 + 1.0 / Z2)


def Z_C_1(params, w):
    Rs, Ws1_sigma, C1, R1, T1, P1, R2 = params
    Z0 = Z_parallel(Z_Ws_sigma(Ws1_sigma, w), Z_C(C1, w))
    Z1 = Z0 + Z_R(R1, w)
    Z2 = Z_parallel(Z1, Z_CPE(T1, P1, w))
    Z3 = Z2 + Z_R(R2, w)
    return Z3 + Z_R(Rs, w)


def Z_B_1(params, w):
    CPE_dl_T, CPE_dl_P, R_ct, R_int, T_int, P_int, R_sam, T_sam, P_sam, Rs_sol = params
    Z0 = Z_parallel(Z_CPE(CPE_dl_T, CPE_dl_P, w), Z_R(R_ct, w))
    Z1 = Z0 + Z_R(R_int, w)
    Z2 = Z_parallel(Z1, Z_CPE(T_int, P_int, w))
    Z3 = Z2 + Z_R(R_sam, w)
    Z4 = Z_parallel(Z3, Z_CPE(T_sam, P_sam, w))
    return Z4 + Z_R(Rs_sol, w)


def get_model_info(sam_type):
    sam_type = normalize_sam_name(sam_type)

    if sam_type == "B_1":
        names = [
            "CPE_dl_T", "CPE_dl_P",
            "R_ct", "R_int",
            "CPE_int_T", "CPE_int_P",
            "R_sam", "CPE_sam_T", "CPE_sam_P",
            "Rs_sol"
        ]
        lower = np.array([
            1e-12, 0.0,
            1e-6, 1e-6,
            1e-12, 0.0,
            1e-6, 1e-12, 0.0,
            1e-6
        ], dtype=float)
        upper = np.array([
            1e-1, 1.0,
            1e12, 1e12,
            1e-1, 1.0,
            1e12, 1e-1, 1.0,
            1e12
        ], dtype=float)
        return names, lower, upper, Z_B_1

    names = ["Rs_sol", "W1_Sigma", "C1_inner", "R1_inner", "CPE1_T_outer", "CPE1_P_outer", "R2_interface"]
    lower = np.array([1e-6, 1e-9, 1e-12, 1e-6, 1e-12, 0.0, 1e-6], dtype=float)
    upper = np.array([1e12, 1e9, 1e-1, 1e12, 1e-1, 1.0, 1e12], dtype=float)
    return names, lower, upper, Z_C_1


# =========================================================
# 4. 데이터 읽기
# =========================================================
def normalize_column_name(col):
    c = str(col).strip().lower()
    c = c.replace(" ", "").replace("_", "").replace("-", "")
    return c


def find_matching_column(columns, aliases):
    normalized_map = {normalize_column_name(c): c for c in columns}
    for alias in aliases:
        alias_norm = normalize_column_name(alias)
        if alias_norm in normalized_map:
            return normalized_map[alias_norm]
    return None


def standardize_eis_columns(df):
    freq_col = find_matching_column(df.columns, ["Frequency(Hz)", "Frequency", "Freq", "f", "Hz"])
    zre_col = find_matching_column(df.columns, ["Zre(ohm)", "Zre", "ReZ", "Zreal", "RealZ"])
    zim_col = find_matching_column(df.columns, ["Zim(ohm)", "Zim", "ImZ", "Zimag", "ImagZ"])
    neg_zim_col = find_matching_column(df.columns, ["-Zim(ohm)", "-Zim", "NegZim", "-ImZ", "-ImagZ"])

    if freq_col is None or zre_col is None:
        raise ValueError("Frequency 또는 Zre 컬럼을 찾지 못했습니다.")
    if zim_col is None and neg_zim_col is None:
        raise ValueError("Zim 또는 -Zim 컬럼을 찾지 못했습니다.")

    out = pd.DataFrame()
    out["Frequency(Hz)"] = pd.to_numeric(df[freq_col], errors="coerce")
    out["Zre(ohm)"] = pd.to_numeric(df[zre_col], errors="coerce")
    out["Zim(ohm)"] = pd.to_numeric(df[zim_col], errors="coerce") if zim_col is not None else -pd.to_numeric(df[neg_zim_col], errors="coerce")
    return out


def pick_data_sheet(xl):
    candidate_sheets = [s for s in xl.sheet_names if str(s).upper().startswith("DATA")]
    if not candidate_sheets:
        candidate_sheets = xl.sheet_names

    best_sheet = None
    best_count = -1
    for s in candidate_sheets:
        try:
            df_try = xl.parse(s)
            df_std = standardize_eis_columns(df_try)
            cnt = int(df_std.dropna().shape[0])
            if cnt > best_count:
                best_count = cnt
                best_sheet = s
        except Exception:
            continue

    if best_sheet is None:
        raise ValueError("Data 시트를 찾지 못했습니다.")
    return best_sheet


@st.cache_data(show_spinner=False)
def read_eis_from_bytes(file_bytes: bytes):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet = pick_data_sheet(xl)
    df = xl.parse(sheet)
    out = standardize_eis_columns(df)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out["Frequency(Hz)"] > 0].sort_values(by="Frequency(Hz)", ascending=False).reset_index(drop=True)

    if out.empty:
        raise ValueError("유효한 EIS 데이터가 없습니다.")

    freq = out["Frequency(Hz)"].to_numpy(dtype=float)
    zexp = out["Zre(ohm)"].to_numpy(dtype=float) + 1j * out["Zim(ohm)"].to_numpy(dtype=float)
    return sheet, out, freq, zexp


def parse_metadata_from_filename(name):
    stem = Path(name).stem.upper().replace("[", "_").replace("]", "_")
    sam = "B_1" if any(k in stem for k in ["ODPA", "B-1", "B_1"]) else ("C_1" if any(k in stem for k in ["BTA", "C-1", "C_1"]) else None)
    sub = "CU" if re.search(r"(^|[_\-\s])CU([_\-\s]|$)", stem) else ("CO" if re.search(r"(^|[_\-\s])CO([_\-\s]|$)", stem) else None)
    m = re.search(r"(\d+(?:\.\d+)?)\s*MM", stem)
    conc = float(m.group(1)) if m else None
    return sam, sub, conc


# =========================================================
# 5. 공통 계산 유틸
# =========================================================
def build_exclude_indices(freq, manual_exclude_indices=None, lowfreq_cutoff_hz=None, enable_lowfreq_cut=False):
    idx_set = set()
    if manual_exclude_indices:
        idx_set.update([i for i in manual_exclude_indices if 0 <= i < len(freq)])

    if enable_lowfreq_cut and lowfreq_cutoff_hz is not None:
        idx_set.update(np.where(np.asarray(freq, dtype=float) < float(lowfreq_cutoff_hz))[0].tolist())

    return sorted(idx_set)


def apply_exclusion(freq, zexp, exclude_indices=None):
    f_fit = np.asarray(freq, dtype=float).copy()
    z_fit = np.asarray(zexp, dtype=np.complex128).copy()
    mask = np.ones(len(f_fit), dtype=bool)
    valid_idx = []
    if exclude_indices:
        valid_idx = [i for i in exclude_indices if 0 <= i < len(f_fit)]
        mask[valid_idx] = False
        f_fit = f_fit[mask]
        z_fit = z_fit[mask]
    return f_fit, z_fit, mask, valid_idx


def calc_fit_metrics(z_true, z_pred):
    z_true = np.asarray(z_true, dtype=np.complex128)
    z_pred = np.asarray(z_pred, dtype=np.complex128)
    ss_res = np.sum(np.abs(z_pred - z_true) ** 2)
    ss_tot = np.sum(np.abs(z_true - np.mean(z_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(np.abs(z_pred - z_true) ** 2))
    return float(r2), float(rmse)


# =========================================================
# 6. DRT
# =========================================================
def build_drt_matrices(freq, tau):
    omega = 2 * np.pi * np.asarray(freq, dtype=float)
    tau = np.asarray(tau, dtype=float)

    wt = np.outer(omega, tau)
    den = 1.0 + wt ** 2
    a_re = 1.0 / den
    a_im = -wt / den

    a_re_full = np.hstack([np.ones((len(freq), 1)), a_re])
    a_im_full = np.hstack([np.zeros((len(freq), 1)), a_im])
    a_full = np.vstack([a_re_full, a_im_full])
    return a_full, a_re, a_im


def build_second_diff_matrix(n):
    if n < 3:
        return np.zeros((0, n))
    l = np.zeros((n - 2, n))
    idx = np.arange(n - 2)
    l[idx, idx] = 1.0
    l[idx, idx + 1] = -2.0
    l[idx, idx + 2] = 1.0
    return l


@st.cache_data(show_spinner=False)
def compute_drt(freq, zexp, reg_param=1e-3, tau_density=3):
    """
    비음수 제약 + 2차 미분 Tikhonov 정규화.
    목적함수:
        ||A x - y||^2 + λ ||L x||^2
    를 증강 시스템으로 변환하여 lsq_linear로 풉니다.
    """
    freq = np.asarray(freq, dtype=float)
    zexp = np.asarray(zexp, dtype=np.complex128)

    if len(freq) < 5:
        raise ValueError("DRT 계산을 위한 데이터 포인트가 부족합니다.")

    min_f, max_f = np.min(freq), np.max(freq)
    tau_min = 1.0 / (2 * np.pi * max_f) / 10.0
    tau_max = 1.0 / (2 * np.pi * min_f) * 10.0
    n_tau = max(int(len(freq) * tau_density), 40)

    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
    a_full, a_re, a_im = build_drt_matrices(freq, tau)
    y_full = np.concatenate([zexp.real, zexp.imag])

    l = build_second_diff_matrix(n_tau)
    l_full = np.hstack([np.zeros((l.shape[0], 1)), l])

    reg_scale = np.sqrt(float(reg_param))
    a_aug = np.vstack([a_full, reg_scale * l_full])
    y_aug = np.concatenate([y_full, np.zeros(l_full.shape[0])])

    lb = np.zeros(n_tau + 1)
    ub = np.full(n_tau + 1, np.inf)

    x0_rinf = max(1e-6, float(np.min(zexp.real)))
    x0 = np.zeros(n_tau + 1)
    x0[0] = x0_rinf

    res = lsq_linear(
        a_aug,
        y_aug,
        bounds=(lb, ub),
        method="trf",
        lsmr_tol="auto",
        max_iter=5000,
        verbose=0,
    )

    if not res.success:
        raise ValueError(f"DRT 계산 실패: {res.message}")

    x = res.x
    r_inf = float(x[0])
    gamma = np.maximum(x[1:], 0.0)
    f_drt = 1.0 / (2 * np.pi * tau)

    z_drt = r_inf + gamma @ a_re.T + 1j * (gamma @ a_im.T)

    tau_sorted_idx = np.argsort(tau)
    tau_sorted = tau[tau_sorted_idx]
    gamma_sorted = gamma[tau_sorted_idx]
    total_r = float(np.trapezoid(gamma_sorted, x=np.log(tau_sorted)))

    if np.sum(gamma_sorted) > 0:
        ln_tau_char = np.average(np.log(tau_sorted), weights=gamma_sorted + 1e-30)
        tau_char = float(np.exp(ln_tau_char))
    else:
        tau_char = float(np.median(tau_sorted))

    c_char = float(tau_char / max(total_r, 1e-30))
    return {
        "f_drt": f_drt,
        "tau": tau,
        "gamma": gamma,
        "r_inf": r_inf,
        "z_drt": z_drt,
        "total_r": total_r,
        "tau_char": tau_char,
        "c_char": c_char,
        "status_text": "ok",
    }


def extract_drt_peaks(tau, gamma, max_peaks=3):
    tau = np.asarray(tau, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if len(tau) < 3 or np.all(gamma <= 0):
        return []

    prominence = max(np.max(gamma) * 0.05, 1e-12)
    peaks, props = find_peaks(gamma, prominence=prominence)
    if len(peaks) == 0:
        idx = int(np.argmax(gamma))
        peaks = np.array([idx])

    peak_list = []
    for p in peaks:
        peak_list.append({
            "idx": int(p),
            "tau": float(tau[p]),
            "freq": float(1.0 / (2 * np.pi * tau[p])),
            "height": float(gamma[p]),
            "prominence": float(props["prominences"][np.where(peaks == p)[0][0]]) if "prominences" in props and len(props["prominences"]) else float(gamma[p]),
        })

    peak_list = sorted(peak_list, key=lambda x: x["height"], reverse=True)[:max_peaks]
    peak_list = sorted(peak_list, key=lambda x: x["freq"], reverse=True)
    return peak_list


def build_initial_guess_from_drt(freq, zexp, sam_type, drt_result):
    sam_type = normalize_sam_name(sam_type)
    zre = np.asarray(zexp.real, dtype=float)
    rs_guess = max(1e-3, float(np.min(zre)))
    r_span = max(float(np.max(zre) - np.min(zre)), 1.0)

    peaks = extract_drt_peaks(drt_result["tau"], drt_result["gamma"], max_peaks=3)
    total_r = max(drt_result["total_r"], 1.0)
    c_char = np.clip(drt_result["c_char"], 1e-10, 1e-3)

    if sam_type == "B_1":
        if len(peaks) >= 3:
            hf, mf, lf = peaks[0], peaks[1], peaks[2]
            r_ct = max(hf["height"] * 0.7, 0.08 * total_r)
            r_int = max(mf["height"] * 0.9, 0.15 * total_r)
            r_sam = max(lf["height"] * 1.1, 0.30 * total_r)
            cpe_dl_t = np.clip(hf["tau"] / max(r_ct, 1e-12), 1e-8, 1e-3)
            cpe_int_t = np.clip(mf["tau"] / max(r_int, 1e-12), 1e-9, 1e-3)
            cpe_sam_t = np.clip(lf["tau"] / max(r_sam, 1e-12), 1e-10, 1e-3)
        elif len(peaks) == 2:
            hf, lf = peaks[0], peaks[1]
            r_ct = max(hf["height"] * 0.8, 0.12 * total_r)
            r_int = max(lf["height"] * 0.5, 0.20 * total_r)
            r_sam = max(lf["height"] * 1.0, 0.35 * total_r)
            cpe_dl_t = np.clip(hf["tau"] / max(r_ct, 1e-12), 1e-8, 1e-3)
            cpe_int_t = np.clip(np.sqrt(hf["tau"] * lf["tau"]) / max(r_int, 1e-12), 1e-9, 1e-3)
            cpe_sam_t = np.clip(lf["tau"] / max(r_sam, 1e-12), 1e-10, 1e-3)
        else:
            peak_tau = peaks[0]["tau"] if peaks else 1.0 / (2 * np.pi * max(float(np.median(freq)), 1e-6))
            r_ct = 0.15 * max(total_r, r_span)
            r_int = 0.25 * max(total_r, r_span)
            r_sam = 0.45 * max(total_r, r_span)
            cpe_dl_t = np.clip(peak_tau / max(r_ct, 1e-12), 1e-8, 1e-3)
            cpe_int_t = np.clip((peak_tau * 3.0) / max(r_int, 1e-12), 1e-9, 1e-3)
            cpe_sam_t = np.clip((peak_tau * 10.0) / max(r_sam, 1e-12), 1e-10, 1e-3)

        return np.array([
            cpe_dl_t, 0.95,
            max(r_ct, 1e-3), max(r_int, 1e-3),
            cpe_int_t, 0.90,
            max(r_sam, 1e-3), max(cpe_sam_t, 1e-10), 0.85,
            rs_guess
        ], dtype=float)

    # C_1
    if len(peaks) >= 2:
        hf, lf = peaks[0], peaks[1]
        c1 = np.clip(hf["tau"] / max(hf["height"], 1e-12), 1e-10, 1e-3)
        r1 = max(hf["height"], 0.30 * total_r)
        t1 = np.clip(lf["tau"] / max(lf["height"], 1e-12), 1e-10, 1e-3)
        r2 = max(lf["height"], 0.50 * total_r)
        sigma = max(np.sqrt(max(r2, 1.0)), 1e-3)
    else:
        peak_tau = peaks[0]["tau"] if peaks else 1.0 / (2 * np.pi * max(float(np.median(freq)), 1e-6))
        c1 = np.clip(c_char, 1e-10, 1e-3)
        r1 = 0.35 * max(total_r, r_span)
        t1 = np.clip((peak_tau * 5.0) / max(0.65 * max(total_r, r_span), 1e-12), 1e-10, 1e-3)
        r2 = 0.65 * max(total_r, r_span)
        sigma = 10.0

    return np.array([
        rs_guess,
        sigma,
        c1,
        max(r1, 1e-3),
        max(t1, 1e-10),
        0.90,
        max(r2, 1e-3),
    ], dtype=float)


def make_drt_figure(f_drt, gamma, title="DRT Analysis"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(f_drt, gamma, "-", linewidth=2)
    ax.fill_between(f_drt, gamma, 0, alpha=0.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gamma (ohm)")
    ax.set_title(f"DRT | {title}")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    fig.tight_layout()
    return fig


def build_drt_distribution_df(file_name, sam_type, substrate, conc, drt_result):
    order = np.argsort(drt_result["tau"])
    tau_sorted = drt_result["tau"][order]
    gamma_sorted = drt_result["gamma"][order]
    freq_sorted = 1.0 / (2 * np.pi * tau_sorted)
    return pd.DataFrame({
        "Source": file_name,
        "SAM": display_sam_name(sam_type),
        "Substrate": substrate,
        "Concentration_mM": conc,
        "Tau_s": tau_sorted,
        "Freq_Hz": freq_sorted,
        "Gamma_Ohm": gamma_sorted,
    })


# =========================================================
# 7. 피팅
# =========================================================
def residuals(params, w, zexp, model_func):
    zfit = model_func(params, w)
    scale = np.maximum(np.abs(zexp), 1.0)
    return np.concatenate([
        (zfit.real - zexp.real) / scale,
        (zfit.imag - zexp.imag) / scale
    ])


def fit_eis(freq, zexp, sam_type, x0, exclude_indices=None, custom_bounds=None):
    names, lb, ub, model_func = get_model_info(sam_type)
    if custom_bounds is not None:
        lb, ub = custom_bounds

    f_fit, z_fit, _, _ = apply_exclusion(freq, zexp, exclude_indices)
    if len(f_fit) < 3:
        raise ValueError("피팅 가능한 데이터 포인트가 너무 적습니다.")

    w = 2 * np.pi * f_fit
    x0 = np.asarray(x0, dtype=float)
    x0 = np.clip(x0, lb + 1e-15, ub - 1e-15)

    res = least_squares(
        residuals,
        x0=x0,
        bounds=(lb, ub),
        args=(w, z_fit, model_func),
        method="trf",
        max_nfev=20000,
    )

    p = res.x
    zfit_full = model_func(p, 2 * np.pi * np.asarray(freq, dtype=float))
    zfit_eval = model_func(p, w)
    r2, rmse = calc_fit_metrics(z_fit, zfit_eval)

    return {k: float(v) for k, v in zip(names, p)}, zfit_full, r2, rmse, res


def evaluate_current_params(freq, zexp, sam_type, params, exclude_indices=None):
    names, _, _, model_func = get_model_info(sam_type)
    freq = np.asarray(freq, dtype=float)
    zexp = np.asarray(zexp, dtype=np.complex128)
    params = np.asarray(params, dtype=float)

    zfit_full = model_func(params, 2 * np.pi * freq)
    f_eval, z_eval, mask, _ = apply_exclusion(freq, zexp, exclude_indices)
    zfit_eval = zfit_full[mask] if mask is not None else zfit_full

    if len(f_eval) == 0:
        return {k: float(v) for k, v in zip(names, params)}, zfit_full, 0.0, np.inf

    r2, rmse = calc_fit_metrics(z_eval, zfit_eval)
    return {k: float(v) for k, v in zip(names, params)}, zfit_full, r2, rmse


# =========================================================
# 8. 시각화
# =========================================================
def compute_nyquist_limits_positive(zexp_list, zfit_list=None, pad_ratio=0.05):
    if zfit_list is None:
        zfit_list = []

    xs, ys = [], []
    for z in list(zexp_list) + list(zfit_list):
        if z is None or len(z) == 0:
            continue
        x = np.asarray(np.real(z), dtype=float)
        y = np.asarray(-np.imag(z), dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size > 0:
            xs.append(x)
        if y.size > 0:
            ys.append(y)

    if not xs or not ys:
        return None

    upper = max(float(np.nanmax(np.concatenate(xs))), float(np.nanmax(np.concatenate(ys))), 1.0)
    upper *= (1.0 + pad_ratio)
    return (0.0, upper), (0.0, upper)


def apply_equal_nyquist_axes_positive(ax, zexp_list, zfit_list=None, pad_ratio=0.05):
    lims = compute_nyquist_limits_positive(zexp_list, zfit_list, pad_ratio)
    if lims is None:
        return
    xlim, ylim = lims
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")


def make_nyquist_figure(zexp, zfit, concentration, title="Nyquist Plot", exclude_indices=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    mask = np.ones(len(zexp), dtype=bool)
    valid_idx = []
    if exclude_indices:
        valid_idx = [i for i in exclude_indices if 0 <= i < len(zexp)]
        mask[valid_idx] = False

    ax.plot(zexp.real[mask], -zexp.imag[mask], "o", alpha=0.6, label="Exp")
    if exclude_indices and len(valid_idx) > 0:
        ax.plot(zexp.real[~mask], -zexp.imag[~mask], "x", label="Excluded")
        for i in valid_idx:
            ax.annotate(str(i), (zexp.real[i], -zexp.imag[i]))

    if zfit is not None:
        ax.plot(zfit.real, -zfit.imag, "-", label="Fit")
    ax.set_xlabel("Zre (ohm)")
    ax.set_ylabel("-Zim (ohm)")
    ax.set_title(f"{title} | {concentration} mM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    apply_equal_nyquist_axes_positive(ax, [zexp], [zfit] if zfit is not None else [], pad_ratio=0.05)
    return fig


def make_bode_figure(freq, zexp, zfit=None, title="Bode Plot", z_drt=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.semilogx(freq, np.abs(zexp), "o", label="Exp")
    if zfit is not None:
        ax1.semilogx(freq, np.abs(zfit), "-", label="Fit (Circuit)")
    if z_drt is not None:
        ax1.semilogx(freq, np.abs(z_drt), "--", label="Fit (DRT)")
    ax1.set_ylabel("|Z| (ohm)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.semilogx(freq, np.degrees(np.angle(zexp)), "o", label="Exp")
    if zfit is not None:
        ax2.semilogx(freq, np.degrees(np.angle(zfit)), "-", label="Fit (Circuit)")
    if z_drt is not None:
        ax2.semilogx(freq, np.degrees(np.angle(z_drt)), "--", label="Fit (DRT)")
    ax2.set_xlabel("Freq (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =========================================================
# 9. 결과 빌더
# =========================================================
def classify_blocking_status(total_r, substrate):
    substrate = str(substrate).upper()
    target_info = {
        "CO": (2500.0, 4500.0),
        "CU": (1500.0, 2500.0),
    }
    warning_r, safe_r = target_info.get(substrate, (2000.0, 4000.0))
    if total_r >= safe_r:
        return "Safe"
    if total_r >= warning_r:
        return "Warning"
    return "Risk"


def compute_apparent_thickness_nm(sam_type, area_cm2, c_char):
    sam_type = normalize_sam_name(sam_type)
    if c_char <= 0 or area_cm2 <= 0:
        return np.nan
    return float((E0 * EPSILON_R[sam_type] * area_cm2 / c_char) * 1e7)


def build_summary_row(
    file_name,
    sheet_name,
    substrate,
    sam_type,
    concentration,
    area_cm2,
    param_dict,
    fit_r2,
    fit_rmse,
    excluded_indices,
    lowfreq_cutoff_hz,
    drt_result,
):
    sam_type = normalize_sam_name(sam_type)

    row = {
        "Source_File": file_name,
        "Sheet_Name": sheet_name,
        "Substrate": substrate,
        "SAM": display_sam_name(sam_type),
        "SAM_INTERNAL": sam_type,
        "Concentration_mM": concentration,
        "Area_cm2": area_cm2,
        "Fit_R2": fit_r2,
        "Fit_RMSE": fit_rmse,
        "Excluded_Indices": ",".join(map(str, excluded_indices)) if excluded_indices else "",
        "LowFreq_Cutoff_Hz": lowfreq_cutoff_hz if lowfreq_cutoff_hz is not None else np.nan,
        "DRT_R_inf_Ohm": drt_result["r_inf"],
        "DRT_Total_R_Ohm": drt_result["total_r"],
        "DRT_Tau_Char_s": drt_result["tau_char"],
        "DRT_C_Char_F": drt_result["c_char"],
    }
    row.update(param_dict)

    row["Blocking_Index_Ohm_cm2"] = drt_result["total_r"] * area_cm2
    row["Apparent_Thickness_nm"] = compute_apparent_thickness_nm(sam_type, area_cm2, drt_result["c_char"])
    row["Blocking_Status"] = classify_blocking_status(drt_result["total_r"], substrate)
    return row


def build_pointwise_df(file_name, sam_type, substrate, conc, freq, zexp, zfit, z_drt):
    return pd.DataFrame({
        "Source": file_name,
        "SAM": display_sam_name(sam_type),
        "Substrate": substrate,
        "Concentration_mM": conc,
        "Freq_Hz": freq,
        "Zre_exp": zexp.real,
        "Zim_exp": zexp.imag,
        "Zre_fit_circuit": zfit.real if zfit is not None else np.nan,
        "Zim_fit_circuit": zfit.imag if zfit is not None else np.nan,
        "Zre_fit_drt": z_drt.real if z_drt is not None else np.nan,
        "Zim_fit_drt": z_drt.imag if z_drt is not None else np.nan,
    })


def build_batch_summary(df_raw):
    return df_raw.groupby(["Substrate", "SAM", "SAM_INTERNAL", "Concentration_mM"]).agg(
        n=("Concentration_mM", "count"),
        Fit_R2_mean=("Fit_R2", "mean"),
        Blocking_Index_mean=("Blocking_Index_Ohm_cm2", "mean"),
        Blocking_Index_std=("Blocking_Index_Ohm_cm2", lambda x: x.std() if len(x) > 1 else 0.0),
        DRT_Total_R_mean=("DRT_Total_R_Ohm", "mean"),
        DRT_Total_R_std=("DRT_Total_R_Ohm", lambda x: x.std() if len(x) > 1 else 0.0),
        Apparent_Thickness_mean=("Apparent_Thickness_nm", "mean"),
        Apparent_Thickness_std=("Apparent_Thickness_nm", lambda x: x.std() if len(x) > 1 else 0.0),
        Blocking_Status=("Blocking_Status", "first"),
    ).reset_index()


def make_summary_plot(df_sum, sam_type, substrate, x_log=False):
    sam_type = normalize_sam_name(sam_type)
    plot_df = df_sum[(df_sum["SAM_INTERNAL"] == sam_type) & (df_sum["Substrate"] == substrate)].sort_values("Concentration_mM")
    if plot_df.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Conc (mM)")
    ax1.set_ylabel("DRT Total R Index (Ohm cm2)")
    ax1.set_yscale("log")
    if x_log:
        ax1.set_xscale("log")

    ax1.errorbar(
        plot_df["Concentration_mM"],
        plot_df["Blocking_Index_mean"],
        yerr=plot_df["Blocking_Index_std"],
        fmt="o",
        label="Blocking Index",
    )

    for _, row in plot_df.iterrows():
        val = row["Blocking_Index_mean"]
        if pd.notna(val):
            ax1.annotate(f"{val:.2e}", (row["Concentration_mM"], val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Apparent Thickness (nm)")
    ax2.errorbar(
        plot_df["Concentration_mM"],
        plot_df["Apparent_Thickness_mean"],
        yerr=plot_df["Apparent_Thickness_std"],
        fmt="s",
        label="Apparent Thickness",
    )

    for _, row in plot_df.iterrows():
        val = row["Apparent_Thickness_mean"]
        if pd.notna(val):
            ax2.annotate(f"{val:.1f}", (row["Concentration_mM"], val), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9)

    ax1.grid(True, alpha=0.3)
    plt.title(f"Analysis: {substrate} / {display_sam_name(sam_type)}")
    return fig


def make_batch_nyquist_panel_from_queue(queue_items, sam_type, substrate):
    sam_type = normalize_sam_name(sam_type)
    filtered = sorted([i for i in queue_items if i["SAM"] == sam_type and i["Substrate"] == substrate], key=lambda x: x["Concentration_mM"])
    if not filtered:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    zexp_list, zfit_list = [], []
    for item in filtered:
        ax.plot(item["zexp"].real, -item["zexp"].imag, "o", alpha=0.4, label=f"{item['Concentration_mM']}mM")
        if item["zfit"] is not None:
            ax.plot(item["zfit"].real, -item["zfit"].imag, "-", alpha=0.8)
        zexp_list.append(item["zexp"])
        if item["zfit"] is not None:
            zfit_list.append(item["zfit"])

    ax.set_title(f"Batch Nyquist: {substrate} / {display_sam_name(sam_type)}")
    ax.set_xlabel("Zre (ohm)")
    ax.set_ylabel("-Zim (ohm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    apply_equal_nyquist_axes_positive(ax, zexp_list, zfit_list, pad_ratio=0.05)
    return fig


def build_batch_excel_bytes(df_raw, df_sum, export_sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="Raw_Fit", index=False)
        df_sum.to_excel(writer, sheet_name="Summary", index=False)

        for sheet_name, df in export_sheets.items():
            if df is not None and not df.empty:
                safe_sheet = re.sub(r"[^A-Za-z0-9_]+", "_", sheet_name)[:31]
                df.to_excel(writer, sheet_name=safe_sheet, index=False)
    buf.seek(0)
    return buf


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]+', "_", str(name))


def build_batch_zip_bytes(excel_bytes, png_groups):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"Report_{ts}.xlsx", excel_bytes.getvalue())
        for folder_name, png_dict in png_groups.items():
            for name, data in png_dict.items():
                zf.writestr(f"{folder_name}/{sanitize_filename(name)}", data)
    zip_buf.seek(0)
    return zip_buf


# =========================================================
# 10. 메인 UI
# =========================================================
st.header("1) Single-file review")
uploaded_file = st.file_uploader("xlsx 파일 업로드", type=["xlsx"], key="single_uploader")

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_token = make_safe_key(uploaded_file.name, uploaded_file.size)

    try:
        sheet_name, df_eis, freq, zexp = read_eis_from_bytes(file_bytes)
        sam_guess, substrate_guess, conc_guess = parse_metadata_from_filename(uploaded_file.name)

        col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
        with col_meta1:
            sam_display = st.selectbox("Model", ["B-1 (3-tc)", "C-1 (Warburg)"], index=0 if sam_guess != "C_1" else 1, key=f"{file_token}_mod")
            sam_type = "B_1" if "B-1" in sam_display else "C_1"

        with col_meta2:
            substrate = st.selectbox("Substrate", ["CU", "CO"], index=0 if substrate_guess != "CO" else 1, key=f"{file_token}_sub")

        with col_meta3:
            concentration = st.number_input("Conc (mM)", value=float(conc_guess) if conc_guess is not None else 0.0, key=f"{file_token}_conc")

        with col_meta4:
            area_cm2 = st.number_input("Area (cm2)", value=0.14, format="%.4f", key=f"{file_token}_area")

        st.subheader("DRT 선분석 설정")
        c1, c2, c3 = st.columns(3)
        with c1:
            reg_lambda = st.select_slider("Regularization λ", options=[1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1], value=1e-3, key=f"{file_token}_lam")
        with c2:
            tau_density = st.select_slider("Tau density", options=[2, 3, 4, 5], value=3, key=f"{file_token}_taud")
        with c3:
            enable_lowfreq_cut = st.checkbox("저주파 제외 사용", value=False, key=f"{file_token}_lf_on")
            lowfreq_cutoff_hz = st.number_input("저주파 cutoff (Hz)", value=0.1, format="%.4f", key=f"{file_token}_lf_cut")

        with st.expander("Outlier 설정"):
            outlier_key = f"{file_token}_{sam_type}_out"
            sel_out = st.multiselect("제외 인덱스", options=list(range(len(freq))), default=st.session_state.get(outlier_key, []), key=f"ms_{outlier_key}")
            st.session_state[outlier_key] = sel_out

        effective_exclude_indices = build_exclude_indices(
            freq=freq,
            manual_exclude_indices=st.session_state.get(outlier_key, []),
            lowfreq_cutoff_hz=lowfreq_cutoff_hz,
            enable_lowfreq_cut=enable_lowfreq_cut,
        )

        f_fit, z_fit, _, _ = apply_exclusion(freq, zexp, effective_exclude_indices)
        drt_result = compute_drt(f_fit, z_fit, reg_param=reg_lambda, tau_density=tau_density)
        drt_init = build_initial_guess_from_drt(freq, zexp, sam_type, drt_result)

        names, lb, ub, _ = get_model_info(sam_type)

        init_signature = (file_token, sam_type)
        if st.session_state.get(f"{file_token}_init_sig") != init_signature:
            st.session_state[f"{file_token}_init_sig"] = init_signature
            for n, v in zip(names, drt_init):
                st.session_state[f"{file_token}_{sam_type}_{n}_val"] = float(v)
            st.session_state[f"{file_token}_{sam_type}_cur_fit"] = drt_init.tolist()
            st.session_state[f"{file_token}_wver"] = st.session_state.get(f"{file_token}_wver", 0) + 1

        for n, v in zip(names, drt_init):
            key_name = f"{file_token}_{sam_type}_{n}_val"
            if key_name not in st.session_state:
                st.session_state[key_name] = float(v)

        if f"{file_token}_{sam_type}_cur_fit" not in st.session_state:
            st.session_state[f"{file_token}_{sam_type}_cur_fit"] = drt_init.tolist()

        if f"{file_token}_wver" not in st.session_state:
            st.session_state[f"{file_token}_wver"] = 0

        left, right = st.columns([1.2, 1.8])

        with left:
            st.subheader("DRT 기반 초기값 / 수동 파라미터")

            peak_df = pd.DataFrame(extract_drt_peaks(drt_result["tau"], drt_result["gamma"], max_peaks=3))
            if not peak_df.empty:
                st.caption("DRT 주요 피크")
                st.dataframe(peak_df[["freq", "tau", "height"]].rename(columns={"freq": "Freq_Hz", "tau": "Tau_s", "height": "Peak_Height_Ohm"}), use_container_width=True, height=150)

            curr_params = []
            curr_lb = lb.copy()
            curr_ub = ub.copy()

            if st.button("DRT 기반 초기값으로 재설정", use_container_width=True):
                for n, v in zip(names, drt_init):
                    st.session_state[f"{file_token}_{sam_type}_{n}_val"] = float(v)
                st.session_state[f"{file_token}_{sam_type}_cur_fit"] = drt_init.tolist()
                st.session_state[f"{file_token}_wver"] += 1
                st.rerun()

            for i, name in enumerate(names):
                s_key = f"{file_token}_{sam_type}_{name}_val"
                is_p = "_P" in name

                col_input, col_fix = st.columns([4, 1])
                with col_fix:
                    is_locked = st.checkbox("Fix", key=f"fix_{s_key}")
                with col_input:
                    val = st.number_input(
                        name,
                        value=float(st.session_state[s_key]),
                        format="%.4e" if not is_p else "%.4f",
                        key=f"in_{s_key}_{st.session_state[f'{file_token}_wver']}",
                    )
                    st.session_state[s_key] = float(val)
                    curr_params.append(float(val))

                    if is_locked:
                        eps = max(abs(val) * 1e-6, 1e-12)
                        lo = max(lb[i], val - eps)
                        hi = min(ub[i], val + eps)
                        if hi <= lo:
                            hi = min(ub[i], lo + max(abs(lo) * 1e-9, 1e-15))
                            if hi <= lo:
                                lo = max(lb[i], hi - max(abs(hi) * 1e-9, 1e-15))
                        curr_lb[i] = lo
                        curr_ub[i] = hi

            st.write("---")
            st.caption(f"현재 제외 포인트 수: {len(effective_exclude_indices)}")
            st.caption("현재 weighting: |Z| 기반")

            if st.button("현재 값에서 Auto Fit 시작", type="primary", use_container_width=True):
                try:
                    p_fit, _, _, _, _ = fit_eis(
                        freq=freq,
                        zexp=zexp,
                        sam_type=sam_type,
                        x0=curr_params,
                        exclude_indices=effective_exclude_indices,
                        custom_bounds=(curr_lb, curr_ub),
                    )
                    for n, v in p_fit.items():
                        st.session_state[f"{file_token}_{sam_type}_{n}_val"] = float(v)
                    st.session_state[f"{file_token}_{sam_type}_cur_fit"] = [float(p_fit[n]) for n in names]
                    st.session_state[f"{file_token}_wver"] += 1
                    st.success("Auto Fit 완료")
                    st.rerun()
                except Exception as e:
                    st.error(f"Auto Fit 실패: {e}")

        with right:
            live_dict, live_zfit, live_r2, live_rmse = evaluate_current_params(
                freq=freq,
                zexp=zexp,
                sam_type=sam_type,
                params=curr_params,
                exclude_indices=effective_exclude_indices,
            )

            st.subheader(f"Fit Quality | R²: {live_r2:.4f}, RMSE: {live_rmse:.2e}")

            t1, t2, t3 = st.tabs(["DRT Analysis", "Nyquist Plot", "Bode Plot"])

            with t1:
                st.caption("DRT 적분 저항 기반 방어력 지표")
                fig_drt = make_drt_figure(drt_result["f_drt"], drt_result["gamma"], title=uploaded_file.name)
                st.pyplot(fig_drt, use_container_width=True)
                plt.close(fig_drt)

                st.write("---")
                st.subheader("방어력 지표")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("DRT Total R", f"{drt_result['total_r']:.1f} Ω")
                m2.metric("R_inf", f"{drt_result['r_inf']:.1f} Ω")
                m3.metric("C_char", f"{drt_result['c_char']:.2e} F")
                m4.metric("겉보기 두께", f"{compute_apparent_thickness_nm(sam_type, area_cm2, drt_result['c_char']):.2f} nm")

                status = classify_blocking_status(drt_result["total_r"], substrate)
                if status == "Safe":
                    st.success("안전: DRT 적분 저항이 높은 편으로, 상대 비교상 차단 성능이 양호합니다.")
                elif status == "Warning":
                    st.warning("주의: 차단 성능이 중간 수준입니다. 회로 피팅 및 후속 측정과 함께 해석하세요.")
                else:
                    st.error("위험: DRT 적분 저항이 낮은 편입니다. 핀홀 또는 누설 가능성을 우선 점검하세요.")

            with t2:
                fig1 = make_nyquist_figure(
                    zexp=zexp,
                    zfit=live_zfit,
                    concentration=concentration,
                    title=uploaded_file.name,
                    exclude_indices=effective_exclude_indices,
                )
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)

            with t3:
                fig2 = make_bode_figure(freq, zexp, live_zfit, title=uploaded_file.name, z_drt=drt_result["z_drt"])
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            st.subheader("Current Live Parameters")
            st.dataframe(pd.DataFrame(live_dict.items(), columns=["Parameter", "Value"]), use_container_width=True, height=350)

            if st.button("Batch Queue에 결과 추가", use_container_width=True):
                try:
                    summary_row = build_summary_row(
                        file_name=uploaded_file.name,
                        sheet_name=sheet_name,
                        substrate=substrate,
                        sam_type=sam_type,
                        concentration=concentration,
                        area_cm2=area_cm2,
                        param_dict=live_dict,
                        fit_r2=live_r2,
                        fit_rmse=live_rmse,
                        excluded_indices=effective_exclude_indices,
                        lowfreq_cutoff_hz=lowfreq_cutoff_hz if enable_lowfreq_cut else None,
                        drt_result=drt_result,
                    )

                    pointwise_df = build_pointwise_df(
                        file_name=uploaded_file.name,
                        sam_type=sam_type,
                        substrate=substrate,
                        conc=concentration,
                        freq=freq,
                        zexp=zexp,
                        zfit=live_zfit,
                        z_drt=drt_result["z_drt"],
                    )

                    drt_df = build_drt_distribution_df(
                        file_name=uploaded_file.name,
                        sam_type=sam_type,
                        substrate=substrate,
                        conc=concentration,
                        drt_result=drt_result,
                    )

                    st.session_state["reviewed_batch_items"].append({
                        "Source_File": uploaded_file.name,
                        "Sheet_Name": sheet_name,
                        "SAM": normalize_sam_name(sam_type),
                        "Substrate": substrate,
                        "Concentration_mM": concentration,
                        "Area_cm2": area_cm2,
                        "Excluded_Indices": effective_exclude_indices,
                        "freq": freq.copy(),
                        "zexp": zexp.copy(),
                        "zfit": live_zfit.copy(),
                        "z_drt": drt_result["z_drt"].copy(),
                        "drt_result": {
                            "f_drt": drt_result["f_drt"].copy(),
                            "tau": drt_result["tau"].copy(),
                            "gamma": drt_result["gamma"].copy(),
                            "r_inf": drt_result["r_inf"],
                            "total_r": drt_result["total_r"],
                            "tau_char": drt_result["tau_char"],
                            "c_char": drt_result["c_char"],
                        },
                        "Summary_Row": summary_row,
                        "Pointwise_DF": pointwise_df.copy(),
                        "DRT_DF": drt_df.copy(),
                    })
                    st.success("Batch Queue에 추가되었습니다.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Queue 추가 실패: {e}")

            with st.expander("Loaded EIS Preview"):
                st.dataframe(df_eis, use_container_width=True)

    except Exception as e:
        st.error(f"파일 처리 실패: {e}")


# =========================================================
# 11. Batch UI
# =========================================================
st.header("2) Batch review / export")
q_items = st.session_state.get("reviewed_batch_items", [])

if not q_items:
    st.info("Queue가 비어 있습니다.")
else:
    df_raw = pd.DataFrame([i["Summary_Row"] for i in q_items])
    df_sum = build_batch_summary(df_raw)

    st.subheader("Batch Summary")
    st.dataframe(df_sum, use_container_width=True)

    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        export_nyquist = st.checkbox("Nyquist 포함", value=True)
        export_nyquist_data = st.checkbox("Nyquist/Bode pointwise 데이터 포함", value=True)
    with export_col2:
        export_bode = st.checkbox("Bode 포함", value=True)
        export_drt_data = st.checkbox("DRT 분포 데이터 포함", value=True)
    with export_col3:
        export_drt_plot = st.checkbox("DRT 플롯 포함", value=True)
        export_summary_plot = st.checkbox("Summary 플롯 포함", value=True)

    col_q1, col_q2 = st.columns([1, 1])
    with col_q1:
        if st.button("전체 Queue 비우기", use_container_width=True):
            st.session_state["reviewed_batch_items"] = []
            st.rerun()
    with col_q2:
        st.write(f"현재 Queue 항목 수: {len(q_items)}")

    with st.expander("큐 개별 항목 관리 (삭제)"):
        for i, item in enumerate(q_items):
            col_item, col_btn = st.columns([5, 1])
            with col_item:
                st.write(f"**{i+1}.** {item['Source_File']} | {item['Substrate']}/{item['SAM']} | {item['Concentration_mM']} mM")
            with col_btn:
                btn_key = f"del_{make_safe_key(i, item['Source_File'], item['Concentration_mM'])}"
                if st.button("삭제", key=btn_key):
                    st.session_state["reviewed_batch_items"].pop(i)
                    st.rerun()

    ny_p, bo_p, drt_p, su_p, bny_p = {}, {}, {}, {}, {}

    for idx, item in enumerate(q_items):
        src_name = sanitize_filename(item["Source_File"])

        if export_nyquist:
            ny_fig = make_nyquist_figure(item["zexp"], item["zfit"], item["Concentration_mM"], title=item["Source_File"], exclude_indices=item["Excluded_Indices"])
            ny_p[f"Ny_{idx}_{src_name}.png"] = fig_to_png_bytes(ny_fig)

        if export_bode:
            bo_fig = make_bode_figure(item["freq"], item["zexp"], item["zfit"], title=item["Source_File"], z_drt=item["z_drt"])
            bo_p[f"Bo_{idx}_{src_name}.png"] = fig_to_png_bytes(bo_fig)

        if export_drt_plot:
            drt_fig = make_drt_figure(item["drt_result"]["f_drt"], item["drt_result"]["gamma"], title=item["Source_File"])
            drt_p[f"DRT_{idx}_{src_name}.png"] = fig_to_png_bytes(drt_fig)

    if export_summary_plot:
        for _, pair in df_sum[["SAM_INTERNAL", "Substrate"]].drop_duplicates().iterrows():
            f1 = make_summary_plot(df_sum, pair["SAM_INTERNAL"], pair["Substrate"])
            if f1 is not None:
                su_p[f"Sum_{pair['Substrate']}_{pair['SAM_INTERNAL']}.png"] = fig_to_png_bytes(f1)

            f2 = make_batch_nyquist_panel_from_queue(q_items, pair["SAM_INTERNAL"], pair["Substrate"])
            if f2 is not None and export_nyquist:
                bny_p[f"BatchNy_{pair['Substrate']}_{pair['SAM_INTERNAL']}.png"] = fig_to_png_bytes(f2)

    export_sheets = {}
    if export_nyquist_data or export_bode:
        export_sheets["Pointwise"] = pd.concat([i["Pointwise_DF"] for i in q_items], ignore_index=True)
    if export_drt_data:
        export_sheets["DRT_Distribution"] = pd.concat([i["DRT_DF"] for i in q_items], ignore_index=True)

    exc = build_batch_excel_bytes(df_raw, df_sum, export_sheets)

    png_groups = {}
    if export_nyquist:
        png_groups["Nyquist"] = ny_p
    if export_bode:
        png_groups["Bode"] = bo_p
    if export_drt_plot:
        png_groups["DRT"] = drt_p
    if export_summary_plot:
        png_groups["Summary"] = su_p
    if export_nyquist and export_summary_plot:
        png_groups["BatchNy"] = bny_p

    st.download_button(
        "ZIP 다운로드 (선택 데이터셋 + PNG)",
        data=build_batch_zip_bytes(exc, png_groups),
        file_name=f"EIS_Batch_{ts}.zip",
        mime="application/zip",
        use_container_width=True,
    )
