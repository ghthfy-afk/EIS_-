import io
import os
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import least_squares


# =========================================================
# 1. 기본 설정
# =========================================================
st.set_page_config(page_title="EIS Interactive Fitter + Batch Review", layout="wide")
st.title("🔬 EIS Interactive / Batch Circuit Fitter")
st.caption("B-1 / C-1 equivalent circuit fitting with manual inputs, outlier exclusion, reviewed batch queue, and export.")

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


# =========================================================
# 3. 회로 수학
# =========================================================
def Z_R(R, w):
    return np.full_like(w, complex(R, 0.0), dtype=np.complex128)


def Z_C(C, w):
    return 1.0 / (1j * w * max(float(C), 1e-30))


def Z_CPE(T, P, w):
    t = max(float(T), 1e-30)
    p = min(max(float(P), 0.0), 1.0)
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
# 4. 데이터 읽기 & 유틸
# =========================================================
def normalize_column_name(col):
    c = str(col).strip().lower()
    c = c.replace(" ", "")
    c = c.replace("_", "")
    c = c.replace("-", "")
    return c


def find_matching_column(columns, aliases):
    normalized_map = {normalize_column_name(c): c for c in columns}
    for alias in aliases:
        if normalize_column_name(alias) in normalized_map:
            return normalized_map[normalize_column_name(alias)]
    return None


def standardize_eis_columns(df):
    freq_col = find_matching_column(
        df.columns,
        ["Frequency(Hz)", "Frequency", "Freq", "f", "Hz"]
    )
    zre_col = find_matching_column(
        df.columns,
        ["Zre(ohm)", "Zre", "ReZ", "Zreal", "RealZ"]
    )
    zim_col = find_matching_column(
        df.columns,
        ["Zim(ohm)", "Zim", "ImZ", "Zimag", "ImagZ"]
    )
    neg_zim_col = find_matching_column(
        df.columns,
        ["-Zim(ohm)", "-Zim", "NegZim", "-ImZ", "-ImagZ"]
    )

    if freq_col is None or zre_col is None:
        raise ValueError("Frequency 또는 Zre 컬럼을 찾지 못했습니다.")

    if zim_col is None and neg_zim_col is None:
        raise ValueError("Zim 또는 -Zim 컬럼을 찾지 못했습니다.")

    out = pd.DataFrame()
    out["Frequency(Hz)"] = pd.to_numeric(df[freq_col], errors="coerce")
    out["Zre(ohm)"] = pd.to_numeric(df[zre_col], errors="coerce")

    if zim_col is not None:
        out["Zim(ohm)"] = pd.to_numeric(df[zim_col], errors="coerce")
    else:
        out["Zim(ohm)"] = -pd.to_numeric(df[neg_zim_col], errors="coerce")

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


def read_eis_from_uploaded(uploaded_file):
    xl = pd.ExcelFile(uploaded_file)
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
    stem = os.path.splitext(name)[0].upper().replace("[", "_").replace("]", "_")
    sam = "B_1" if any(k in stem for k in ["ODPA", "B-1", "B_1"]) else ("C_1" if any(k in stem for k in ["BTA", "C-1", "C_1"]) else None)
    sub = "CU" if re.search(r"(^|[_\-\s])CU([_\-\s]|$)", stem) else ("CO" if re.search(r"(^|[_\-\s])CO([_\-\s]|$)", stem) else None)
    m = re.search(r"(\d+(?:\.\d+)?)\s*MM", stem)
    conc = float(m.group(1)) if m else None
    return sam, sub, conc


# =========================================================
# 5. 피팅 로직
# =========================================================
def build_initial_guess(freq, zexp, sam_type):
    sam_type = normalize_sam_name(sam_type)
    zre = np.asarray(zexp.real, dtype=float)
    zim = np.asarray(zexp.imag, dtype=float)
    zmag = np.abs(zexp)

    Rs_guess = max(1e-3, float(np.min(zre)))
    Rmax = max(float(np.max(zre)), float(np.max(zmag)))
    Rspan = max(1.0, Rmax - Rs_guess)

    idx_peak = np.argmax(np.abs(zim))
    f_peak = max(float(freq[idx_peak]), 1e-6)
    w_peak = 2 * np.pi * f_peak
    C_guess = max(1e-9, min(1e-4, 1.0 / (w_peak * max(Rspan, 1.0))))

    if sam_type == "B_1":
        return np.array([
            C_guess, 0.95,
            0.15 * Rspan, 0.20 * Rspan,
            1e-5, 0.90,
            0.45 * Rspan, 1e-6, 0.85,
            Rs_guess
        ], dtype=float)

    return np.array([
        Rs_guess,
        10.0,
        C_guess,
        0.35 * Rspan,
        1e-5,
        0.90,
        0.65 * Rspan
    ], dtype=float)


def build_exclude_indices(freq, manual_exclude_indices=None, exclude_below_100hz=False):
    idx_set = set()

    if manual_exclude_indices:
        idx_set.update([i for i in manual_exclude_indices if 0 <= i < len(freq)])

    if exclude_below_100hz:
        idx_set.update(np.where(np.asarray(freq, dtype=float) < 100.0)[0].tolist())

    return sorted(idx_set)


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

    f_fit = freq.copy()
    z_fit = zexp.copy()

    if exclude_indices:
        mask = np.ones(len(f_fit), dtype=bool)
        valid_idx = [i for i in exclude_indices if 0 <= i < len(f_fit)]
        mask[valid_idx] = False
        f_fit = f_fit[mask]
        z_fit = z_fit[mask]

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
        max_nfev=20000
    )

    p = res.x
    zfit_full = model_func(p, 2 * np.pi * freq)
    zfit_eval = model_func(p, w)

    ss_res = np.sum((zfit_eval.real - z_fit.real) ** 2 + (zfit_eval.imag - z_fit.imag) ** 2)
    ss_tot = np.sum((z_fit.real - np.mean(z_fit.real)) ** 2 + (z_fit.imag - np.mean(z_fit.imag)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(np.abs(zfit_eval - z_fit) ** 2))

    return {k: float(v) for k, v in zip(names, p)}, zfit_full, float(r2), float(rmse), res


def evaluate_current_params(freq, zexp, sam_type, params, exclude_indices=None):
    names, _, _, model_func = get_model_info(sam_type)

    freq = np.asarray(freq, dtype=float)
    zexp = np.asarray(zexp, dtype=np.complex128)
    params = np.asarray(params, dtype=float)

    zfit_full = model_func(params, 2 * np.pi * freq)

    f_eval = freq.copy()
    z_eval = zexp.copy()
    zfit_eval = zfit_full.copy()

    if exclude_indices:
        mask = np.ones(len(freq), dtype=bool)
        valid_idx = [i for i in exclude_indices if 0 <= i < len(freq)]
        mask[valid_idx] = False
        f_eval = freq[mask]
        z_eval = zexp[mask]
        zfit_eval = zfit_full[mask]

    if len(f_eval) == 0:
        return {k: float(v) for k, v in zip(names, params)}, zfit_full, 0.0, np.inf

    rmse = np.sqrt(np.mean(np.abs(zfit_eval - z_eval) ** 2))
    ss_res = np.sum(np.abs(zfit_eval - z_eval) ** 2)
    ss_tot = np.sum(np.abs(z_eval - np.mean(z_eval)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {k: float(v) for k, v in zip(names, params)}, zfit_full, float(r2), float(rmse)


# =========================================================
# 6. 시각화 함수
# =========================================================
def compute_nyquist_limits_positive(zexp_list, zfit_list=None, pad_ratio=0.05):
    """
    Nyquist plot limits for 1st quadrant only:
    - x >= 0
    - y >= 0, where y = -Zim
    - x/y same upper limit
    """
    if zfit_list is None:
        zfit_list = []

    xs = []
    ys = []

    for z in zexp_list:
        if z is None or len(z) == 0:
            continue
        x = np.asarray(z.real, dtype=float)
        y = np.asarray(-z.imag, dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size > 0:
            xs.append(x)
        if y.size > 0:
            ys.append(y)

    for z in zfit_list:
        if z is None or len(z) == 0:
            continue
        x = np.asarray(z.real, dtype=float)
        y = np.asarray(-z.imag, dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size > 0:
            xs.append(x)
        if y.size > 0:
            ys.append(y)

    if not xs or not ys:
        return None

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)

    x_max = max(float(np.nanmax(x_all)), 1.0)
    y_max = max(float(np.nanmax(y_all)), 1.0)

    upper = max(x_max, y_max)
    upper = upper * (1.0 + pad_ratio)

    return (0.0, upper), (0.0, upper)


def apply_equal_nyquist_axes_positive(ax, zexp_list, zfit_list=None, pad_ratio=0.05):
    lims = compute_nyquist_limits_positive(zexp_list, zfit_list=zfit_list, pad_ratio=pad_ratio)
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

    ax.plot(zexp.real[mask], -zexp.imag[mask], "o", alpha=0.6, color="navy", label="Exp")

    if exclude_indices:
        ax.plot(zexp.real[~mask], -zexp.imag[~mask], "x", color="orange", label="Excluded")
        for i in valid_idx:
            ax.annotate(str(i), (zexp.real[i], -zexp.imag[i]), color="orange")

    ax.plot(zfit.real, -zfit.imag, "-", color="red", label="Fit")
    ax.set_xlabel("Zre (ohm)")
    ax.set_ylabel("-Zim (ohm)")
    ax.set_title(f"{title} | {concentration} mM")
    ax.legend()
    ax.grid(True, alpha=0.3)

    apply_equal_nyquist_axes_positive(ax, [zexp], [zfit], pad_ratio=0.05)
    return fig


def make_bode_figure(freq, zexp, zfit, title="Bode Plot"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.semilogx(freq, np.abs(zexp), "o", label="Exp")
    ax1.semilogx(freq, np.abs(zfit), "-", label="Fit")
    ax1.set_ylabel("|Z| (ohm)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.semilogx(freq, np.degrees(np.angle(zexp)), "o", label="Exp")
    ax2.semilogx(freq, np.degrees(np.angle(zfit)), "-", label="Fit")
    ax2.set_xlabel("Freq (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def make_summary_plot(df_sum, sam_type, substrate, x_log=False):
    sam_type = normalize_sam_name(sam_type)
    plot_df = df_sum[
        (df_sum["SAM_INTERNAL"] == sam_type) &
        (df_sum["Substrate"] == substrate)
    ].sort_values("Concentration_mM")

    if plot_df.empty:
        return None

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Conc (mM)")
    ax1.set_ylabel("R Index (Ohm cm2)")
    ax1.set_yscale("log")
    if x_log:
        ax1.set_xscale("log")

    # 1. 선 제거 (fmt="o" 사용)
    ax1.errorbar(
        plot_df["Concentration_mM"],
        plot_df["Total_R_Index_Norm_mean"],
        yerr=plot_df["Total_R_Index_Norm_std"],
        fmt="o",
        label="R Index"
    )

    # 각 포인트에 R Index 값 라벨링
    for _, row in plot_df.iterrows():
        val = row["Total_R_Index_Norm_mean"]
        if pd.notna(val):
            ax1.annotate(f"{val:.2e}",
                         (row["Concentration_mM"], val),
                         textcoords="offset points",
                         xytext=(0, 10), # 점 위쪽으로 배치
                         ha="center",
                         fontsize=9)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Thickness (nm)", color="red")
    
    # 1. 선 제거 (fmt="s" 사용)
    ax2.errorbar(
        plot_df["Concentration_mM"],
        plot_df["Thickness_mean"],
        yerr=plot_df["Thickness_std"],
        fmt="s",
        color="red",
        label="Thickness"
    )

    # 각 포인트에 Thickness 값 라벨링
    for _, row in plot_df.iterrows():
        val = row["Thickness_mean"]
        if pd.notna(val):
            ax2.annotate(f"{val:.1f}",
                         (row["Concentration_mM"], val),
                         textcoords="offset points",
                         xytext=(0, -15), # R Index와 겹치지 않게 점 아래쪽으로 배치
                         ha="center",
                         fontsize=9,
                         color="red")

    plt.title(f"Analysis: {substrate} / {display_sam_name(sam_type)}")
    ax1.grid(True, alpha=0.3)
    return fig


def make_batch_nyquist_panel_from_queue(queue_items, sam_type, substrate):
    sam_type = normalize_sam_name(sam_type)
    filtered = sorted(
        [i for i in queue_items if i["SAM"] == sam_type and i["Substrate"] == substrate],
        key=lambda x: x["Concentration_mM"]
    )

    if not filtered:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    zexp_list = []
    zfit_list = []

    for item in filtered:
        ax.plot(item["zexp"].real, -item["zexp"].imag, "o", alpha=0.4, label=f"{item['Concentration_mM']}mM")
        ax.plot(item["zfit"].real, -item["zfit"].imag, "-", alpha=0.8)
        zexp_list.append(item["zexp"])
        zfit_list.append(item["zfit"])

    ax.set_title(f"Batch Nyquist: {substrate} / {display_sam_name(sam_type)}")
    ax.set_xlabel("Zre (ohm)")
    ax.set_ylabel("-Zim (ohm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    apply_equal_nyquist_axes_positive(ax, zexp_list, zfit_list, pad_ratio=0.05)
    return fig


# =========================================================
# 7. 결과 및 엑셀 빌더
# =========================================================
def classify_status(cpe_p):
    if cpe_p >= 0.9:
        return "Good"
    if cpe_p >= 0.85:
        return "Borderline"
    return "Warning"


def build_summary_row(file_name, sheet_name, substrate, sam_type, concentration, area_cm2, param_dict, fit_r2, fit_rmse, excluded_indices, exclude_below_100hz):
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
        "Exclude_Below_100Hz": bool(exclude_below_100hz),
    }
    row.update(param_dict)

    if sam_type == "C_1":
        r_tot = row["R1_inner"] + row["R2_interface"]
        c_t = row["CPE1_T_outer"]
        c_p = row["CPE1_P_outer"]
    else:
        # 3. R_ct를 제외하고 방어력에 기여하는 저항만 합산 (R_int + R_sam)
        r_tot = row["R_int"] + row["R_sam"]
        c_t = row["CPE_sam_T"]
        c_p = row["CPE_sam_P"]

    row["Total_R_Index_Norm_Ohm_cm2"] = r_tot * area_cm2
    row["Status"] = classify_status(c_p)

    if row["Status"] == "Good" and c_t > 0 and area_cm2 > 0:
        row["Thickness_nm"] = round((E0 * EPSILON_R[sam_type] / (c_t / area_cm2)) * 1e7, 2)
    else:
        row["Thickness_nm"] = np.nan

    return row


def build_pointwise_df(file_name, sam_type, sub, conc, freq, zexp, zfit):
    return pd.DataFrame({
        "Source": file_name,
        "SAM": display_sam_name(sam_type),
        "Substrate": sub,
        "Concentration_mM": conc,
        "Freq": freq,
        "Zre_exp": zexp.real,
        "Zim_exp": zexp.imag,
        "Zre_fit": zfit.real,
        "Zim_fit": zfit.imag,
    })


def build_batch_summary(df_raw):
    return df_raw.groupby(["Substrate", "SAM", "SAM_INTERNAL", "Concentration_mM"]).agg(
        n=("Concentration_mM", "count"),
        Fit_R2_mean=("Fit_R2", "mean"),
        Total_R_Index_Norm_mean=("Total_R_Index_Norm_Ohm_cm2", "mean"),
        Total_R_Index_Norm_std=("Total_R_Index_Norm_Ohm_cm2", lambda x: x.std() if len(x) > 1 else 0),
        Thickness_mean=("Thickness_nm", "mean"),
        Thickness_std=("Thickness_nm", lambda x: x.std() if len(x) > 1 else 0),
        Status=("Status", "first")
    ).reset_index()


def build_batch_excel_bytes(df_raw, df_sum, df_points):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="Raw_Fit", index=False)
        df_sum.to_excel(writer, sheet_name="Summary", index=False)
        df_points.to_excel(writer, sheet_name="Pointwise", index=False)
    buf.seek(0)
    return buf


def sanitize_filename(name):
    name = str(name)
    return re.sub(r'[\\/*?:"<>|]+', "_", name)


def build_batch_zip_bytes(excel_bytes, nyquist_pngs, bode_pngs, summary_pngs, batch_nyquist_pngs):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"Report_{ts}.xlsx", excel_bytes.getvalue())

        for folder_name, png_dict in [
            ("Nyquist", nyquist_pngs),
            ("Bode", bode_pngs),
            ("Summary", summary_pngs),
            ("BatchNy", batch_nyquist_pngs),
        ]:
            for name, data in png_dict.items():
                zf.writestr(f"{folder_name}/{sanitize_filename(name)}", data)

    zip_buf.seek(0)
    return zip_buf


# =========================================================
# 8. 메인 UI (Single Review)
# =========================================================
st.header("1) Single-file review")
uploaded_file = st.file_uploader("xlsx 파일 업로드", type=["xlsx"], key="single_uploader")

if uploaded_file:
    file_token = f"{uploaded_file.name}_{uploaded_file.size}"

    try:
        sheet_name, df_eis, freq, zexp = read_eis_from_uploaded(uploaded_file)
        sam_guess, substrate_guess, conc_guess = parse_metadata_from_filename(uploaded_file.name)

        col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)

        with col_meta1:
            sam_display = st.selectbox(
                "Model",
                ["B-1 (3-tc)", "C-1 (Warburg)"],
                index=0 if sam_guess != "C_1" else 1,
                key=f"{file_token}_mod"
            )
            sam_type = "B_1" if "B-1" in sam_display else "C_1"

        with col_meta2:
            substrate = st.selectbox(
                "Substrate",
                ["CU", "CO"],
                index=0 if substrate_guess != "CO" else 1,
                key=f"{file_token}_sub"
            )

        with col_meta3:
            concentration = st.number_input(
                "Conc (mM)",
                value=float(conc_guess) if conc_guess is not None else 0.0,
                key=f"{file_token}_conc"
            )

        with col_meta4:
            area_cm2 = st.number_input(
                "Area (cm2)",
                value=0.14,
                format="%.4f",
                key=f"{file_token}_area"
            )

        names, lb, ub, _ = get_model_info(sam_type)
        default_guess = build_initial_guess(freq, zexp, sam_type)

        if st.session_state.get(f"{file_token}_last_mod") != sam_type:
            st.session_state[f"{file_token}_last_mod"] = sam_type
            for n, v in zip(names, default_guess):
                st.session_state[f"{file_token}_{sam_type}_{n}_val"] = float(v)
            st.session_state[f"{file_token}_{sam_type}_cur_fit"] = default_guess.tolist()
            st.session_state[f"{file_token}_wver"] = st.session_state.get(f"{file_token}_wver", 0) + 1
            st.rerun()

        for n, v in zip(names, default_guess):
            key_name = f"{file_token}_{sam_type}_{n}_val"
            if key_name not in st.session_state:
                st.session_state[key_name] = float(v)

        if f"{file_token}_{sam_type}_cur_fit" not in st.session_state:
            st.session_state[f"{file_token}_{sam_type}_cur_fit"] = default_guess.tolist()

        if f"{file_token}_wver" not in st.session_state:
            st.session_state[f"{file_token}_wver"] = 0

        outlier_key = f"{file_token}_{sam_type}_out"
        exclude_lowfreq_key = f"{file_token}_{sam_type}_exclude_below_100hz"

        with st.expander("Outlier 설정"):
            sel_out = st.multiselect(
                "제외 인덱스",
                options=list(range(len(freq))),
                default=st.session_state.get(outlier_key, []),
                key=f"ms_{outlier_key}"
            )
            st.session_state[outlier_key] = sel_out

            exclude_below_100hz = st.checkbox(
                "피팅에서 100 Hz 미만 영역 제외",
                value=st.session_state.get(exclude_lowfreq_key, False),
                key=exclude_lowfreq_key
            )

        effective_exclude_indices = build_exclude_indices(
            freq=freq,
            manual_exclude_indices=st.session_state.get(outlier_key, []),
            exclude_below_100hz=st.session_state.get(exclude_lowfreq_key, False)
        )

        left, right = st.columns([1.2, 1.8])

        with left:
            st.subheader("Manual Parameter Input")

            curr_params = []
            curr_lb = lb.copy()
            curr_ub = ub.copy()

            for i, name in enumerate(names):
                s_key = f"{file_token}_{sam_type}_{name}_val"
                is_p = "_P" in name

                col_input, col_fix = st.columns([4, 1])

                with col_fix:
                    is_locked = st.checkbox("Fix", key=f"fix_{s_key}")

                with col_input:
                    val = st.number_input(
                        f"{name}",
                        value=float(st.session_state[s_key]),
                        format="%.4e" if not is_p else "%.4f",
                        key=f"in_{s_key}_{st.session_state[f'{file_token}_wver']}"
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
            st.caption(f"현재 피팅 제외 포인트 수: {len(effective_exclude_indices)}")

            if st.button("🚀 현재 값에서 Auto Fit 시작", type="primary", use_container_width=True):
                try:
                    p_fit, _, _, _, _ = fit_eis(
                        freq=freq,
                        zexp=zexp,
                        sam_type=sam_type,
                        x0=curr_params,
                        exclude_indices=effective_exclude_indices,
                        custom_bounds=(curr_lb, curr_ub)
                    )

                    for n, v in p_fit.items():
                        st.session_state[f"{file_token}_{sam_type}_{n}_val"] = float(v)

                    st.session_state[f"{file_token}_{sam_type}_cur_fit"] = [float(p_fit[n]) for n in names]
                    st.session_state[f"{file_token}_wver"] += 1
                    st.success("Auto Fit 완료")
                    st.rerun()

                except Exception as e:
                    st.error(f"Auto Fit 실패: {e}")

            if st.button("✅ Batch Queue에 결과 추가", use_container_width=True):
                try:
                    live_dict, live_zfit, live_r2, live_rmse = evaluate_current_params(
                        freq=freq,
                        zexp=zexp,
                        sam_type=sam_type,
                        params=curr_params,
                        exclude_indices=effective_exclude_indices
                    )

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
                        exclude_below_100hz=st.session_state.get(exclude_lowfreq_key, False)
                    )

                    pointwise_df = build_pointwise_df(
                        file_name=uploaded_file.name,
                        sam_type=sam_type,
                        sub=substrate,
                        conc=concentration,
                        freq=freq,
                        zexp=zexp,
                        zfit=live_zfit
                    )

                    st.session_state["reviewed_batch_items"].append({
                        "Source_File": uploaded_file.name,
                        "Sheet_Name": sheet_name,
                        "SAM": normalize_sam_name(sam_type),
                        "Substrate": substrate,
                        "Concentration_mM": concentration,
                        "Area_cm2": area_cm2,
                        "Excluded_Indices": effective_exclude_indices,
                        "Exclude_Below_100Hz": st.session_state.get(exclude_lowfreq_key, False),
                        "freq": freq.copy(),
                        "zexp": zexp.copy(),
                        "zfit": live_zfit.copy(),
                        "Summary_Row": summary_row,
                        "Pointwise_DF": pointwise_df.copy(),
                    })

                    st.success("Batch Queue에 추가되었습니다.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Queue 추가 실패: {e}")

        with right:
            live_dict, live_zfit, live_r2, live_rmse = evaluate_current_params(
                freq=freq,
                zexp=zexp,
                sam_type=sam_type,
                params=curr_params,
                exclude_indices=effective_exclude_indices
            )

            st.subheader(f"Fit Quality (Live) | R²: {live_r2:.4f}, RMSE: {live_rmse:.2e}")

            t1, t2 = st.tabs(["Nyquist Plot", "Bode Plot"])

            with t1:
                fig1 = make_nyquist_figure(
                    zexp=zexp,
                    zfit=live_zfit,
                    concentration=concentration,
                    title=f"{uploaded_file.name}",
                    exclude_indices=effective_exclude_indices
                )
                st.pyplot(fig1, use_container_width=True)

            with t2:
                fig2 = make_bode_figure(freq, zexp, live_zfit)
                st.pyplot(fig2, use_container_width=True)

            st.subheader("Current Live Parameters")
            st.dataframe(
                pd.DataFrame(live_dict.items(), columns=["Parameter", "Value"]),
                use_container_width=True,
                height=350
            )

            with st.expander("Loaded EIS Preview"):
                st.dataframe(df_eis, use_container_width=True)

    except Exception as e:
        st.error(f"파일 처리 실패: {e}")


# =========================================================
# 9. Batch UI
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

    col_q1, col_q2 = st.columns([1, 1])

    with col_q1:
        if st.button("Queue 비우기", use_container_width=True):
            st.session_state["reviewed_batch_items"] = []
            st.rerun()

    with col_q2:
        st.write(f"현재 Queue 항목 수: {len(q_items)}")

    ny_p = {}
    bo_p = {}
    su_p = {}
    bny_p = {}

    for idx, item in enumerate(q_items):
        src_name = sanitize_filename(item["Source_File"])
        ny_name = f"Ny_{idx}_{src_name}.png"
        bo_name = f"Bo_{idx}_{src_name}.png"

        ny_fig = make_nyquist_figure(
            item["zexp"],
            item["zfit"],
            item["Concentration_mM"],
            title=item["Source_File"],
            exclude_indices=item["Excluded_Indices"]
        )
        bo_fig = make_bode_figure(item["freq"], item["zexp"], item["zfit"], title=item["Source_File"])

        ny_p[ny_name] = fig_to_png_bytes(ny_fig)
        bo_p[bo_name] = fig_to_png_bytes(bo_fig)

    for _, pair in df_sum[["SAM_INTERNAL", "Substrate"]].drop_duplicates().iterrows():
        f1 = make_summary_plot(df_sum, pair["SAM_INTERNAL"], pair["Substrate"])
        if f1 is not None:
            su_p[f"Sum_{pair['Substrate']}_{pair['SAM_INTERNAL']}.png"] = fig_to_png_bytes(f1)

        f2 = make_batch_nyquist_panel_from_queue(q_items, pair["SAM_INTERNAL"], pair["Substrate"])
        if f2 is not None:
            bny_p[f"BatchNy_{pair['Substrate']}_{pair['SAM_INTERNAL']}.png"] = fig_to_png_bytes(f2)

    df_points = pd.concat([i["Pointwise_DF"] for i in q_items], ignore_index=True)
    exc = build_batch_excel_bytes(df_raw, df_sum, df_points)

    st.download_button(
        "📥 ZIP 다운로드 (Excel + PNG)",
        data=build_batch_zip_bytes(exc, ny_p, bo_p, su_p, bny_p),
        file_name=f"EIS_Batch_{ts}.zip",
        mime="application/zip",
        use_container_width=True
    )
