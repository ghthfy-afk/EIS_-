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
st.caption("B-1 / C-1 equivalent circuit fitting with manual sliders, outlier exclusion, reviewed batch queue, and export.")

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

    # 과거 명칭까지 모두 허용
    if n == "ODPA":
        return "B_1"
    if n == "BTA":
        return "C_1"

    if n in ["B_1", "C_1"]:
        return n

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
    C = max(float(C), 1e-30)
    return 1.0 / (1j * w * C)


def Z_CPE(T, P, w):
    T = max(float(T), 1e-30)
    P = min(max(float(P), 0.0), 1.0)
    return 1.0 / (T * (1j * w) ** P)


def Z_Ws_sigma(sigma, w):
    sigma = max(float(sigma), 1e-30)
    return sigma / np.sqrt(1j * w)


def Z_parallel(Z1, Z2):
    return 1.0 / (1.0 / Z1 + 1.0 / Z2)


# C-1 회로
def Z_C_1(params, w):
    Rs, Ws1_sigma, C1, R1, T1, P1, R2 = params
    Z0 = Z_parallel(Z_Ws_sigma(Ws1_sigma, w), Z_C(C1, w))
    Z1 = Z0 + Z_R(R1, w)
    Z2 = Z_parallel(Z1, Z_CPE(T1, P1, w))
    Z3 = Z2 + Z_R(R2, w)
    Z_total = Z3 + Z_R(Rs, w)
    return Z_total


# B-1 회로
def Z_B_1(params, w):
    C_dl, R_ct, R_int, T_int, P_int, R_sam, T_sam, P_sam, Rs_sol = params
    Z0 = Z_parallel(Z_C(C_dl, w), Z_R(R_ct, w))
    Z1 = Z0 + Z_R(R_int, w)
    Z2 = Z_parallel(Z1, Z_CPE(T_int, P_int, w))
    Z3 = Z2 + Z_R(R_sam, w)
    Z4 = Z_parallel(Z3, Z_CPE(T_sam, P_sam, w))
    Z_total = Z4 + Z_R(Rs_sol, w)
    return Z_total


def get_model_info(sam_type):
    sam_type = normalize_sam_name(sam_type)

    if sam_type == "B_1":
        names = ["C_dl", "R_ct", "R_int", "CPE_int_T", "CPE_int_P", "R_sam", "CPE_sam_T", "CPE_sam_P", "Rs_sol"]
        lower = np.array([1e-12, 1e-6, 1e-6, 1e-12, 0.0, 1e-6, 1e-12, 0.0, 1e-6], dtype=float)
        upper = np.array([1e-1, 1e12, 1e12, 1e-1, 1.0, 1e12, 1e-1, 1.0, 1e12], dtype=float)
        return names, lower, upper, Z_B_1
    else:
        names = ["Rs_sol", "W1_Sigma", "C1_inner", "R1_inner", "CPE1_T_outer", "CPE1_P_outer", "R2_interface"]
        lower = np.array([1e-6, 1e-9, 1e-12, 1e-6, 1e-12, 0.0, 1e-6], dtype=float)
        upper = np.array([1e12, 1e9, 1e-1, 1e12, 1e-1, 1.0, 1e12], dtype=float)
        return names, lower, upper, Z_C_1


# =========================================================
# 4. 데이터 읽기
# =========================================================
def pick_data_sheet(xl):
    candidate_sheets = [s for s in xl.sheet_names if str(s).upper().startswith("DATA")]
    if not candidate_sheets:
        candidate_sheets = xl.sheet_names

    best_sheet = None
    best_count = -1

    for s in candidate_sheets:
        try:
            df_try = xl.parse(s)
            cols = set(df_try.columns.astype(str))
            needed = {"Frequency(Hz)", "Zre(ohm)", "Zim(ohm)"}
            if needed.issubset(cols):
                usable_rows = int(df_try[list(needed)].dropna().shape[0])
                if usable_rows > best_count:
                    best_count = usable_rows
                    best_sheet = s
        except Exception:
            continue

    if best_sheet is None:
        raise ValueError("적절한 Data 시트를 찾지 못했습니다.")
    return best_sheet


def read_eis_from_uploaded(uploaded_file):
    xl = pd.ExcelFile(uploaded_file)
    sheet = pick_data_sheet(xl)
    df = xl.parse(sheet)

    needed = ["Frequency(Hz)", "Zre(ohm)", "Zim(ohm)"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    out = df[needed].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna().copy()
    out = out[out["Frequency(Hz)"] > 0].copy()

    if out.empty:
        raise ValueError("유효한 EIS 데이터가 없습니다.")

    out = out.sort_values(by="Frequency(Hz)", ascending=False).reset_index(drop=True)

    freq = out["Frequency(Hz)"].to_numpy(dtype=float)
    zre = out["Zre(ohm)"].to_numpy(dtype=float)
    zim = out["Zim(ohm)"].to_numpy(dtype=float)
    zexp = zre + 1j * zim

    return sheet, out, freq, zexp


def parse_metadata_from_filename(name):
    stem = os.path.splitext(name)[0].upper()
    text = stem.replace("[", "_").replace("]", "_")

    sam = None
    substrate = None
    concentration = None

    # 구명칭 + 신명칭 모두 허용
    if any(k in text for k in ["ODPA", "B-1", "B_1"]):
        sam = "B_1"
    elif any(k in text for k in ["BTA", "C-1", "C_1"]):
        sam = "C_1"

    if re.search(r"(^|[_\-\s])CU([_\-\s]|$)", text):
        substrate = "CU"
    elif re.search(r"(^|[_\-\s])CO([_\-\s]|$)", text):
        substrate = "CO"

    m = re.search(r"(\d+(?:\.\d+)?)\s*MM", text)
    if m:
        concentration = float(m.group(1))

    return sam, substrate, concentration


# =========================================================
# 5. 피팅
# =========================================================
def build_initial_guess(freq, zexp, sam_type):
    sam_type = normalize_sam_name(sam_type)

    zmag = np.abs(zexp)
    Rs_guess = max(1e-3, float(np.min(zmag)))
    Rspan = max(1.0, float(np.max(zmag) - np.min(zmag)))
    C_guess = 1e-6
    CPE_T_guess = 1e-5
    P_guess = 0.90
    P2_guess = 0.80
    Ws_sigma_guess = 10.0

    if sam_type == "B_1":
        x0 = np.array([
            C_guess,
            max(1.0, 0.10 * Rspan),
            max(1.0, 0.20 * Rspan),
            CPE_T_guess,
            P_guess,
            max(1.0, 0.30 * Rspan),
            1e-6,
            P2_guess,
            Rs_guess,
        ], dtype=float)
    else:
        x0 = np.array([
            Rs_guess,
            Ws_sigma_guess,
            C_guess,
            max(1.0, 0.30 * Rspan),
            CPE_T_guess,
            P_guess,
            max(1.0, 0.70 * Rspan),
        ], dtype=float)

    return x0


def residuals(params, w, zexp, model_func):
    zfit = model_func(params, w)
    scale = np.maximum(np.abs(zexp), 1.0)
    r_re = (zfit.real - zexp.real) / scale
    r_im = (zfit.imag - zexp.imag) / scale
    return np.concatenate([r_re, r_im])


def fit_eis(freq, zexp, sam_type, x0, exclude_indices=None):
    names, lb, ub, model_func = get_model_info(sam_type)

    freq_fit = np.array(freq, dtype=float).copy()
    zexp_fit = np.array(zexp, dtype=np.complex128).copy()

    if exclude_indices is not None and len(exclude_indices) > 0:
        exclude_indices = sorted(set(int(i) for i in exclude_indices if 0 <= int(i) < len(freq_fit)))
        mask = np.ones(len(freq_fit), dtype=bool)
        mask[exclude_indices] = False
        freq_fit = freq_fit[mask]
        zexp_fit = zexp_fit[mask]

    if len(freq_fit) < 5:
        raise ValueError("제외 후 남은 데이터 포인트가 너무 적습니다. 최소 5개 이상 필요합니다.")

    w = 2 * np.pi * freq_fit

    x0 = np.array(x0, dtype=float)
    x0 = np.minimum(np.maximum(x0, lb * 1.001), ub / 1.001)

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lb, ub),
        args=(w, zexp_fit, model_func),
        method="trf",
        max_nfev=30000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    p = result.x
    zfit_full = model_func(p, 2 * np.pi * np.array(freq, dtype=float))

    zfit_eval = model_func(p, 2 * np.pi * freq_fit)
    ss_res = np.sum((zfit_eval.real - zexp_fit.real) ** 2 + (zfit_eval.imag - zexp_fit.imag) ** 2)
    zmean = np.mean(zexp_fit)
    ss_tot = np.sum((zexp_fit.real - zmean.real) ** 2 + (zexp_fit.imag - zmean.imag) ** 2)
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((zfit_eval.real - zexp_fit.real) ** 2 + (zfit_eval.imag - zexp_fit.imag) ** 2)))

    param_dict = {k: float(v) for k, v in zip(names, p)}
    return param_dict, zfit_full, float(r2), rmse, result


def evaluate_current_params(freq, zexp, sam_type, params):
    names, _, _, model_func = get_model_info(sam_type)
    w = 2 * np.pi * freq
    zfit = model_func(np.array(params, dtype=float), w)

    ss_res = np.sum((zfit.real - zexp.real) ** 2 + (zfit.imag - zexp.imag) ** 2)
    zmean = np.mean(zexp)
    ss_tot = np.sum((zexp.real - zmean.real) ** 2 + (zexp.imag - zmean.imag) ** 2)
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((zfit.real - zexp.real) ** 2 + (zfit.imag - zexp.imag) ** 2)))

    param_dict = {k: float(v) for k, v in zip(names, params)}
    return param_dict, zfit, float(r2), rmse


# =========================================================
# 6. 시각화
# =========================================================
def make_nyquist_figure(zexp, zfit, concentration, title="Nyquist Plot", exclude_indices=None):
    fig, ax = plt.subplots(figsize=(7, 5))

    zexp = np.array(zexp)

    if exclude_indices is None:
        exclude_indices = []
    exclude_indices = sorted(set(int(i) for i in exclude_indices if 0 <= int(i) < len(zexp)))

    keep_mask = np.ones(len(zexp), dtype=bool)
    if exclude_indices:
        keep_mask[exclude_indices] = False

    ax.plot(
        zexp.real[keep_mask], -zexp.imag[keep_mask],
        "o",
        markersize=5,
        alpha=0.6,
        color="navy",
        linestyle="none",
        label=f"{concentration} mM (Used Exp)"
    )

    if exclude_indices:
        ax.plot(
            zexp.real[~keep_mask], -zexp.imag[~keep_mask],
            "x",
            markersize=8,
            alpha=0.9,
            color="orange",
            linestyle="none",
            label="Excluded Exp"
        )

        for i in exclude_indices:
            ax.annotate(
                str(i),
                xy=(zexp.real[i], -zexp.imag[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="orange",
                fontweight="bold"
            )

    ax.plot(
        zfit.real, -zfit.imag,
        "-",
        linewidth=2,
        color="red",
        label=f"{concentration} mM (Fit)"
    )

    ax.set_xlabel("Zre (ohm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("-Zim (ohm)", fontsize=11, fontweight="bold")
    ax.set_title(f"{title} | {concentration} mM", fontsize=12, pad=15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    return fig


def make_bode_figure(freq, zexp, zfit, title="Bode Plot"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.semilogx(freq, np.abs(zexp), "o", label="Exp", markersize=4)
    ax1.semilogx(freq, np.abs(zfit), "-", label="Fit", linewidth=2)
    ax1.set_ylabel("|Z| (ohm)")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend()

    ph_exp = np.degrees(np.angle(zexp))
    ph_fit = np.degrees(np.angle(zfit))
    ax2.semilogx(freq, ph_exp, "o", label="Exp", markersize=4)
    ax2.semilogx(freq, ph_fit, "-", label="Fit", linewidth=2)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def make_summary_plot(df_sum, sam_type, substrate, x_log=False):
    sam_type = normalize_sam_name(sam_type)
    plot_df = df_sum[(df_sum["SAM_INTERNAL"] == sam_type) & (df_sum["Substrate"] == substrate)].copy()
    plot_df = plot_df.sort_values(by="Concentration_mM").reset_index(drop=True)
    if plot_df.empty:
        return None

    status_marker = {"Good": "o", "Borderline": "^", "Warning": "x"}
    x = plot_df["Concentration_mM"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel("Concentration (mM)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Normalized Total R Index (Ohm·cm²)", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    if x_log:
        ax1.set_xscale("log")

    for st_name in ["Good", "Borderline", "Warning"]:
        m = plot_df["Status"].eq(st_name).to_numpy()
        if not np.any(m):
            continue

        y = plot_df.loc[m, "Total_R_Index_Norm_mean"].to_numpy()
        yerr = plot_df.loc[m, "Total_R_Index_Norm_std"].to_numpy()

        ok = np.isfinite(y) & (y > 0)
        if not np.any(ok):
            continue

        ax1.errorbar(
            x[m][ok],
            y[ok],
            yerr=yerr[ok],
            fmt=status_marker[st_name],
            markersize=10,
            capsize=5,
            linestyle="none",
            label=f"R Index ({st_name})",
        )

        for xi, yi in zip(x[m][ok], y[ok]):
            ax1.annotate(
                f"Conc:{xi}\nR:{yi:.1e}",
                xy=(xi, yi),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="blue",
                alpha=0.8
            )

    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Effective Thickness (nm)", fontsize=12, fontweight="bold", color="darkred")

    good = plot_df["Status"].eq("Good").to_numpy()
    if np.any(good):
        y2 = plot_df.loc[good, "Thickness_mean"].to_numpy()
        y2e = plot_df.loc[good, "Thickness_std"].to_numpy()
        ok2 = np.isfinite(y2) & (y2 > 0)
        if np.any(ok2):
            ax2.errorbar(
                x[good][ok2],
                y2[ok2],
                yerr=y2e[ok2],
                fmt="s",
                markersize=10,
                color="darkred",
                capsize=5,
                linestyle="none",
                label="Thickness (Good)",
            )

            for xi, yi in zip(x[good][ok2], y2[ok2]):
                ax2.annotate(
                    f"Thk:{yi:.1f}nm",
                    xy=(xi, yi),
                    xytext=(5, -15),
                    textcoords="offset points",
                    fontsize=8,
                    color="darkred",
                    fontweight="bold"
                )

            ax2.set_ylim(0, max(5, float(np.nanmax(y2[ok2])) * 1.3))
        else:
            ax2.set_ylim(0, 5)
    else:
        ax2.set_ylim(0, 5)

    bad = ~good
    for xi in x[bad]:
        ax2.annotate(
            "Invalid",
            xy=(xi, 0),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="gray"
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True
    )

    plt.title(
        f"SAM Performance Analysis: {substrate} / {display_sam_name(sam_type)}",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    fig.tight_layout(rect=[0, 0, 0.80, 1])
    return fig


def make_batch_nyquist_panel_from_queue(queue_items, sam_type, substrate):
    sam_type = normalize_sam_name(sam_type)
    filtered = [
        item for item in queue_items
        if item["SAM"] == sam_type and item["Substrate"] == substrate
    ]
    if not filtered:
        return None

    filtered = sorted(filtered, key=lambda d: d["Concentration_mM"])

    fig, ax = plt.subplots(figsize=(10, 7))

    for item in filtered:
        conc = item["Concentration_mM"]
        zexp = item["zexp"]
        zfit = item["zfit"]

        p = ax.plot(
            zexp.real, -zexp.imag,
            "o",
            markersize=5,
            alpha=0.5,
            linestyle="none",
            label=f"{float(conc):g} mM"
        )

        line_color = p[0].get_color()
        ax.plot(
            zfit.real, -zfit.imag,
            "-",
            linewidth=1.8,
            color=line_color,
            alpha=0.9,
            label="_nolegend_"
        )

    ax.set_xlabel("Zre (ohm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("-Zim (ohm)", fontsize=12, fontweight="bold")
    ax.set_title(f"Batch Nyquist: {substrate} / {display_sam_name(sam_type)}", fontsize=14, pad=20)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
        title="Concentration",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# =========================================================
# 7. 결과 계산
# =========================================================
def classify_status(cpe_p: float) -> str:
    if cpe_p >= 0.90:
        return "Good"
    if cpe_p >= 0.85:
        return "Borderline"
    return "Warning"


def build_summary_row(file_name, file_path, sheet_name, substrate, sam_type, concentration, area_cm2, param_dict, fit_r2, fit_rmse):
    sam_type = normalize_sam_name(sam_type)

    row = {
        "Source_File": file_name,
        "Source_Path": file_path,
        "Used_Sheet": sheet_name,
        "Substrate": substrate,
        "SAM": display_sam_name(sam_type),
        "SAM_INTERNAL": sam_type,
        "Concentration_mM": concentration,
        "Area_cm2": area_cm2,
        "Model": "3-tc Nested" if sam_type == "B_1" else "Semi-infinite W Nested",
        "Fit_R2": fit_r2,
        "Fit_RMSE": fit_rmse,
    }
    row.update(param_dict)

    if sam_type == "C_1":
        raw_total_R_index = row["R1_inner"] + row["R2_interface"]
        cpe_t_for_thickness = row["CPE1_T_outer"]
        cpe_p_for_status = row["CPE1_P_outer"]
    else:
        raw_total_R_index = row["R_ct"] + row["R_int"] + row["R_sam"]
        cpe_t_for_thickness = row["CPE_sam_T"]
        cpe_p_for_status = row["CPE_sam_P"]

    row["Total_R_Index_raw"] = raw_total_R_index
    row["Total_R_Index_Norm_Ohm_cm2"] = raw_total_R_index * area_cm2
    row["CPE_T_Norm_F_per_cm2"] = cpe_t_for_thickness / area_cm2
    row["Status"] = classify_status(cpe_p_for_status)

    if row["Status"] == "Good":
        d_nm = (E0 * EPSILON_R[sam_type] / row["CPE_T_Norm_F_per_cm2"]) * 1e7
        row["Thickness_nm"] = round(float(d_nm), 2)
    else:
        row["Thickness_nm"] = np.nan

    return row


def build_pointwise_df(file_name, sam_type, substrate, concentration, freq, zexp, zfit):
    sam_type = normalize_sam_name(sam_type)
    return pd.DataFrame({
        "Source_File": file_name,
        "SAM": display_sam_name(sam_type),
        "SAM_INTERNAL": sam_type,
        "Substrate": substrate,
        "Concentration_mM": concentration,
        "Frequency_Hz": freq,
        "Zre_exp_ohm": zexp.real,
        "Zim_exp_ohm": zexp.imag,
        "Zmag_exp_ohm": np.abs(zexp),
        "Zph_exp_deg": np.degrees(np.angle(zexp)),
        "Zre_fit_ohm": zfit.real,
        "Zim_fit_ohm": zfit.imag,
        "Zmag_fit_ohm": np.abs(zfit),
        "Zph_fit_deg": np.degrees(np.angle(zfit)),
        "Residual_Re_ohm": zfit.real - zexp.real,
        "Residual_Im_ohm": zfit.imag - zexp.imag,
    })


def std0(x):
    s = x.std(ddof=1)
    return 0.0 if (pd.isna(s) or np.isinf(s)) else float(s)


def build_batch_summary(df_raw):
    group_cols = ["Substrate", "SAM", "SAM_INTERNAL", "Concentration_mM"]

    agg_dict = {
        "n": ("Concentration_mM", "count"),
        "Area_cm2_mean": ("Area_cm2", "mean"),
        "Fit_R2_mean": ("Fit_R2", "mean"),
        "Fit_R2_std": ("Fit_R2", std0),
        "Fit_RMSE_mean": ("Fit_RMSE", "mean"),
        "Fit_RMSE_std": ("Fit_RMSE", std0),
        "Total_R_Index_Norm_mean": ("Total_R_Index_Norm_Ohm_cm2", "mean"),
        "Total_R_Index_Norm_std": ("Total_R_Index_Norm_Ohm_cm2", std0),
        "Thickness_mean": ("Thickness_nm", "mean"),
        "Thickness_std": ("Thickness_nm", std0),
        "Status": ("Status", "first"),
    }

    exclude_cols = {
        "Concentration_mM", "Area_cm2",
        "Fit_R2", "Fit_RMSE",
        "Total_R_Index_raw", "Total_R_Index_Norm_Ohm_cm2",
        "CPE_T_Norm_F_per_cm2", "Thickness_nm"
    }

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

    for c in numeric_cols:
        if c in exclude_cols:
            continue
        if c in group_cols:
            continue
        agg_dict[f"{c}_mean"] = (c, "mean")
        agg_dict[f"{c}_std"] = (c, std0)

    df_sum = (
        df_raw.groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .sort_values(by=["Substrate", "SAM_INTERNAL", "Concentration_mM"])
        .reset_index(drop=True)
    )

    return df_sum


# =========================================================
# 8. 다운로드 빌더
# =========================================================
def build_single_excel_bytes(raw_row_df, point_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        raw_row_df.to_excel(writer, index=False, sheet_name="Raw_Fit_Params")
        raw_row_df.to_excel(writer, index=False, sheet_name="Summary")
        point_df.to_excel(writer, index=False, sheet_name="Pointwise_Fit")
    output.seek(0)
    return output


def build_batch_excel_bytes(df_raw, df_sum, df_points_all):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_raw.to_excel(writer, index=False, sheet_name="Raw_Fit_Params")
        df_sum.to_excel(writer, index=False, sheet_name="Summary")
        df_points_all.to_excel(writer, index=False, sheet_name="Pointwise_Fit")
    output.seek(0)
    return output


def build_batch_zip_bytes(excel_bytes, nyquist_pngs, bode_pngs, summary_pngs, batch_nyquist_pngs):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"EIS_Batch_Report_{ts}.xlsx", excel_bytes.getvalue())

        for name, data in nyquist_pngs.items():
            zf.writestr(f"Nyquist/{name}", data)

        for name, data in bode_pngs.items():
            zf.writestr(f"Bode/{name}", data)

        for name, data in summary_pngs.items():
            zf.writestr(f"SummaryPlots/{name}", data)

        for name, data in batch_nyquist_pngs.items():
            zf.writestr(f"BatchNyquist/{name}", data)

    zip_buf.seek(0)
    return zip_buf


# --- 1) Single-file review ---
st.header("1) Single-file review")
uploaded_file = st.file_uploader("xlsx 파일 업로드 (single)", type=["xlsx"], key="single_uploader")

if uploaded_file is not None:
    file_token = f"{uploaded_file.name}_{uploaded_file.size}"
    try:
        sheet_name, df_eis, freq, zexp = read_eis_from_uploaded(uploaded_file)
        sam_guess, substrate_guess, conc_guess = parse_metadata_from_filename(uploaded_file.name)

        col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
        with col_meta1:
            # 사용자가 선택할 수 있는 모델 명칭 (사용자 기존 로직 반영)
            sam_display = st.selectbox(
                "Select Model",
                ["B-1 (3-tc Nested)", "C-1 (Semi-infinite W)"],
                index=0 if sam_guess != "C_1" else 1,
                key=f"{file_token}_model_selector"
            )
            # 내부적으로 사용할 sam_type 결정
            sam_type = "B_1" if "B-1" in sam_display else "C_1"

        with col_meta2:
            substrate = st.selectbox(
                "Substrate",
                ["CU", "CO"],
                index=0 if substrate_guess != "CO" else 1,
                key=f"{file_token}_single_sub"
            )

        with col_meta3:
            concentration = st.number_input(
                "Concentration (mM)",
                min_value=0.0,
                value=float(conc_guess) if conc_guess is not None else 0.0,
                step=0.1,
                key=f"{file_token}_single_conc"
            )

        with col_meta4:
            area_cm2 = st.number_input(
                "Area (cm²)",
                min_value=1e-9,
                value=0.14,
                step=0.01,
                format="%.4f",
                key=f"{file_token}_single_area"
            )
        st.write(f"사용 시트: {sheet_name}")
        st.write(f"총 포인트 수: {len(freq)}")



        # 1. 모델 정보와 초기 추정치 계산 (항상 수행)
        names, lb, ub, _ = get_model_info(sam_type)
        default_guess = build_initial_guess(freq, zexp, sam_type) # <--- 여기서 항상 정의됩니다.

        # 2. 모델 변경 감지 및 세션 강제 초기화
        model_changed_key = f"{file_token}_last_model"
        if st.session_state.get(model_changed_key) != sam_type:
            st.session_state[model_changed_key] = sam_type
            # 모델이 바뀌면 해당 모델 전용 파라미터 값들을 세션에 새로 덮어씀
            for name, dft in zip(names, default_guess):
                st.session_state[f"{file_token}_{sam_type}_{name}_val"] = float(dft)
            st.session_state[f"{file_token}_{sam_type}_current_fit_params"] = default_guess.tolist()
            st.rerun()

        # 파라미터 상태 초기화
        for name, dft in zip(names, default_guess):
            state_key = f"{file_token}_{sam_type}_{name}_val"
            if state_key not in st.session_state:
                st.session_state[state_key] = float(dft)

        # 현재 표시용 fit 상태 저장
        fit_state_key = f"{file_token}_{sam_type}_current_fit_params"
        if fit_state_key not in st.session_state:
            st.session_state[fit_state_key] = default_guess.astype(float).tolist()

        # 아웃라이어 상태
        outlier_state_key = f"{file_token}_{sam_type}_excluded_idx_single"
        if outlier_state_key not in st.session_state:
            st.session_state[outlier_state_key] = []
            
        widget_ver_key = f"{file_token}_{sam_type}_widget_ver"
        if widget_ver_key not in st.session_state:
            st.session_state[widget_ver_key] = 0
            
        # 인터랙티브 아웃라이어 선택용 데이터프레임
        with st.expander("Outlier 제외 설정", expanded=False):
            st.caption("아래 인덱스를 선택하면 Auto Fit에서 제외됩니다. Nyquist에서 주황색 X와 번호로 표시됩니다.")
            df_points_preview = pd.DataFrame({
                "Index": np.arange(len(freq)),
                "Freq_Hz": freq,
                "Zre": zexp.real,
                "Zim": zexp.imag,
                "AbsZ": np.abs(zexp),
            })
            st.dataframe(df_points_preview, use_container_width=True, height=260)

            selected_outliers = st.multiselect(
                "제외할 포인트 인덱스",
                options=df_points_preview["Index"].tolist(),
                default=st.session_state[outlier_state_key],
                key=f"multiselect_{outlier_state_key}"
            )
            st.session_state[outlier_state_key] = selected_outliers

        left, right = st.columns([1.15, 1.85])

        with left:
            st.subheader("Manual Parameters")

            current_params = []
            for name, lo, hi in zip(names, lb, ub):
                state_key = f"{file_token}_{sam_type}_{name}_val"
                is_p = "_P" in name

                if is_p:
                    slider_val = st.slider(
                        f"{name} slider",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state[state_key]),
                        step=0.01,
                        key=f"s_{state_key}_v{st.session_state[widget_ver_key]}"
                    )
                    st.session_state[state_key] = float(slider_val)
                else:
                    e_lo = float(np.log10(max(lo, 1e-12)))
                    e_hi = float(np.log10(max(hi, 1e12)))
                    curr_log = float(np.log10(max(st.session_state[state_key], 1e-12)))
                    slider_val = st.slider(
                        f"{name} log10",
                        min_value=e_lo,
                        max_value=e_hi,
                        value=float(np.clip(curr_log, e_lo, e_hi)),
                        step=0.1,
                        key=f"s_{state_key}_v{st.session_state[widget_ver_key]}"
                    )
                    st.session_state[state_key] = float(10 ** slider_val)

                input_val = st.number_input(
                    f"{name} input",
                    value=float(st.session_state[state_key]),
                    format="%.6e" if not is_p else "%.4f",
                    key=f"i_{state_key}_v{st.session_state[widget_ver_key]}"
                )
                st.session_state[state_key] = float(input_val)
                current_params.append(float(input_val))

            c1, c2 = st.columns(2)

            with c1:
                if st.button("Manual Fit 반영", use_container_width=True):
                    st.session_state[fit_state_key] = list(current_params)
                    st.rerun()

            with c2:
                    
                if st.button("Auto Fit 실행", use_container_width=True, type="primary"):
                     excluded_idx = st.session_state.get(outlier_state_key, [])
                     p_fitted, _, _, _, _ = fit_eis(
                          freq, zexp, sam_type, current_params, exclude_indices=excluded_idx
                     )

                     fitted_list = [float(p_fitted[n]) for n in names]

                     for name, val in zip(names, fitted_list):
                         state_key = f"{file_token}_{sam_type}_{name}_val"
                         st.session_state[state_key] = float(val)

                     st.session_state[fit_state_key] = fitted_list

                # 최초 1회에 한해 manual 입력칸/슬라이더를 새 값으로 재생성
                     auto_applied_key = f"{file_token}_{sam_type}_autofit_applied_once"
                     if auto_applied_key not in st.session_state:
                         st.session_state[auto_applied_key] = False

                     if not st.session_state[auto_applied_key]:
                         st.session_state[widget_ver_key] += 1
                         st.session_state[auto_applied_key] = True

                     st.rerun()
                
            if st.button("현재 결과를 Batch Queue에 추가", use_container_width=True):
                excluded_idx = st.session_state.get(outlier_state_key, [])
                active_params = st.session_state[fit_state_key]

                param_dict, zfit, fit_r2, fit_rmse = evaluate_current_params(freq, zexp, sam_type, active_params)

                raw_row = build_summary_row(
                    uploaded_file.name,
                    uploaded_file.name,
                    sheet_name,
                    substrate,
                    sam_type,
                    concentration,
                    area_cm2,
                    param_dict,
                    fit_r2,
                    fit_rmse
                )

                point_df = build_pointwise_df(uploaded_file.name, sam_type, substrate, concentration, freq, zexp, zfit)

                reviewed_item = {
                    "Source_File": uploaded_file.name,
                    "Used_Sheet": sheet_name,
                    "SAM": sam_type,
                    "SAM_DISPLAY": display_sam_name(sam_type),
                    "Substrate": substrate,
                    "Concentration_mM": concentration,
                    "Area_cm2": area_cm2,
                    "Excluded_Indices": list(excluded_idx),
                    "Fit_R2": fit_r2,
                    "Fit_RMSE": fit_rmse,
                    "Param_Dict": param_dict,
                    "Summary_Row": raw_row,
                    "Pointwise_DF": point_df,
                    "zexp": np.array(zexp).copy(),
                    "zfit": np.array(zfit).copy(),
                    "freq": np.array(freq).copy(),
                }

                new_queue = []
                replaced = False
                for item in st.session_state["reviewed_batch_items"]:
                    same_key = (
                        item["Source_File"] == reviewed_item["Source_File"]
                        and item["SAM"] == reviewed_item["SAM"]
                        and item["Substrate"] == reviewed_item["Substrate"]
                        and float(item["Concentration_mM"]) == float(reviewed_item["Concentration_mM"])
                    )
                    if same_key:
                        new_queue.append(reviewed_item)
                        replaced = True
                    else:
                        new_queue.append(item)

                if not replaced:
                    new_queue.append(reviewed_item)

                st.session_state["reviewed_batch_items"] = new_queue
                st.success("현재 결과를 Batch Queue에 저장했습니다.")

        with right:
            active_params = st.session_state[fit_state_key]
            excluded_idx = st.session_state.get(outlier_state_key, [])

            param_dict, zfit, fit_r2, fit_rmse = evaluate_current_params(freq, zexp, sam_type, active_params)

            st.subheader(f"Fit Quality (R²: {fit_r2:.6f}, RMSE: {fit_rmse:.6e})")
            t1, t2 = st.tabs(["Nyquist", "Bode"])

            with t1:
                fig1 = make_nyquist_figure(
                    zexp, zfit, concentration,
                    f"{uploaded_file.name} | {display_sam_name(sam_type)}",
                    exclude_indices=excluded_idx
                )
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)

            with t2:
                fig2 = make_bode_figure(freq, zexp, zfit, f"{uploaded_file.name} | {display_sam_name(sam_type)}")
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            st.subheader("Current Parameters")
            st.dataframe(pd.DataFrame({"Parameter": list(param_dict.keys()), "Value": list(param_dict.values())}), use_container_width=True)

            raw_row = build_summary_row(
                uploaded_file.name,
                uploaded_file.name,
                sheet_name,
                substrate,
                sam_type,
                concentration,
                area_cm2,
                param_dict,
                fit_r2,
                fit_rmse
            )
            raw_row_df = pd.DataFrame([raw_row])
            point_df = build_pointwise_df(uploaded_file.name, sam_type, substrate, concentration, freq, zexp, zfit)

            st.subheader("Single Summary Result")
            st.dataframe(raw_row_df, use_container_width=True)

            single_excel = build_single_excel_bytes(raw_row_df, point_df)
            st.download_button(
                label="📥 Single 결과 Excel 다운로드",
                data=single_excel,
                file_name=f"EIS_Single_Report_{display_sam_name(sam_type)}_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"실행 중 오류 발생: {e}")


# =========================================================
# 10. UI - Batch review / export
# =========================================================
st.header("2) Batch review / export")

queue_items = st.session_state.get("reviewed_batch_items", [])

if not queue_items:
    st.info("아직 Batch Queue에 추가된 reviewed dataset이 없습니다.")
else:
    queue_preview = pd.DataFrame([
        {
            "Queue_Key": f"{item['Source_File']}|{item['SAM']}|{item['Substrate']}|{item['Concentration_mM']}",
            "Source_File": item["Source_File"],
            "SAM": item["SAM_DISPLAY"],
            "Substrate": item["Substrate"],
            "Concentration_mM": item["Concentration_mM"],
            "Area_cm2": item["Area_cm2"],
            "Fit_R2": item["Fit_R2"],
            "Fit_RMSE": item["Fit_RMSE"],
            "Excluded_Count": len(item["Excluded_Indices"]),
        }
        for item in queue_items
    ])

    st.subheader("Batch Queue")
    st.dataframe(queue_preview, use_container_width=True)

    q1, q2 = st.columns(2)

    with q1:
        remove_name = st.selectbox(
            "Queue에서 제거할 항목",
            options=["(선택 안함)"] + queue_preview["Queue_Key"].tolist(),
            key="queue_remove_select"
        )

    with q2:
        if st.button("선택 파일 Queue에서 제거", use_container_width=True):
            if remove_name != "(선택 안함)":
                st.session_state["reviewed_batch_items"] = [
                    item for item in queue_items
                    if f"{item['Source_File']}|{item['SAM']}|{item['Substrate']}|{item['Concentration_mM']}" != remove_name
                ]
                st.rerun()

    if st.button("Batch Queue 전체 비우기", use_container_width=True):
        st.session_state["reviewed_batch_items"] = []
        st.rerun()

    x_axis_log = st.toggle("X축 로그 스케일", value=False, key="batch_xlog")

    df_raw = pd.DataFrame([item["Summary_Row"] for item in queue_items])
    df_raw = df_raw.sort_values(by=["Substrate", "SAM_INTERNAL", "Concentration_mM"]).reset_index(drop=True)
    df_sum = build_batch_summary(df_raw)

    point_dfs = [item["Pointwise_DF"] for item in queue_items]
    df_points_all = pd.concat(point_dfs, ignore_index=True) if point_dfs else pd.DataFrame()

    st.subheader("Batch Summary Result")
    st.dataframe(df_sum, use_container_width=True)

    unique_pairs = df_sum[["SAM_INTERNAL", "Substrate"]].drop_duplicates()

    summary_pngs = {}
    batch_nyquist_pngs = {}
    nyquist_pngs = {}
    bode_pngs = {}

    for idx, item in enumerate(queue_items, start=1):
        fig_n = make_nyquist_figure(
            item["zexp"], item["zfit"], item["Concentration_mM"],
            f"{item['Source_File']} | {item['SAM_DISPLAY']}",
            exclude_indices=item["Excluded_Indices"]
        )
        nyquist_pngs[f"Nyquist_{idx:03d}_{item['Source_File']}.png"] = fig_to_png_bytes(fig_n)
        plt.close(fig_n)

        fig_b = make_bode_figure(
            item["freq"], item["zexp"], item["zfit"],
            f"{item['Source_File']} | {item['SAM_DISPLAY']}"
        )
        bode_pngs[f"Bode_{idx:03d}_{item['Source_File']}.png"] = fig_to_png_bytes(fig_b)
        plt.close(fig_b)

    st.subheader("Batch Summary Plots")
    for _, pair in unique_pairs.iterrows():
        fig = make_summary_plot(df_sum, pair["SAM_INTERNAL"], pair["Substrate"], x_log=x_axis_log)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            summary_pngs[f"Summary_{pair['Substrate']}_{display_sam_name(pair['SAM_INTERNAL'])}.png"] = fig_to_png_bytes(fig)
            plt.close(fig)

    st.subheader("Batch Nyquist Panels")
    for _, pair in unique_pairs.iterrows():
        fig_ny = make_batch_nyquist_panel_from_queue(queue_items, pair["SAM_INTERNAL"], pair["Substrate"])
        if fig_ny is not None:
            st.pyplot(fig_ny, use_container_width=True)
            batch_nyquist_pngs[f"BatchNyquist_{pair['Substrate']}_{display_sam_name(pair['SAM_INTERNAL'])}.png"] = fig_to_png_bytes(fig_ny)
            plt.close(fig_ny)

    batch_excel = build_batch_excel_bytes(df_raw, df_sum, df_points_all)

    st.download_button(
        label="📥 Batch 결과 Excel 다운로드",
        data=batch_excel,
        file_name=f"EIS_Batch_Report_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    batch_zip = build_batch_zip_bytes(
        batch_excel,
        nyquist_pngs,
        bode_pngs,
        summary_pngs,
        batch_nyquist_pngs
    )

    st.download_button(
        label="📦 Batch 결과 ZIP 다운로드 (Excel + 모든 PNG)",
        data=batch_zip,
        file_name=f"EIS_Batch_Report_{ts}.zip",
        mime="application/zip",
        use_container_width=True
    )
