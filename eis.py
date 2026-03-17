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
    if "ODPA" in n or "B_1" in n or "B-1" in n: return "B_1"
    if "BTA" in n or "C_1" in n or "C-1" in n: return "C_1"
    return n

def display_sam_name(name: str) -> str:
    n = normalize_sam_name(name)
    if n == "B_1": return "B-1"
    if n == "C_1": return "C-1"
    return str(name)


# =========================================================
# 3. 회로 수학
# =========================================================
def Z_R(R, w): return np.full_like(w, complex(R, 0.0), dtype=np.complex128)
def Z_C(C, w): return 1.0 / (1j * w * max(float(C), 1e-30))
def Z_CPE(T, P, w): return 1.0 / (max(float(T), 1e-30) * (1j * w) ** min(max(float(P), 0.0), 1.0))
def Z_Ws_sigma(sigma, w): return max(float(sigma), 1e-30) / np.sqrt(1j * w)
def Z_parallel(Z1, Z2): return 1.0 / (1.0 / Z1 + 1.0 / Z2)

def Z_C_1(params, w):
    Rs, Ws1_sigma, C1, R1, T1, P1, R2 = params
    Z0 = Z_parallel(Z_Ws_sigma(Ws1_sigma, w), Z_C(C1, w))
    Z1 = Z0 + Z_R(R1, w)
    Z2 = Z_parallel(Z1, Z_CPE(T1, P1, w))
    Z3 = Z2 + Z_R(R2, w)
    return Z3 + Z_R(Rs, w)

def Z_B_1(params, w):
    C_dl, R_ct, R_int, T_int, P_int, R_sam, T_sam, P_sam, Rs_sol = params
    Z0 = Z_parallel(Z_C(C_dl, w), Z_R(R_ct, w))
    Z1 = Z0 + Z_R(R_int, w)
    Z2 = Z_parallel(Z1, Z_CPE(T_int, P_int, w))
    Z3 = Z2 + Z_R(R_sam, w)
    Z4 = Z_parallel(Z3, Z_CPE(T_sam, P_sam, w))
    return Z4 + Z_R(Rs_sol, w)

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
# 4. 데이터 읽기 & 유틸
# =========================================================
def pick_data_sheet(xl):
    candidate_sheets = [s for s in xl.sheet_names if str(s).upper().startswith("DATA")]
    if not candidate_sheets: candidate_sheets = xl.sheet_names
    best_sheet, best_count = None, -1
    for s in candidate_sheets:
        try:
            df_try = xl.parse(s)
            needed = {"Frequency(Hz)", "Zre(ohm)", "Zim(ohm)"}
            if needed.issubset(set(df_try.columns.astype(str))):
                cnt = int(df_try[list(needed)].dropna().shape[0])
                if cnt > best_count: best_count, best_sheet = cnt, s
        except: continue
    if not best_sheet: raise ValueError("Data 시트를 찾지 못했습니다.")
    return best_sheet

def read_eis_from_uploaded(uploaded_file):
    xl = pd.ExcelFile(uploaded_file)
    sheet = pick_data_sheet(xl)
    df = xl.parse(sheet)
    needed = ["Frequency(Hz)", "Zre(ohm)", "Zim(ohm)"]
    out = df[needed].copy().replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out["Frequency(Hz)"] > 0].sort_values(by="Frequency(Hz)", ascending=False).reset_index(drop=True)
    return sheet, out, out["Frequency(Hz)"].to_numpy(dtype=float), out["Zre(ohm)"].to_numpy() + 1j * out["Zim(ohm)"].to_numpy()

def parse_metadata_from_filename(name):
    stem = os.path.splitext(name)[0].upper().replace("[", "_").replace("]", "_")
    sam = "B_1" if any(k in stem for k in ["ODPA", "B-1", "B_1"]) else ("C_1" if any(k in stem for k in ["BTA", "C-1", "C_1"]) else None)
    sub = "CU" if re.search(r"(^|[_\-\s])CU([_\-\s]|$)", stem) else ("CO" if re.search(r"(^|[_\-\s])CO([_\-\s]|$)", stem) else None)
    m = re.search(r"(\d+(?:\.\d+)?)\s*MM", stem)
    conc = float(m.group(1)) if m else None
    return sam, sub, conc


# =========================================================
# 5. 피팅 로직 (Interactive 반영)
# =========================================================
def build_initial_guess(freq, zexp, sam_type):
    sam_type = normalize_sam_name(sam_type)
    zmag = np.abs(zexp)
    Rs_guess = max(1e-3, float(np.min(zmag)))
    Rspan = max(1.0, float(np.max(zmag) - np.min(zmag)))
    if sam_type == "B_1":
        return np.array([1e-6, 0.1*Rspan, 0.2*Rspan, 1e-5, 0.9, 0.3*Rspan, 1e-6, 0.8, Rs_guess])
    return np.array([Rs_guess, 10.0, 1e-6, 0.3*Rspan, 1e-5, 0.9, 0.7*Rspan])

def residuals(params, w, zexp, model_func):
    zfit = model_func(params, w)
    scale = np.maximum(np.abs(zexp), 1.0)
    return np.concatenate([(zfit.real - zexp.real) / scale, (zfit.imag - zexp.imag) / scale])

def fit_eis(freq, zexp, sam_type, x0, exclude_indices=None, custom_bounds=None):
    names, lb, ub, model_func = get_model_info(sam_type)
    if custom_bounds: lb, ub = custom_bounds
    
    f_fit, z_fit = freq.copy(), zexp.copy()
    if exclude_indices:
        mask = np.ones(len(f_fit), dtype=bool)
        mask[exclude_indices] = False
        f_fit, z_fit = f_fit[mask], z_fit[mask]
    
    w = 2 * np.pi * f_fit
    x0 = np.clip(x0, lb * 1.0001, ub / 1.0001)
    
    res = least_squares(residuals, x0=x0, bounds=(lb, ub), args=(w, z_fit, model_func), method="trf", max_nfev=20000)
    
    p = res.x
    zfit_full = model_func(p, 2 * np.pi * freq)
    zfit_eval = model_func(p, w)
    ss_res = np.sum((zfit_eval.real - z_fit.real)**2 + (zfit_eval.imag - z_fit.imag)**2)
    ss_tot = np.sum((z_fit.real - np.mean(z_fit))**2 + (z_fit.imag - np.mean(z_fit))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(np.abs(zfit_eval - z_fit)**2))
    
    return {k: float(v) for k, v in zip(names, p)}, zfit_full, float(r2), float(rmse), res

def evaluate_current_params(freq, zexp, sam_type, params):
    _, _, _, model_func = get_model_info(sam_type)
    zfit = model_func(np.array(params), 2 * np.pi * freq)
    rmse = np.sqrt(np.mean(np.abs(zfit - zexp)**2))
    ss_res = np.sum(np.abs(zfit - zexp)**2)
    ss_tot = np.sum(np.abs(zexp - np.mean(zexp))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0
    return {k: v for k, v in zip(get_model_info(sam_type)[0], params)}, zfit, float(r2), float(rmse)


# =========================================================
# 6. 시각화 함수 (기존 유지)
# =========================================================
def make_nyquist_figure(zexp, zfit, concentration, title="Nyquist Plot", exclude_indices=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.ones(len(zexp), dtype=bool)
    if exclude_indices: mask[exclude_indices] = False
    ax.plot(zexp.real[mask], -zexp.imag[mask], "o", alpha=0.6, color="navy", label="Exp")
    if exclude_indices: 
        ax.plot(zexp.real[~mask], -zexp.imag[~mask], "x", color="orange", label="Excluded")
        for i in exclude_indices: ax.annotate(str(i), (zexp.real[i], -zexp.imag[i]), color="orange")
    ax.plot(zfit.real, -zfit.imag, "-", color="red", label="Fit")
    ax.set_xlabel("Zre (ohm)"); ax.set_ylabel("-Zim (ohm)")
    ax.set_title(f"{title} | {concentration} mM"); ax.legend(); ax.grid(True, alpha=0.3)
    return fig

def make_bode_figure(freq, zexp, zfit, title="Bode Plot"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.semilogx(freq, np.abs(zexp), "o", freq, np.abs(zfit), "-")
    ax1.set_ylabel("|Z| (ohm)"); ax1.grid(True, which="both", alpha=0.3)
    ax2.semilogx(freq, np.degrees(np.angle(zexp)), "o", freq, np.degrees(np.angle(zfit)), "-")
    ax2.set_xlabel("Freq (Hz)"); ax2.set_ylabel("Phase (deg)"); ax2.grid(True, which="both", alpha=0.3)
    return fig

# (make_summary_plot, make_batch_nyquist_panel_from_queue, fig_to_png_bytes 등 기존 시각화 함수 생략 없이 유지됨을 전제)
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def make_summary_plot(df_sum, sam_type, substrate, x_log=False):
    sam_type = normalize_sam_name(sam_type)
    plot_df = df_sum[(df_sum["SAM_INTERNAL"] == sam_type) & (df_sum["Substrate"] == substrate)].sort_values("Concentration_mM")
    if plot_df.empty: return None
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Conc (mM)"); ax1.set_ylabel("R Index (Ohm cm2)"); ax1.set_yscale("log")
    if x_log: ax1.set_xscale("log")
    ax1.errorbar(plot_df["Concentration_mM"], plot_df["Total_R_Index_Norm_mean"], yerr=plot_df["Total_R_Index_Norm_std"], fmt='o-', label="R Index")
    ax2 = ax1.twinx(); ax2.set_ylabel("Thickness (nm)", color="red")
    ax2.errorbar(plot_df["Concentration_mM"], plot_df["Thickness_mean"], yerr=plot_df["Thickness_std"], fmt='s-', color="red", label="Thickness")
    plt.title(f"Analysis: {substrate} / {display_sam_name(sam_type)}"); ax1.grid(True, alpha=0.3)
    return fig

def make_batch_nyquist_panel_from_queue(queue_items, sam_type, substrate):
    sam_type = normalize_sam_name(sam_type)
    filtered = sorted([i for i in queue_items if i["SAM"] == sam_type and i["Substrate"] == substrate], key=lambda x: x["Concentration_mM"])
    if not filtered: return None
    fig, ax = plt.subplots(figsize=(8, 6))
    for item in filtered:
        ax.plot(item["zexp"].real, -item["zexp"].imag, "o", alpha=0.4, label=f"{item['Concentration_mM']}mM")
        ax.plot(item["zfit"].real, -item["zfit"].imag, "-", alpha=0.8)
    ax.set_title(f"Batch Nyquist: {substrate} / {display_sam_name(sam_type)}"); ax.legend(); ax.grid(True, alpha=0.3)
    return fig


# =========================================================
# 7. 결과 및 엑셀 빌더 (기존 유지)
# =========================================================
def classify_status(cpe_p): return "Good" if cpe_p >= 0.9 else ("Borderline" if cpe_p >= 0.85 else "Warning")

def build_summary_row(file_name, file_path, sheet_name, substrate, sam_type, concentration, area_cm2, param_dict, fit_r2, fit_rmse):
    sam_type = normalize_sam_name(sam_type)
    row = {"Source_File": file_name, "Substrate": substrate, "SAM": display_sam_name(sam_type), "SAM_INTERNAL": sam_type, "Concentration_mM": concentration, "Area_cm2": area_cm2, "Fit_R2": fit_r2, "Fit_RMSE": fit_rmse}
    row.update(param_dict)
    if sam_type == "C_1":
        r_tot, c_t, c_p = row["R1_inner"] + row["R2_interface"], row["CPE1_T_outer"], row["CPE1_P_outer"]
    else:
        r_tot, c_t, c_p = row["R_ct"] + row["R_int"] + row["R_sam"], row["CPE_sam_T"], row["CPE_sam_P"]
    row.update({"Total_R_Index_Norm_Ohm_cm2": r_tot * area_cm2, "Status": classify_status(c_p)})
    row["Thickness_nm"] = round((E0 * EPSILON_R[sam_type] / (c_t / area_cm2)) * 1e7, 2) if row["Status"] == "Good" else np.nan
    return row

def build_pointwise_df(file_name, sam_type, sub, conc, freq, zexp, zfit):
    return pd.DataFrame({"Source": file_name, "Freq": freq, "Zre_exp": zexp.real, "Zim_exp": zexp.imag, "Zre_fit": zfit.real, "Zim_fit": zfit.imag})

def build_batch_summary(df_raw):
    return df_raw.groupby(["Substrate", "SAM", "SAM_INTERNAL", "Concentration_mM"]).agg(
        n=("Concentration_mM", "count"), Fit_R2_mean=("Fit_R2", "mean"), Total_R_Index_Norm_mean=("Total_R_Index_Norm_Ohm_cm2", "mean"),
        Total_R_Index_Norm_std=("Total_R_Index_Norm_Ohm_cm2", lambda x: x.std() if len(x)>1 else 0),
        Thickness_mean=("Thickness_nm", "mean"), Thickness_std=("Thickness_nm", lambda x: x.std() if len(x)>1 else 0), Status=("Status", "first")
    ).reset_index()

def build_batch_excel_bytes(df_raw, df_sum, df_points):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="Raw_Fit", index=False)
        df_sum.to_excel(writer, sheet_name="Summary", index=False)
        df_points.to_excel(writer, sheet_name="Pointwise", index=False)
    return buf

def build_batch_zip_bytes(excel_bytes, nyquist_pngs, bode_pngs, summary_pngs, batch_nyquist_pngs):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"Report_{ts}.xlsx", excel_bytes.getvalue())
        for d, p in [("Nyquist", nyquist_pngs), ("Bode", bode_pngs), ("Summary", summary_pngs), ("BatchNy", batch_nyquist_pngs)]:
            for name, data in p.items(): zf.writestr(f"{d}/{name}", data)
    zip_buf.seek(0); return zip_buf


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
            sam_display = st.selectbox("Model", ["B-1 (3-tc)", "C-1 (Warburg)"], index=0 if sam_guess != "C_1" else 1, key=f"{file_token}_mod")
            sam_type = "B_1" if "B-1" in sam_display else "C_1"
        with col_meta2:
            substrate = st.selectbox("Substrate", ["CU", "CO"], index=0 if substrate_guess != "CO" else 1, key=f"{file_token}_sub")
        with col_meta3:
            concentration = st.number_input("Conc (mM)", value=float(conc_guess) if conc_guess else 0.0, key=f"{file_token}_conc")
        with col_meta4:
            area_cm2 = st.number_input("Area (cm2)", value=0.14, format="%.4f", key=f"{file_token}_area")

        names, lb, ub, _ = get_model_info(sam_type)
        default_guess = build_initial_guess(freq, zexp, sam_type)

        # 모델 변경 감지 및 세션 초기화
        if st.session_state.get(f"{file_token}_last_mod") != sam_type:
            st.session_state[f"{file_token}_last_mod"] = sam_type
            for n, v in zip(names, default_guess): st.session_state[f"{file_token}_{sam_type}_{n}_val"] = float(v)
            st.session_state[f"{file_token}_{sam_type}_cur_fit"] = default_guess.tolist()
            st.session_state[f"{file_token}_wver"] = st.session_state.get(f"{file_token}_wver", 0) + 1
            st.rerun()

        for n, v in zip(names, default_guess):
            if f"{file_token}_{sam_type}_{n}_val" not in st.session_state: st.session_state[f"{file_token}_{sam_type}_{n}_val"] = float(v)
        if f"{file_token}_{sam_type}_cur_fit" not in st.session_state: st.session_state[f"{file_token}_{sam_type}_cur_fit"] = default_guess.tolist()
        if f"{file_token}_wver" not in st.session_state: st.session_state[f"{file_token}_wver"] = 0

        # Outlier
        outlier_key = f"{file_token}_{sam_type}_out"
        with st.expander("Outlier 설정"):
            sel_out = st.multiselect("제외 인덱스", options=list(range(len(freq))), default=st.session_state.get(outlier_key, []), key=f"ms_{outlier_key}")
            st.session_state[outlier_key] = sel_out

        left, right = st.columns([1.2, 1.8])
        with left:
            st.subheader("Manual & Interactive Fit")
            curr_params, curr_lb, curr_ub = [], lb.copy(), ub.copy()
            for i, name in enumerate(names):
                s_key = f"{file_token}_{sam_type}_{name}_val"
                c_in, c_fix = st.columns([4, 1])
                with c_fix: 
                    if st.checkbox("Fix", key=f"fix_{s_key}"):
                        curr_lb[i], curr_ub[i] = st.session_state[s_key]*0.999, st.session_state[s_key]*1.001
                with c_in:
                    val = st.number_input(name, value=st.session_state[s_key], format="%.4e" if "_P" not in name else "%.4f", key=f"in_{s_key}_{st.session_state[f'{file_token}_wver']}")
                    st.session_state[s_key] = val
                    curr_params.append(val)

            if st.button("🚀 현재 값에서 Auto Fit", type="primary", use_container_width=True):
                p_fit, _, _, _, _ = fit_eis(freq, zexp, sam_type, x0=curr_params, exclude_indices=st.session_state[outlier_key], custom_bounds=(curr_lb, curr_ub))
                for n, v in p_fit.items(): st.session_state[f"{file_token}_{sam_type}_{n}_val"] = v
                st.session_state[f"{file_token}_{sam_type}_cur_fit"] = [p_fit[n] for n in names]
                st.session_state[f"{file_token}_wver"] += 1
                st.rerun()

            if st.button("Add to Batch Queue", use_container_width=True):
                p_dict, zfit, r2, rmse = evaluate_current_params(freq, zexp, sam_type, st.session_state[f"{file_token}_{sam_type}_cur_fit"])
                row = build_summary_row(uploaded_file.name, uploaded_file.name, sheet_name, substrate, sam_type, concentration, area_cm2, p_dict, r2, rmse)
                item = {"Source_File": uploaded_file.name, "SAM": sam_type, "SAM_DISPLAY": display_sam_name(sam_type), "Substrate": substrate, "Concentration_mM": concentration, "Area_cm2": area_cm2, "Excluded_Indices": st.session_state[outlier_key], "zexp": zexp, "zfit": zfit, "freq": freq, "Summary_Row": row, "Pointwise_DF": build_pointwise_df(uploaded_file.name, sam_type, substrate, concentration, freq, zexp, zfit)}
                st.session_state["reviewed_batch_items"] = [i for i in st.session_state["reviewed_batch_items"] if not (i["Source_File"]==item["Source_File"] and i["SAM"]==item["SAM"] and i["Concentration_mM"]==item["Concentration_mM"])] + [item]
                st.success("Queue 추가 완료")

        with right:
            p_dict, zfit, r2, rmse = evaluate_current_params(freq, zexp, sam_type, st.session_state[f"{file_token}_{sam_type}_cur_fit"])
            st.subheader(f"Quality: R²={r2:.4f}, RMSE={rmse:.2e}")
            t1, t2 = st.tabs(["Nyquist", "Bode"])
            with t1: st.pyplot(make_nyquist_figure(zexp, zfit, concentration, exclude_indices=st.session_state[outlier_key]))
            with t2: st.pyplot(make_bode_figure(freq, zexp, zfit))
            st.dataframe(pd.DataFrame(p_dict.items(), columns=["Param", "Value"]), use_container_width=True)

    except Exception as e: st.error(f"오류: {e}")


# =========================================================
# 9. Batch UI (기존 유지)
# =========================================================
st.header("2) Batch review / export")
q_items = st.session_state.get("reviewed_batch_items", [])
if not q_items:
    st.info("Queue가 비어 있습니다.")
else:
    df_raw = pd.DataFrame([i["Summary_Row"] for i in q_items])
    df_sum = build_batch_summary(df_raw)
    st.dataframe(df_sum, use_container_width=True)
    
    if st.button("Queue 비우기"): st.session_state["reviewed_batch_items"] = []; st.rerun()
    
    # 리포트 생성
    ny_p, bo_p, su_p, bny_p = {}, {}, {}, {}
    for idx, item in enumerate(q_items):
        ny_p[f"Ny_{idx}_{item['Source_File']}.png"] = fig_to_png_bytes(make_nyquist_figure(item["zexp"], item["zfit"], item["Concentration_mM"], exclude_indices=item["Excluded_Indices"]))
        bo_p[f"Bo_{idx}_{item['Source_File']}.png"] = fig_to_png_bytes(make_bode_figure(item["freq"], item["zexp"], item["zfit"]))
    
    for _, pair in df_sum[["SAM_INTERNAL", "Substrate"]].drop_duplicates().iterrows():
        f1 = make_summary_plot(df_sum, pair["SAM_INTERNAL"], pair["Substrate"])
        if f1: su_p[f"Sum_{pair['Substrate']}_{pair['SAM_INTERNAL']}.png"] = fig_to_png_bytes(f1)
        f2 = make_batch_nyquist_panel_from_queue(q_items, pair["SAM_INTERNAL"], pair["Substrate"])
        if f2: bny_p[f"BatchNy_{pair['Substrate']}_{pair['SAM_INTERNAL']}.png"] = fig_to_png_bytes(f2)

    exc = build_batch_excel_bytes(df_raw, df_sum, pd.concat([i["Pointwise_DF"] for i in q_items]))
    st.download_button("📥 ZIP 다운로드 (Excel+PNG)", build_batch_zip_bytes(exc, ny_p, bo_p, su_p, bny_p), f"EIS_Batch_{ts}.zip", "application/zip", use_container_width=True)
