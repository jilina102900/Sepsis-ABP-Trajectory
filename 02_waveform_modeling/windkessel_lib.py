# -*- coding: utf-8 -*-
"""
Windkessel 仿真库 + 最小二乘反演（本地自包含实现）
================================================================
REPOSITORY NOTE
---------------
This module is reused verbatim from the archived analysis code.  Two fitting
routes are provided and they must not be mixed:

* ``invert_waveform`` / ``fit_tau``  -- shared-slope weighted least squares on
  ln P, with optional Huber re-weighting.  Used for the synthetic validation
  experiment (Supplementary Figure S1) and for waveform-level QC.
* ``estimate_tau_freePinf``          -- non-linear least squares on the
  free-asymptote decay model ``P(t) = P_inf + (P0 - P_inf) exp(-t / tau)``.
  This is the primary tau extraction route reported in Methods for MIMIC-III
  (fit window from 0.15 s after the systolic peak to the next trough,
  beats with NRMSE > 0.08 rejected).
================================================================
用途：P2/P3 主分析所需"真 Windkessel 参数"的本地反演实现，及其仿真库自测。
      不依赖 P1 的 PINN；P1 产出只作敏感性核对。

单位约定（全库统一，务必注意 R 的口径）
--------------------------------------
压力 mmHg | 容积 mL | 时间 s | 流量 Q mL/s
R  : mmHg*min/L  —— 临床口径（与 MAP/CO 同量纲；SVR[dyn*s*cm^-5] = 80 * R）
C  : mL/mmHg
Z0 : mmHg*s/mL
CO : L/min,  CO = SV(mL) * HR(bpm) / 1000
tau: s,      tau = R_SML * C

关键换算（本库唯一一处，集中定义避免口径漂移）：
    R_SML[mmHg*s/mL] = R[mmHg*min/L] * 60/1000 = R * 0.06
    R_SML * Q[mL/s] = R * CO[L/min] = MAP[mmHg]      （两条路径必须一致）

模型（三元件 Windkessel, RCR）
------------------------------
    C * dP_wk/dt = Q(t) - P_wk/(R/60)
    P_ao(t) = P_wk(t) + Z0 * Q(t)
舒张期 Q=0 => P_ao = P_wk => 纯指数衰减，其斜率给出 tau。

可辨识性（重要，直接决定任务4能做哪些对照）
------------------------------------------
* 仅凭 ABP 波形：只能稳健辨识 tau = R*C（舒张期衰减）。
* 要拆出 R 与 C 各自，必须另有心输出量 CO：R = MAP/CO, C = tau/(R*60/1000)。
  没有 CO 时 R 与 C 只以乘积形式可辨识，单独不可辨识。
* 另有系统性偏倚：R_hat = mean(P_ao)/CO 比真 R 高 (1000/60)*Z0（特征阻抗项），
  因 mean(P_ao) = mean(P_wk) + Z0*CO。
* Z0 需要主动脉流量 Q(t)，ABP 单独不可辨识。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

FS_DEFAULT = 125.0          # 与 MIMIC-III 波形一致
DT_INT = 0.001              # 内部积分步长
ANALYZE_BEATS = 4           # 参与分析的搏数
R_MINL_TO_SML = 60.0 / 1000.0   # mmHg*min/L -> mmHg*s/mL
Z0_BIAS_COEF = 1000.0 / 60.0    # R_hat 的 Z0 偏倚系数 (mmHg*min/L per mmHg*s/mL)
HUBER_K = 1.5                   # Huber 稳健回归拐点（与 _wls_slope 默认一致）


# ----------------------------------------------------------------------------
# 1. 前向模型：仿真库
# ----------------------------------------------------------------------------
def aortic_flow(t: np.ndarray, T: float, T_ej: float, SV: float) -> np.ndarray:
    """主动脉流量 Q(t) (mL/s)：收缩期半正弦射血，舒张期为 0。

    解析归一化：∫ sin(pi*u) dt over [0,T_ej] = T_ej*2/pi，故每搏积分严格等于 SV。
    """
    u = (t % T) / T_ej
    A = SV * np.pi / (2.0 * T_ej)
    return np.where(u < 1.0, A * np.sin(np.pi * np.clip(u, 0.0, 1.0)), 0.0)


def simulate_windkessel(R, C, Z0, HR, SV, fs=FS_DEFAULT, dt=DT_INT,
                        reflection=None, noise_sd=0.0, quant=0.05, seed=None,
                        burn_in_tau=8.0):
    """三元件 Windkessel 前向仿真。

    R : mmHg*min/L ; C : mL/mmHg ; Z0 : mmHg*s/mL ; HR : bpm ; SV : mL
    reflection : None 或 (amp_mmHg, t_bump_s, width_s)。在射血结束后 t_bump 秒处
                 叠加幅度 amp、宽度 width 的高斯隆起，模拟真实动脉波形中的二重波
                 /波反射，用于检验"纯指数舒张期衰减"假设被破坏时的稳健性。
    noise_sd : 叠加高斯噪声 sd (mmHg)
    quant    : 量化步长 (mmHg)，模拟数字存储分辨率
    burn_in_tau: 预热时长（以 tau 为单位），保证周期稳态。

    返回 dict：真值、真值导出量、稳态段采样波形与生理指标。
    """
    rng = np.random.default_rng(seed)
    T = 60.0 / HR
    T_ej = min(0.40 * np.sqrt(T), 0.45 * T)
    R_s = R * R_MINL_TO_SML                      # mmHg*s/mL
    tau = R_s * C                                # s

    n_burn = int(np.ceil(max(burn_in_tau * tau + 2.0 * T, 6.0 * T) / T))
    t = np.arange(0.0, (n_burn + ANALYZE_BEATS) * T, dt)
    q = aortic_flow(t, T, T_ej, SV)

    # 一阶线性递推的精确离散解（分段常流量近似）:
    #   P_wk[i] = alpha*P_wk[i-1] + (1-alpha)*R_s*q[i-1],  alpha = exp(-dt/(R_s*C))
    alpha = np.exp(-dt / (R_s * C))
    b = (1.0 - alpha) * R_s
    pwk = lfilter([0.0, b], [1.0, -alpha], q)
    p_ao = pwk + Z0 * q

    if reflection is not None:
        # 二重波式局部隆起（波反射在时域的体现）：局部瞬态、不引入常数偏移，
        # 只破坏"纯指数衰减"假设，不人为改变平均压。
        amp, t_bump, width = reflection
        phase = t % T
        center = T_ej + t_bump
        bump = amp * np.exp(-(((phase - center) / width) ** 2))
        bump = bump - bump.mean()
        p_ao = p_ao + bump

    # 只取最后 ANALYZE_BEATS 个完整搏（已过预热），降采样到 fs
    i0 = int(round(n_burn * T / dt))
    step = int(round((1.0 / fs) / dt))
    idx = np.arange(i0, len(t), step)
    t_s = t[idx] - t[idx[0]]
    p_s = p_ao[idx]
    q_s = q[idx]

    sbps, dbps = [], []
    for j in range(ANALYZE_BEATS):
        m = (t_s >= j * T) & (t_s < (j + 1) * T)
        if m.sum() < 10:
            continue
        sbps.append(p_s[m].max())
        dbps.append(p_s[m].min())
    sbp = float(np.mean(sbps)) if sbps else float("nan")
    dbp = float(np.mean(dbps)) if dbps else float("nan")
    map_true = float(np.mean(p_s))

    if noise_sd > 0:
        p_s = p_s + rng.normal(0.0, noise_sd, size=p_s.shape)
    if quant is not None and quant > 0:
        p_s = np.round(p_s / quant) * quant

    return {
        "t": t_s, "p": p_s, "q": q_s, "p_clean": p_ao[idx],
        "R": R, "C": C, "Z0": Z0, "tau": tau,
        "HR": HR, "SV": SV, "T": T, "T_ej": T_ej,
        "CO": SV * HR / 1000.0,
        "SBP": sbp, "DBP": dbp, "MAP": map_true, "PP": sbp - dbp,
        "n_burn": n_burn,
    }


def sample_params(rng, tau_band=(0.40, 2.50), co_band=(3.0, 10.0)):
    """按脓毒性休克生理范围抽样一组真值参数。

    参数范围（脓毒性休克 + NE 人群）：
      HR 60-140 bpm; SV 40-120 mL; CO 3-10 L/min; MAP 55-90 mmHg;
      C 0.8-2.5 mL/mmHg; Z0 0.03-0.12 mmHg*s/mL; tau 0.40-2.50 s
    """
    HR = rng.uniform(60.0, 140.0)
    SV = rng.uniform(40.0, 120.0)
    CO = SV * HR / 1000.0                        # L/min
    if not (co_band[0] <= CO <= co_band[1]):
        return None
    MAP_target = rng.uniform(55.0, 90.0)         # mmHg
    R = MAP_target / CO                          # mmHg*min/L
    C = rng.uniform(0.8, 2.5)                    # mL/mmHg
    tau = R * R_MINL_TO_SML * C                  # s
    Z0 = rng.uniform(0.03, 0.12)                 # mmHg*s/mL
    return dict(R=R, C=C, Z0=Z0, HR=HR, SV=SV,
                MAP_target=MAP_target, CO=CO, tau=tau)


def plausible(sim, sbp=(70, 185), dbp=(30, 105), pp=(15, 85), mapr=(50, 105)):
    """用前向仿真出的血压指标判断该参数组是否落在生理范围。"""
    for v in (sim["SBP"], sim["DBP"], sim["MAP"], sim["PP"]):
        if not np.isfinite(v):
            return False
    return bool(sbp[0] <= sim["SBP"] <= sbp[1] and
                dbp[0] <= sim["DBP"] <= dbp[1] and
                pp[0] <= sim["PP"] <= pp[1] and
                mapr[0] <= sim["MAP"] <= mapr[1])


def sample_plausible(rng, tau_band=(0.40, 2.50), max_try=400000, **sim_kw):
    """拒绝采样：返回一组生理合理的 (参数, 无噪声仿真结果)。"""
    for _ in range(max_try):
        p = sample_params(rng, tau_band=tau_band)
        if p is None:
            continue
        if not (tau_band[0] <= p["tau"] <= tau_band[1]):
            continue
        sim = simulate_windkessel(p["R"], p["C"], p["Z0"], p["HR"], p["SV"],
                                  noise_sd=0.0, quant=None, **sim_kw)
        if plausible(sim):
            return p, sim
    return None, None


# ----------------------------------------------------------------------------
# 2. 反演：收缩峰检测 + 舒张期衰减加权最小二乘
# ----------------------------------------------------------------------------
def detect_peaks(p: np.ndarray, fs: float, min_dist_s=0.40, win_s=0.12):
    """收缩压峰检测：局部极大 + 幅度门限 + 最小间距。返回峰索引数组。"""
    n = len(p)
    w = max(1, int(round(win_s * fs)))
    if n < 2 * w + 2:
        return np.array([], dtype=int)
    thr = p.min() + 0.35 * (p.max() - p.min())
    cand = [i for i in range(w, n - w)
            if p[i] == p[i - w:i + w + 1].max() and p[i] >= thr]
    if not cand:
        return np.array([], dtype=int)
    min_d = int(round(min_dist_s * fs))
    keep = []
    for i in cand:
        if not keep or i - keep[-1] >= min_d:
            keep.append(i)
        elif p[i] > p[keep[-1]]:
            keep[-1] = i
    return np.array(keep, dtype=int)


def _wls_slope(X, y, w, n_iter=1, huber_k=1.5):
    """加权最小二乘；n_iter>1 时做 Huber 迭代重加权（稳健回归）。

    用途：真实动脉波形的舒张期并非纯指数（早期有二重波/反射隆起），
    该隆起在 ln P 上表现为正向离群残差；Huber 权重会自动把它压下去，
    从而无需精确几何定位"切迹"即可得到稳健的 tau。
    """
    W = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    for _ in range(max(0, n_iter - 1)):
        r = y - X @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        if not np.isfinite(s) or s <= 0:
            break
        z = np.abs(r) / s
        u = np.where(z <= huber_k, 1.0, huber_k / np.maximum(z, 1e-9))
        W = np.sqrt(w * u)
        coef, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    return coef


def fit_tau(p, fs, peaks, frac_start=0.25, frac_end=0.98, robust=False,
            delay_s=None, max_beats=None, qc=True,
            qc_min_pressure=0.0, qc_pp=(15.0, 120.0), qc_tau=(0.2, 5.0)):
    """共享斜率加权最小二乘拟合舒张期衰减常数 tau（含逐搏质控）。

    舒张期界定：每个搏的舒张期 = 【收缩峰 -> 该峰与下一峰之间的压力最低点(谷)】。
    这一步是必需的：射血上升段约占搏长的 35%，若用"峰->下一峰"作分母会把
    上升段算进拟合窗，导致 ln P 斜率变正、tau 无法估计。

    拟合起点二选一：
      delay_s 不为 None -> 起点 = 收缩峰 + delay_s 秒（固定时间延迟）；
      delay_s 为 None   -> 起点 = 收缩峰 + frac_start*span（固定比例）。
    固定延迟更稳：二重波在射血结束后一个近乎固定的时间内出现，用时间截断
    比用比例截断更不容易在高心率（短舒张期）下漏进隆起区。

    模型: ln P_ij = a_j - t_ij / tau  （j 为搏号；各搏独立截距、共享斜率）
    权重: w = P^2（因 ln P 的方差 ≈ sigma_P^2 / P^2）
    robust=True 时叠加 Huber 迭代重加权，进一步抑制残余隆起的影响。

    逐搏质控（qc=True，真实波形必需；仿真波形全部通过，不改变回收结果）：
      仅当一整搏同时满足以下条件才纳入共享斜率——
        1) 整搏最小压力 >= qc_min_pressure（剔除负 ABP / 传感器校零冲管伪差）；
        2) 整搏脉压 SBP-DBP 在 qc_pp 内（剔除异常脉压搏）；
        3) 该搏单独拟合的 tau 有限且落在 qc_tau 内（剔除形态异常搏，
           这是修复"共享 tau 被个别异常搏拉到 17.7s"的关键）。
      qc=False 时退回旧逻辑（凡舒张窗 pseg>0、点数够的搏全部纳入）。

    求解（内存安全）：按 Frisch-Waugh-Lovell 定理对各搏截距组内去心后取
    共享斜率闭式解 b = Σ_j S_ty,j / Σ_j S_tt,j，内存仅 O(点数)。

    max_beats: 参与共享拟合的（已通过质控）搏数上限，整段均匀抽选，None 为全部。

    返回 dict：tau（共享斜率）、n_pts、per_beat（全部有效逐搏 tau）、
    per_beat_qc（通过质控的逐搏 tau）、n_beats_seen / n_beats_qc / n_beats_used。
    """
    beats = []          # 通过质控、参与共享拟合: (tt, yy, base_w)
    per_beat = []       # 所有能单独拟合(衰减斜率为负)的搏 tau
    per_beat_qc = []    # 通过质控的搏 tau
    n_seen = 0
    for j in range(len(peaks) - 1):
        i0, i1 = peaks[j], peaks[j + 1]
        trough = i0 + int(np.argmin(p[i0:i1]))      # 舒张末最低点
        span = trough - i0
        if span < 8:
            continue
        if delay_s is not None:
            s = i0 + int(round(delay_s * fs))
        else:
            s = i0 + int(round(frac_start * span))
        s = max(s, i0 + 1)
        e = i0 + int(round(frac_end * span))
        if e - s < 5:
            continue
        pbeat = p[i0:i1]                # 完整搏，用于生理质控
        pseg = p[s:e]
        if np.any(pseg <= 0):
            continue
        n_seen += 1
        tt = (np.arange(s, e) - i0) / fs
        yy = np.log(pseg)
        ww = pseg ** 2
        A = np.vstack([np.ones_like(tt), tt]).T
        coef = _wls_slope(A, yy, ww, n_iter=1)
        tau_b = -1.0 / coef[1] if coef[1] < 0 else np.nan
        if np.isfinite(tau_b):
            per_beat.append(tau_b)

        passed = True
        if qc:
            pp_b = float(pbeat.max() - pbeat.min())
            min_b = float(pbeat.min())
            if not (np.isfinite(tau_b) and qc_tau[0] <= tau_b <= qc_tau[1]):
                passed = False
            if min_b < qc_min_pressure:
                passed = False
            if not (qc_pp[0] <= pp_b <= qc_pp[1]):
                passed = False
        if passed:
            beats.append((tt, yy, ww.astype(float)))
            if np.isfinite(tau_b):
                per_beat_qc.append(tau_b)

    if not beats:
        return {"tau": np.nan, "n_pts": 0,
                "per_beat": np.array(per_beat), "per_beat_qc": np.array(per_beat_qc),
                "n_beats_seen": n_seen, "n_beats_qc": 0, "n_beats_used": 0}

    if max_beats is not None and len(beats) > max_beats:
        sel = np.unique(np.linspace(0, len(beats) - 1, max_beats).round().astype(int))
        beats = [beats[i] for i in sel]

    npts = int(sum(len(b[0]) for b in beats))
    wcur = [b[2].copy() for b in beats]
    slope = np.nan
    for _ in range(10 if robust else 1):
        S_tt = 0.0
        S_ty = 0.0
        groups = []
        for k, (tt, yy, ww) in enumerate(beats):
            w = wcur[k]
            W = w.sum()
            tb = float((w * tt).sum() / W)
            yb = float((w * yy).sum() / W)
            dt = tt - tb
            S_tt += float((w * dt * dt).sum())
            S_ty += float((w * dt * (yy - yb)).sum())
            groups.append((tb, yb))
        slope = S_ty / S_tt
        if not robust:
            break
        chunks = []
        for k, (tt, yy, ww) in enumerate(beats):
            tb, yb = groups[k]
            chunks.append(yy - yb - slope * (tt - tb))
        rall = np.concatenate(chunks)
        s = 1.4826 * np.median(np.abs(rall - np.median(rall)))
        if not np.isfinite(s) or s <= 0:
            break
        z = np.abs(rall) / s
        u = np.where(z <= HUBER_K, 1.0, HUBER_K / np.maximum(z, 1e-9))
        pos = 0
        for k, (tt, yy, ww) in enumerate(beats):
            n = len(tt)
            wcur[k] = ww * u[pos:pos + n]
            pos += n

    tau = -1.0 / slope if slope < 0 else np.nan
    return {"tau": tau, "n_pts": npts,
            "per_beat": np.array(per_beat), "per_beat_qc": np.array(per_beat_qc),
            "n_beats_seen": n_seen, "n_beats_qc": len(per_beat_qc),
            "n_beats_used": len(beats)}


def invert_waveform(p, fs=FS_DEFAULT, CO=None, frac_start=0.25, frac_end=0.98,
                    min_beats=2, robust=False, delay_s=None, max_beats=None,
                    qc=True, qc_min_pressure=0.0, qc_pp=(15.0, 120.0),
                    qc_tau=(0.2, 5.0)):
    """最小二乘 Windkessel 反演主入口（自包含，仅用 ABP 波形 + 可选 CO）。

    CO 为 None 时只返回 tau（自包含可辨识量），R_hat/C_hat 为 nan。
    max_beats 透传给 fit_tau（超长连续段的搏数上限）。
    qc* 为逐搏质控参数（见 fit_tau）；真实波形应保持 qc=True。

    输出关键量：
      tau_hat   : 通过质控搏的【共享斜率】tau；
      tau_robust: 通过质控搏的【逐搏 tau 中位数】，最稳健，真实数据主用；
      n_beats_qc: 通过质控的搏数；qc_pass_rate 为其占检出搏比例。
    """
    peaks = detect_peaks(p, fs)
    out = {"n_beats": len(peaks), "tau_hat": np.nan, "n_pts": 0,
           "R_hat": np.nan, "C_hat": np.nan, "MAP_hat": np.nan,
           "tau_per_beat_median": np.nan, "tau_robust": np.nan,
           "n_beats_qc": 0, "qc_pass_rate": np.nan, "ok": False}
    if len(peaks) < min_beats + 1:
        return out

    i0, i1 = peaks[0], peaks[-1]            # 只用完整搏，避开截断偏倚
    MAP_hat = float(np.mean(p[i0:i1]))
    out["MAP_hat"] = MAP_hat

    fit = fit_tau(p, fs, peaks, frac_start, frac_end,
                  robust=robust, delay_s=delay_s, max_beats=max_beats,
                  qc=qc, qc_min_pressure=qc_min_pressure, qc_pp=qc_pp, qc_tau=qc_tau)
    out["tau_hat"] = fit["tau"]
    out["n_pts"] = fit["n_pts"]
    out["n_beats_qc"] = fit["n_beats_qc"]
    if len(peaks) > 0:
        out["qc_pass_rate"] = fit["n_beats_qc"] / max(len(peaks) - 1, 1)
    if fit["per_beat"].size:
        out["tau_per_beat_median"] = float(np.median(fit["per_beat"]))
    if fit["per_beat_qc"].size:
        out["tau_robust"] = float(np.median(fit["per_beat_qc"]))

    # R/C 拆分优先用稳健 tau；回退共享 tau
    tau_use = out["tau_robust"] if np.isfinite(out["tau_robust"]) else fit["tau"]
    if np.isfinite(tau_use) and tau_use > 0 and CO is not None and CO > 0:
        R_hat = MAP_hat / CO                              # mmHg*min/L
        out["R_hat"] = R_hat
        out["C_hat"] = tau_use / (R_hat * R_MINL_TO_SML)  # mL/mmHg
    out["ok"] = bool(np.isfinite(tau_use) and tau_use > 0)
    return out


# ----------------------------------------------------------------------------
# 3. 误差度量
# ----------------------------------------------------------------------------
def rel_err(est, true):
    if est is None or not np.isfinite(est) or true == 0:
        return np.nan
    return abs(est - true) / abs(true)


# ----------------------------------------------------------------------------
# 4. 主路径：free-P_inf 非线性最小二乘（Methods 报告的口径）
# ----------------------------------------------------------------------------
def estimate_tau_freePinf(p, fs, peaks=None, start_after_peak_s=0.15,
                          min_window_s=0.15, max_nrmse=0.08,
                          tau_bounds=(0.05, 3.0), max_beats=None):
    """逐搏拟合 ``P(t) = P_inf + (P0 - P_inf) exp(-t / tau)``，返回搏级 tau。

    与 Methods 一致的四个约束：

    1. 拟合窗 = 收缩峰后 ``start_after_peak_s`` 秒 到 下一个谷（舒张末最低点）；
    2. 窗长不足 ``min_window_s`` 秒的搏直接丢弃；
    3. ``tau`` 约束在 ``tau_bounds`` 内，且 ``P_inf`` 不得高于该窗舒张压最小值
       （否则指数外推会得到非生理的负衰减）；
    4. 归一化 RMSE ``NRMSE = RMSE / (SBP - DBP)`` 大于 ``max_nrmse`` 的搏剔除。

    Parameters
    ----------
    p : ndarray
        Pressure samples (mmHg).
    fs : float
        Sampling rate (Hz).
    peaks : ndarray, optional
        Systolic peak indices; detected with :func:`detect_peaks` when omitted.

    Returns
    -------
    dict
        ``tau`` (median of accepted beat-level estimates, seconds),
        ``tau_beats`` (per-beat accepted values), ``n_beats_seen``,
        ``n_beats_accepted``, ``accept_rate``, ``median_nrmse``.
    """
    from scipy.optimize import curve_fit

    if peaks is None:
        peaks = detect_peaks(p, fs)
    out = {"tau": np.nan, "tau_beats": np.array([]), "n_beats_seen": 0,
           "n_beats_accepted": 0, "accept_rate": np.nan, "median_nrmse": np.nan}
    if len(peaks) < 2:
        return out

    def model(t, p0, p_inf, tau):
        return p_inf + (p0 - p_inf) * np.exp(-t / tau)

    taus, nrmse_list, n_seen = [], [], 0
    n_pairs = len(peaks) - 1
    if max_beats is not None and n_pairs > max_beats:
        sel = np.unique(np.linspace(0, n_pairs - 1, max_beats).round().astype(int))
    else:
        sel = np.arange(n_pairs)

    for j in sel:
        i0, i1 = peaks[j], peaks[j + 1]
        trough = i0 + int(np.argmin(p[i0:i1]))
        s = i0 + int(round(start_after_peak_s * fs))
        if trough - s < int(min_window_s * fs):
            continue
        seg = p[s:trough].astype(float)
        if seg.size < 5 or np.any(~np.isfinite(seg)) or np.any(seg <= 0):
            continue
        n_seen += 1
        t = np.arange(seg.size) / fs
        p_dia_min = float(np.min(seg))
        try:
            popt, _ = curve_fit(
                model, t, seg,
                p0=[float(seg[0]), p_dia_min * 0.9, 0.5],
                bounds=([0.0, 0.0, tau_bounds[0]], [np.inf, p_dia_min, tau_bounds[1]]),
                maxfev=4000,
            )
        except Exception:
            continue
        p0_hat, p_inf_hat, tau_hat = popt
        if not (np.isfinite(tau_hat) and tau_bounds[0] < tau_hat < tau_bounds[1]):
            continue
        if p_inf_hat >= p_dia_min:
            continue
        resid = seg - model(t, *popt)
        pp = float(p[i0:i1].max() - p[i0:i1].min())
        if pp <= 0:
            continue
        nrmse = float(np.sqrt(np.mean(resid ** 2)) / pp)
        nrmse_list.append(nrmse)
        if nrmse <= max_nrmse:
            taus.append(tau_hat)

    out["n_beats_seen"] = n_seen
    out["tau_beats"] = np.asarray(taus)
    out["n_beats_accepted"] = len(taus)
    out["accept_rate"] = len(taus) / n_seen if n_seen else np.nan
    if nrmse_list:
        out["median_nrmse"] = float(np.median(nrmse_list))
    if taus:
        out["tau"] = float(np.median(taus))
    return out
