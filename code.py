
import os, re, glob, json, zipfile, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ PATHS
try:
    from google.colab import drive
    drive.mount("/content/drive")
    ZIP_PATH = "/content/drive/MyDrive/Research/2026 mech/CAN Data.zip"
    DATA_DIR = "/content/can_data"
    OUT = "/content/drive/MyDrive/Research/2026 mech/BE_final"
except ImportError:                                   # local run / testing
    ZIP_PATH = os.environ.get("ZIP_PATH", "")
    DATA_DIR = os.environ.get("DATA_DIR", "./can_data")
    OUT = os.environ.get("OUT", "./BE_final")
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for p in (DATA_DIR, FIG, TAB): os.makedirs(p, exist_ok=True)

# ------------------------------------------------------------------ CONSTANTS (Table 3)
FS        = 10            # Hz
M_R       = 450.0         # J1939 reference torque [Nm]
P_N       = 78.3          # rated engine power [kW] (105 hp)
K_PTO     = 0.90          # PTO-to-engine power ratio
N_RT      = 2200          # rated engine speed [rpm]  (assumed — verify spec sheet)
W_PLOUGH  = 1.524         # plough width [m]
RHO_F     = 0.835         # diesel density [kg/L]
WIN_S     = 10            # aggregation window [s]
MIN_OUT_S, MIN_IN_S = 5, 10
DEEP_THR  = 30            # median in-work hitch [%] < 30 -> deep
K_DB      = 0.73          # drawbar-to-PTO ratio (Appendix)
DRAFT_A, DRAFT_B, DRAFT_C, DRAFT_F = 652.0, 0.0, 5.1, 1.0   # D497 mouldboard draft
DRAFT_RIGHT = "HtcDraftSens2Perc"   # right lower-link draft channel [verify: 1 or 2]
FIELD10_AS_PLOUGH = True
N_BOOT, SEED = 1000, 42
rng = np.random.default_rng(SEED)

MODELS = {"4974": "ASAE D497.4", "FT": "D497.7 FT", "PT": "D497.7 PT", "RC": "Recalibrated (LOFO)"}
MCOL   = {"4974": "#b2182b", "FT": "#ef8a62", "PT": "#1b7837", "RC": "#2166ac"}
SCOL   = {"deep": "#2166ac", "shallow": "#ef8a62", "rotary": "#1b7837"}

# ------------------------------------------------------------------ FIGURE STYLE (Elsevier)
MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "axes.linewidth": .6,
    "xtick.major.width": .6, "ytick.major.width": .6, "lines.linewidth": 1.0,
    "axes.grid": False, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42})

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)

def um(s):
    """Typographic minus for labels."""
    return s.replace("-", "\u2212").replace("h$^{\u22121}$", "h$^{-1}$")

def plabel(ax, s):
    ax.set_title(s, loc="left", fontsize=8, pad=3)

# ------------------------------------------------------------------ D497 MODELS
def q_ft(X, Pp):
    X = np.clip(X, 0, 1.2); return (0.22 * X + 0.096) * Pp
def q_pt(X, n, Pp, nrt=N_RT):
    r = np.clip(n / nrt, .5, 1.0); X = np.clip(X, 0, 1.2)
    return q_ft(X, Pp) * (1 - (r - 1) * (0.45 * X - 0.877))
def q_4974(X, Pp):
    X = np.clip(X, 1e-3, 1.2)
    return (2.64 * X + 3.91 - 0.203 * np.sqrt(738 * X + 173)) * X * Pp

# ------------------------------------------------------------------ LOADING
COLS = ["EngineSpeed", "ActualEngine_PercTorque", "EngFuelRate", "WheelBasedVehicleSpeed",
        "HtcPositionSensPerc", "HtcDraftSens1Perc", "HtcDraftSens2Perc", "PTO_Status", "KeypadButton"]

def load_raw():
    pat = f"{DATA_DIR}/**/FIELD_*_RAW.tab"
    if not glob.glob(pat, recursive=True) and ZIP_PATH:
        with zipfile.ZipFile(ZIP_PATH) as z: z.extractall(DATA_DIR)
    out = {}
    for p in glob.glob(pat, recursive=True):
        if "_rar_" in p: continue
        fid = int(re.search(r"FIELD_(\d+)_", os.path.basename(p)).group(1))
        df = pd.read_csv(p, sep="\t", low_memory=False)
        out[fid] = df[[c for c in COLS if c in df]].apply(pd.to_numeric, errors="coerce")
    if not out: raise FileNotFoundError("No FIELD_*_RAW.tab files found")
    return dict(sorted(out.items()))

# ------------------------------------------------------------------ SEGMENTATION (Sec. 2.2)
def otsu(x):
    x = x[(x >= 0) & (x <= 100)]
    h, e = np.histogram(x, 100, (0, 100)); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])

def runs(a):
    idx = np.flatnonzero(np.r_[True, a[1:] != a[:-1], True])
    return [(a[s], s, e) for s, e in zip(idx[:-1], idx[1:])]

def debounce(s, min_out, min_in):
    s = s.copy()
    for tgt, ml in ((0, min_out), (1, min_in)):
        for v, a, b in runs(s):
            if v == tgt and b - a < ml and a > 0 and b < len(s): s[a:b] = 1 - tgt
    return s

def segment(fid, raw):
    d = raw.copy()
    run = (d.EngineSpeed >= 500).to_numpy()
    h = d.HtcPositionSensPerc.ffill().bfill()
    thr = otsu(h.to_numpy())
    work = debounce((h <= thr).astype(int).to_numpy(), MIN_OUT_S * FS, MIN_IN_S * FS).astype(bool)
    d["phase"] = np.where(~run, "off", np.where(work, "work", "head"))
    kp = d.KeypadButton.dropna()
    impl = "rotary" if (len(kp) and kp.mode()[0] == 2) else "plough"
    if fid == 10 and FIELD10_AS_PLOUGH: impl = "plough"
    hmed = h[d.phase == "work"].median()
    setting = "rotary" if impl == "rotary" else ("deep" if hmed < DEEP_THR else "shallow")
    wr = [r for r in runs(work & run) if r[0]]
    d.attrs.update(fid=fid, thr=thr, implement=impl, setting=setting, hitch_med=hmed,
                   turns=max(len(wr) - 1, 0))
    return d

def power(d, mf=0.0, tq_scale=1.0):
    """Engine power [kW] and load ratio, Eq. (1)-(2); negative torque set to zero."""
    me = d.ActualEngine_PercTorque.clip(lower=0) * tq_scale
    pe = (M_R * (me - mf) / 100 * d.EngineSpeed * 2 * np.pi / 60 / 1000).clip(lower=0)
    pe = pe.where(d.phase != "off", 0.0)
    return pe.to_numpy(), (pe / P_N).to_numpy()

# ------------------------------------------------------------------ WINDOWS
def windows(d, X, win_s=WIN_S):
    n = win_s * FS
    df = pd.DataFrame({"g": np.arange(len(d)) // n, "ph": d.phase.values, "X": X,
                       "rpm": d.EngineSpeed.values, "fuel": d.EngFuelRate.values})
    a = df.groupby("g").agg(n=("X", "size"), nph=("ph", "nunique"), phase=("ph", "first"),
                            X=("X", "mean"), rpm=("rpm", "mean"), fuel=("fuel", "mean"))
    a = a[(a.n == n) & (a.nph == 1) & (a.phase != "off")].dropna(subset=["X", "rpm", "fuel"])
    a["field"] = d.attrs["fid"]
    return a.drop(columns=["n", "nph"]).reset_index(drop=True)

def add_preds(W, Pp=None, nrt=N_RT):
    Pp = K_PTO * P_N if Pp is None else Pp
    W["4974"] = q_4974(W.X.values, Pp)
    W["FT"] = q_ft(W.X.values, Pp)
    W["PT"] = q_pt(W.X.values, W.rpm.values, Pp, nrt)
    return W

# ------------------------------------------------------------------ RECALIBRATION (Eq. 6)
def ols_stats(W, Pp):
    """Per-field sufficient statistics for y = fuel/Pp = a X + b."""
    g = W.assign(y=W.fuel / Pp).groupby("field")
    return pd.DataFrame({"n": g.size(), "sx": g.X.sum(), "sy": g.y.sum(),
                         "sxx": g.X.apply(lambda v: (v ** 2).sum()),
                         "sxy": g.apply(lambda v: (v.X * v.y).sum())})

def ols_from(S, w=None):
    w = np.ones(len(S)) if w is None else w
    n, sx, sy, sxx, sxy = [(S[c].to_numpy() * w).sum() for c in ("n", "sx", "sy", "sxx", "sxy")]
    a = (n * sxy - sx * sy) / (n * sxx - sx ** 2); return a, (sy - a * sx) / n

def recalibrate(Wwork, Wall, Pp):
    S = ols_stats(Wwork, Pp); F = S.index.to_numpy()
    a, b = ols_from(S)
    C = rng.multinomial(len(F), np.ones(len(F)) / len(F), size=N_BOOT)
    boot = np.array([ols_from(S, c) for c in C])
    lofo = {f: ols_from(S.drop(f)) for f in F}
    rc = np.full(len(Wall), np.nan)
    for f, (af, bf) in lofo.items():
        m = (Wall.field == f).to_numpy(); rc[m] = (af * np.clip(Wall.X.values[m], 0, 1.2) + bf) * Pp
    return a, b, boot, lofo, rc

# ------------------------------------------------------------------ METRICS
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); e = p - y
    ok = y > 0.1
    return {"n": len(y), "Obs": y.mean(), "Pred": p.mean(), "MBE": 100 * e.mean() / y.mean(),
            "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "MAPE": 100 * np.mean(np.abs(e[ok]) / y[ok]),
            "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "r": stats.pearsonr(y, p)[0] if len(y) > 2 else np.nan,
            "w10": 100 * np.mean(np.abs(e) <= .10 * y), "w15": 100 * np.mean(np.abs(e) <= .15 * y)}

def tci(v):
    v = np.asarray(v, float); n = len(v); h = stats.t.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return v.mean(), v.mean() - h, v.mean() + h

# ------------------------------------------------------------------ LaTeX TABLE WRITER
def f_(d=1, sign=False):
    def g(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)): return "--"
        if isinstance(v, str): return v
        s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
        return s.replace("-", "$-$") if not sign else s.replace("+", "$+$").replace("-", "$-$")
    return g
fS = lambda v: str(v)

def write_tex(name, df, cols, caption, label, notes=None, extra=None, size="\\footnotesize",
              sideways=False, colfmt=None, sep=4):
    """cols: list of (column, header, unit, formatter)."""
    env = "sidewaystable" if sideways else "table"
    if colfmt is None:
        al = ["l" if i == 0 else ("c" if c[1] in ("Set.", "Setting", "FE vs", "Speed vs") else "r")
              for i, c in enumerate(cols)]
        colfmt = "@{}" + "".join(al) + "@{}"
    L = [f"\\begin{{{env}}}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         size, f"\\setlength{{\\tabcolsep}}{{{sep}pt}}", f"\\begin{{tabular}}{{{colfmt}}}", "\\toprule",
         " & ".join(c[1] for c in cols if c[1] is not None) + " \\\\",
         " & ".join(c[2] for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        L.append(" & ".join(c[3](r[c[0]]) for c in cols) + " \\\\")
    if extra:
        L.append("\\midrule")
        for row in extra: L.append(" & ".join(row) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    if notes: L.append(f"\\\\[2pt]\\parbox{{\\linewidth}}{{\\scriptsize {notes}}}")
    L.append(f"\\end{{{env}}}")
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8") as fh: fh.write("\n".join(L))
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    TABLES[name] = df

TABLES, KEY = {}, {}

# =============================================================================
#  MAIN PIPELINE (baseline)
# =============================================================================
RAW = load_raw()
D = {f: segment(f, r) for f, r in RAW.items()}
PP = K_PTO * P_N
PE, XX = {}, {}
for f, d in D.items(): PE[f], XX[f] = power(d)
W = add_preds(pd.concat([windows(D[f], XX[f]) for f in D], ignore_index=True))
SET = {f: d.attrs["setting"] for f, d in D.items()}
a_hat, b_hat, BOOT, LOFO, W["RC"] = recalibrate(W[W.phase == "work"], W, PP)
wk, hd = W[W.phase == "work"], W[W.phase == "head"]
FIELDS = sorted(D)

# ------------------------------------------------------------------ TABLE 4 — session summary
rows = []
for f, d in D.items():
    w = d.phase == "work"; run = d.phase != "off"
    tq = (d.ActualEngine_PercTorque.clip(lower=0) / 100 * M_R)[w]
    pe_w = PE[f][w.to_numpy()]
    fuel_w = d.EngFuelRate[w].fillna(0)
    rows.append({"Field": f, "Set": SET[f][0].upper(), "Dur": len(d) / FS / 3600,
                 "Work": w.sum() / FS / 3600, "Thr": d.attrs["thr"], "Hmed": d.attrs["hitch_med"],
                 "v": d.WheelBasedVehicleSpeed[w].mean(), "rpm": d.EngineSpeed[w].mean(),
                 "Tq": tq.mean(), "Pe": pe_w.mean(), "X": XX[f][w.to_numpy()].mean(),
                 "X95": np.nanpercentile(XX[f][w.to_numpy()], 95), "Fuel": fuel_w.mean(),
                 "FuelTot": d.EngFuelRate[run].fillna(0).sum() / FS / 3600,
                 "BSFC": 1000 * RHO_F * fuel_w.sum() / pe_w.sum(), "Turns": d.attrs["turns"]})
T4 = pd.DataFrame(rows)
wsum = T4.Work.sum()
write_tex("Tab04_session_summary", T4, [
    ("Field", "Field", "", fS), ("Set", "Set.", "", fS), ("Dur", "Duration", "[h]", f_(2)),
    ("Work", "Work", "[h]", f_(2)), ("Thr", "Hitch thr.", "[\\%]", f_(1)),
    ("Hmed", "Hitch med.", "[\\%]", f_(0)), ("v", "Speed", "[km\\,h$^{-1}$]", f_(2)),
    ("rpm", "$n_e$", "[rpm]", f_(0)), ("Tq", "Torque", "[Nm]", f_(1)), ("Pe", "$P_e$", "[kW]", f_(1)),
    ("X", "$X$", "[--]", f_(2)), ("X95", "$X_{95}$", "[--]", f_(2)),
    ("Fuel", "Fuel", "[L\\,h$^{-1}$]", f_(2)), ("FuelTot", "Fuel total", "[L]", f_(2)),
    ("BSFC", "BSFC", "[g\\,kWh$^{-1}$]", f_(1)), ("Turns", "Turns", "[--]", fS)],
    "Session summary. Values refer to working passes unless stated otherwise. D: deep plough; S: shallow plough; R: rotary tiller; $X_{95}$: 95th percentile of $X$.",
    "tab:session", sideways=True,
    extra=[["Total/mean", "--", f"{T4.Dur.sum():.2f}", f"{wsum:.2f}", "--", "--", "--",
            f"{(T4["rpm"] * T4.Work).sum() / wsum:.0f}$^{{a}}$", "--", "--",
            f"{(T4.X * T4.Work).sum() / wsum:.2f}$^{{a}}$", "--", "--",
            f"{T4.FuelTot.sum():.2f}", "--", f"{T4.Turns.sum()}"]],
    notes="$^{a}$ Weighted by working time.")
KEY["session"] = {"hours": T4.Dur.sum(), "work_h": wsum, "work_share_%": 100 * wsum / T4.Dur.sum(),
                  "turns": int(T4.Turns.sum()), "rpm_range": [T4["rpm"].min(), T4["rpm"].max()],
                  "rpm_ratio_range_%": [100 * T4["rpm"].min() / N_RT, 100 * T4["rpm"].max() / N_RT],
                  "X_weighted": (T4.X * T4.Work).sum() / wsum, "fuel_total_L": T4.FuelTot.sum(),
                  "windows_work": len(wk), "windows_head": len(hd)}

# ------------------------------------------------------------------ TABLE 5 — pooled accuracy
rows = []
for m in MODELS:
    for ph, sub in (("Work", wk), ("Headland", hd), ("All", W)):
        r = metrics(sub.fuel, sub[m]); r.update(Model=MODELS[m], Phase=ph); rows.append(r)
T5 = pd.DataFrame(rows)
T5["Model"] = T5.Model.where(T5.Phase == "Work", "")
write_tex("Tab05_pooled_accuracy", T5, [
    ("Model", "Model", "", fS), ("Phase", "Phase", "", fS), ("n", "$n$", "", fS),
    ("Obs", "Obs.", "[L\\,h$^{-1}$]", f_(2)), ("Pred", "Pred.", "[L\\,h$^{-1}$]", f_(2)),
    ("MBE", "MBE", "[\\%]", f_(1, True)),
    ("RMSE", "RMSE", "[L\\,h$^{-1}$]", f_(2)), ("MAPE", "MAPE", "[\\%]", f_(1)),
    ("R2", "$R^2$", "[--]", f_(2)), ("r", "$r$", "[--]", f_(3)), ("w10", "$\\pm$10\\%", "[\\%]", f_(1))],
    "Pooled accuracy of the fuel models for working passes, headland turns and all 10\\,s windows. The recalibrated equation is evaluated out of sample: each field is predicted with coefficients fitted on the other nine fields (LOFO). Pooled window-level values are descriptive (Section~2.7).",
    "tab:pooled", colfmt="@{}llrrrrrrrrr@{}", sep=3.5)
KEY["pooled"] = {f"{m}_{ph}": {k: float(v) for k, v in metrics(s.fuel, s[m]).items()}
                 for m in MODELS for ph, s in (("work", wk), ("head", hd), ("all", W))}

# ------------------------------------------------------------------ TABLE 7 — per-field accuracy
rows = []
for f in FIELDS:
    s = wk[wk.field == f]; r = {"Field": f, "Set": SET[f][0].upper(), "n": len(s), "Obs": s.fuel.mean()}
    for m in ("4974", "FT", "PT"):
        mm = metrics(s.fuel, s[m]); r[f"MBE_{m}"], r[f"RMSE_{m}"] = mm["MBE"], mm["RMSE"]
    r["w10_PT"] = metrics(s.fuel, s.PT)["w10"]; rows.append(r)
T7 = pd.DataFrame(rows)
mean_row = ["\\multicolumn{4}{@{}l}{Field-level mean ($n=10$)}"]
ci_row = ["\\multicolumn{4}{@{}l}{95\\% CI ($t_{9}$)}"]
for c in ["MBE_4974", "RMSE_4974", "MBE_FT", "RMSE_FT", "MBE_PT", "RMSE_PT", "w10_PT"]:
    mu, lo, hi = tci(T7[c]); sgn = c.startswith("MBE")
    mean_row.append(f_(1 if sgn or c.startswith("w") else 2, sgn)(mu))
    ci_row.append(f"{f_(1 if sgn or c.startswith('w') else 2)(lo)}, {f_(1 if sgn or c.startswith('w') else 2)(hi)}")
    KEY.setdefault("field_level_work", {})[c] = [mu, lo, hi]
write_tex("Tab07_per_field_accuracy", T7, [
    ("Field", "Field", "", fS), ("Set", "Set.", "", fS), ("n", "$n$", "", fS),
    ("Obs", "Obs.", "", f_(2)),
    ("MBE_4974", "\\multicolumn{2}{c}{D497.4}", "MBE", f_(1, True)), ("RMSE_4974", None, "RMSE", f_(2)),
    ("MBE_FT", "\\multicolumn{2}{c}{D497.7 FT}", "MBE", f_(1, True)), ("RMSE_FT", None, "RMSE", f_(2)),
    ("MBE_PT", "\\multicolumn{3}{c}{D497.7 PT}", "MBE", f_(1, True)), ("RMSE_PT", None, "RMSE", f_(2)),
    ("w10_PT", None, "$\\pm$10\\%", f_(1))],
    "Per-field accuracy of the ASABE models during working passes: mean measured fuel rate (Obs., L\\,h$^{-1}$), MBE (\\%), RMSE (L\\,h$^{-1}$) and, for the partial-throttle model, the share of windows within $\\pm$10\\%. Inference is made at field level ($n=10$).",
    "tab:perfield", extra=[mean_row, ci_row], sep=3)

# ------------------------------------------------------------------ TABLE 8 — session fuel totals
lofo_coef = LOFO
rows = []
for f, d in D.items():
    run = (d.phase != "off").to_numpy(); X = np.clip(XX[f][run], 0, 1.2); n = d.EngineSpeed.to_numpy()[run]
    meas = d.EngFuelRate.to_numpy()[run]; meas = np.nan_to_num(meas).sum() / FS / 3600
    af, bf = lofo_coef[f]
    pred = {"4974": q_4974(X, PP).sum(), "FT": q_ft(X, PP).sum(), "PT": q_pt(X, n, PP).sum(),
            "RC": ((af * X + bf) * PP).sum()}
    r = {"Field": f, "Set": SET[f][0].upper(), "Meas": meas}
    for m, v in pred.items():
        r[m] = v / FS / 3600; r[f"E_{m}"] = 100 * (r[m] - meas) / meas
    rows.append(r)
T8 = pd.DataFrame(rows)
tot = ["Total", "--", f"{T8.Meas.sum():.1f}"]
ptot = ["Predicted total [L]", "", ""]
for m in MODELS:
    tot.append(f_(1, True)(100 * (T8[m].sum() - T8.Meas.sum()) / T8.Meas.sum()))
    ptot.append(f"{T8[m].sum():.1f}")
ci, mrow = ["95\\% CI ($t_{9}$)", "", ""], ["Field-level mean", "", ""]
for m in MODELS:
    mu, lo, hi = tci(T8[f"E_{m}"]); mrow.append(f_(1, True)(mu)); ci.append(f"{f_(1)(lo)}, {f_(1)(hi)}")
    KEY.setdefault("session_error_CI", {})[m] = [mu, lo, hi]
KEY["session_totals_L"] = {"measured": T8.Meas.sum(), **{m: T8[m].sum() for m in MODELS}}
write_tex("Tab08_session_totals", T8, [
    ("Field", "Field", "", fS), ("Set", "Set.", "", fS), ("Meas", "Measured", "[L]", f_(2)),
    ("E_4974", "D497.4", "[\\%]", f_(1, True)), ("E_FT", "D497.7 FT", "[\\%]", f_(1, True)),
    ("E_PT", "D497.7 PT", "[\\%]", f_(1, True)), ("E_RC", "Recal. (LOFO)", "[\\%]", f_(1, True))],
    "Measured fuel per session and relative error of the predicted totals, including headland turns (integrated at 10\\,Hz). The recalibrated equation is applied out of sample (LOFO).",
    "tab:sessiontotal", extra=[tot, ptot, mrow, ci])

# ------------------------------------------------------------------ TABLE 9 — recalibrated coefficients
la = np.array([v[0] for v in LOFO.values()]); lb = np.array([v[1] for v in LOFO.values()])
T9 = pd.DataFrame([
    {"Coef": "$a$ (slope)", "D497": 0.220, "Est": a_hat, "lo": np.percentile(BOOT[:, 0], 2.5),
     "hi": np.percentile(BOOT[:, 0], 97.5), "Lm": la.mean(), "Ls": la.std(ddof=1)},
    {"Coef": "$b$ (intercept)", "D497": 0.096, "Est": b_hat, "lo": np.percentile(BOOT[:, 1], 2.5),
     "hi": np.percentile(BOOT[:, 1], 97.5), "Lm": lb.mean(), "Ls": lb.std(ddof=1)}])
T9["CI"] = T9.apply(lambda r: f"{r.lo:.4f}--{r.hi:.4f}", axis=1)
write_tex("Tab09_recal_coefficients", T9, [
    ("Coef", "Coefficient", "", fS), ("D497", "ASABE D497.7", "[L\\,kWh$^{-1}$]", f_(3)),
    ("Est", "This study", "[L\\,kWh$^{-1}$]", f_(4)), ("CI", "95\\% CI (bootstrap)", "", fS),
    ("Lm", "LOFO mean", "", f_(4)), ("Ls", "LOFO SD", "", f_(4))],
    "Recalibrated coefficients of $\\dot{f}_d/P_{\\mathrm{PTO}} = aX + b$ (working passes, indicated-torque basis) compared with ASABE D497.7. CI: field-cluster bootstrap (1000 resamples).",
    "tab:recal")
KEY["recal"] = {"a": a_hat, "b": b_hat, "intercept_Lh": b_hat * PP,
                "a_CI": [np.percentile(BOOT[:, 0], 2.5), np.percentile(BOOT[:, 0], 97.5)],
                "b_CI": [np.percentile(BOOT[:, 1], 2.5), np.percentile(BOOT[:, 1], 97.5)]}
lowX = hd[(hd.X < 0.05)]
KEY["measured_near_idle_fuel_Lh"] = {"mean": lowX.fuel.mean(), "median": lowX.fuel.median(),
                                     "rpm_median": lowX.rpm.median(), "n": len(lowX)}

# ------------------------------------------------------------------ TABLE 10 — LOFO validation
rows = []
for f in FIELDS:
    s, h = wk[wk.field == f], hd[hd.field == f]; m = metrics(s.fuel, s.RC)
    rows.append({"Field": f, "a": LOFO[f][0], "b": LOFO[f][1], "n": m["n"], "MBE": m["MBE"],
                 "RMSE": m["RMSE"], "MAPE": m["MAPE"], "R2": m["R2"], "w10": m["w10"],
                 "MBEh": metrics(h.fuel, h.RC)["MBE"] if len(h) > 2 else np.nan})
T10 = pd.DataFrame(rows)
mean = ["Mean", f"{T10.a.mean():.4f}", f"{T10.b.mean():.4f}", "--", f_(1, True)(T10.MBE.mean()),
        f"{T10.RMSE.mean():.2f}", f"{T10.MAPE.mean():.1f}", f"{T10.R2.mean():.3f}", f"{T10.w10.mean():.1f}",
        f_(1, True)(T10.MBEh.mean())]
write_tex("Tab10_LOFO", T10, [
    ("Field", "Held-out", "field", fS), ("a", "$a$", "", f_(4)), ("b", "$b$", "", f_(4)), ("n", "$n$", "", fS),
    ("MBE", "MBE", "[\\%]", f_(1, True)), ("RMSE", "RMSE", "[L\\,h$^{-1}$]", f_(2)),
    ("MAPE", "MAPE", "[\\%]", f_(1)), ("R2", "$R^2$", "[--]", f_(3)), ("w10", "$\\pm$10\\%", "[\\%]", f_(1)),
    ("MBEh", "Headland MBE", "[\\%]", f_(1, True))],
    "Leave-one-field-out validation of the recalibrated equation: coefficients fitted on nine fields (working passes) and metrics on the held-out field for working passes and, out of the fitting domain, headland turns.",
    "tab:lofo", extra=[mean])

# ------------------------------------------------------------------ TABLE 11 — sensitivity
def run_case(mf=0.0, tq=1.0, kpto=K_PTO, nrt=N_RT, win=WIN_S, drop10=False):
    Pp = kpto * P_N; Ws = []
    for f, d in D.items():
        if drop10 and f == 10: continue
        _, X = power(d, mf, tq); Ws.append(windows(d, X, win))
    Wc = add_preds(pd.concat(Ws, ignore_index=True), Pp, nrt)
    w_, h_ = Wc[Wc.phase == "work"], Wc[Wc.phase == "head"]
    S = ols_stats(w_, Pp); a, b = ols_from(S)
    r = {"n": len(w_)}
    for m in ("4974", "FT", "PT"):
        mm = metrics(w_.fuel, w_[m]); r[f"MBE_{m}"], r[f"RMSE_{m}"] = mm["MBE"], mm["RMSE"]
        r[f"H_{m}"] = metrics(h_.fuel, h_[m])["MBE"]
    r["a"], r["b"] = a, b
    return r
CASES = [("Baseline", {}), ("Friction torque 5\\%", {"mf": 5}), ("Friction torque 10\\%", {"mf": 10}),
         ("Torque $+5\\%$", {"tq": 1.05}), ("Torque $+10\\%$", {"tq": 1.10}),
         ("PTO ratio 0.85", {"kpto": .85}), ("PTO ratio 1.00", {"kpto": 1.0}),
         ("Rated speed 2100\\,rpm", {"nrt": 2100}), ("Rated speed 2300\\,rpm", {"nrt": 2300}),
         ("Field 10 excluded", {"drop10": True}), ("Window 5\\,s", {"win": 5}),
         ("Window 30\\,s", {"win": 30}), ("Window 60\\,s", {"win": 60})]
T11 = pd.DataFrame([{"Case": c, **run_case(**kw)} for c, kw in CASES])
write_tex("Tab11_sensitivity", T11, [
    ("Case", "Case", "", fS), ("n", "$n$", "", fS),
    ("MBE_4974", "\\multicolumn{2}{c}{Work: D497.4}", "MBE", f_(1, True)), ("RMSE_4974", None, "RMSE", f_(2)),
    ("MBE_FT", "\\multicolumn{2}{c}{Work: D497.7 FT}", "MBE", f_(1, True)), ("RMSE_FT", None, "RMSE", f_(2)),
    ("MBE_PT", "\\multicolumn{2}{c}{Work: D497.7 PT}", "MBE", f_(1, True)), ("RMSE_PT", None, "RMSE", f_(2)),
    ("H_4974", "\\multicolumn{3}{c}{Headland MBE}", "D497.4", f_(1, True)), ("H_FT", None, "FT", f_(1, True)),
    ("H_PT", None, "PT", f_(1, True)), ("a", "\\multicolumn{2}{c}{Recalibrated}", "$a$", f_(3)), ("b", None, "$b$", f_(4))],
    "Sensitivity of the results to the main assumptions: working-pass MBE (\\%) and RMSE (L\\,h$^{-1}$), headland MBE (\\%), and recalibrated coefficients. Friction cases convert the indicated torque to an approximate brake basis; torque cases represent a systematic under-reporting of torque by the ECU.",
    "tab:sens", sideways=True)
KEY["sensitivity"] = T11.set_index("Case").round(3).to_dict(orient="index")

# ------------------------------------------------------------------ TABLE 12 — ploughing performance
rows = []
for f, d in D.items():
    if SET[f] == "rotary": continue
    w = (d.phase == "work").to_numpy(); v = d.WheelBasedVehicleSpeed.fillna(0).to_numpy()
    A = (v[w] / 3.6 / FS).sum() * W_PLOUGH / 1e4; ts = len(d) / FS / 3600; tw = w.sum() / FS / 3600
    vbar = v[w].mean(); fuel = T4.set_index("Field").FuelTot[f]
    energy = PE[f][(d.phase != "off").to_numpy()].sum() / FS / 3600
    fe = tw / ts
    rows.append({"Field": f, "Set": SET[f][0].upper(), "A": A, "v": vbar, "TFC": W_PLOUGH * vbar / 10,
                 "EFC": A / ts, "FE": fe, "Lha": fuel / A, "kWhha": energy / A,
                 "FEc": "within" if .70 <= fe <= .90 else "outside",
                 "vc": "within" if 5 <= vbar <= 10 else "outside"})
T12 = pd.DataFrame(rows)
write_tex("Tab12_ploughing", T12, [
    ("Field", "Field", "", fS), ("Set", "Set.", "", fS), ("A", "Area", "[ha]", f_(2)),
    ("v", "Speed", "[km\\,h$^{-1}$]", f_(2)), ("TFC", "TFC", "[ha\\,h$^{-1}$]", f_(2)),
    ("EFC", "EFC", "[ha\\,h$^{-1}$]", f_(2)), ("FE", "FE", "[--]", f_(2)),
    ("Lha", "Fuel", "[L\\,ha$^{-1}$]", f_(2)), ("kWhha", "Energy", "[kWh\\,ha$^{-1}$]", f_(1)),
    ("FEc", "FE vs", "D497", fS), ("vc", "Speed vs", "D497", fS)],
    "Ploughing field performance per session. Area is based on wheel speed and is an upper bound (slip not corrected).",
    "tab:fieldperf")
deep, sh = T12[T12.Set == "D"].Lha, T12[T12.Set == "S"].Lha
KEY["plough"] = {"area_ha": T12.A.sum(), "Lha_range": [T12.Lha.min(), T12.Lha.max()], "Lha_mean": T12.Lha.mean(),
                 "kWhha_range": [T12.kWhha.min(), T12.kWhha.max()], "EFC_range": [T12.EFC.min(), T12.EFC.max()],
                 "MWU_deep_vs_shallow_p": stats.mannwhitneyu(deep, sh).pvalue if len(deep) and len(sh) else None,
                 "spearman_Lha_speed": list(stats.spearmanr(T12.v, T12.Lha))}

# ------------------------------------------------------------------ TABLE S1 — SFC by load bin
wk2 = wk[wk.X > 0.02].assign(sfc=lambda x: x.fuel / (x.X * PP))
bins = np.round(np.arange(0, 1.01, .1), 1)
rows = []
for b_, g in wk2.groupby(pd.cut(wk2.X, bins), observed=True):
    rows.append({"bin": f"({b_.left:.1f}, {b_.right:.1f}]", "n": len(g), "X": g.X.mean(),
                 "sfc": g.sfc.median(), "sfc4974": q_4974(np.array([g.X.mean()]), PP)[0] / (g.X.mean() * PP),
                 "bsfc": 1000 * RHO_F * K_PTO * g.sfc.median()})   # engine (brake/indicated) basis
TS1 = pd.DataFrame(rows)
write_tex("TabS1_SFC_bins", TS1, [
    ("bin", "$X$ bin", "", fS), ("n", "$n$", "", fS), ("X", "$\\bar X$", "[--]", f_(3)),
    ("sfc", "$SFC_v$ measured", "[L\\,kWh$^{-1}$]", f_(3)), ("sfc4974", "$SFC_v$ D497.4", "[L\\,kWh$^{-1}$]", f_(3)),
    ("bsfc", "BSFC measured$^{a}$", "[g\\,kWh$^{-1}$]", f_(1))],
    "Measured specific fuel consumption (median, PTO basis) by load-ratio bin during working passes, compared with ASAE D497.4.",
    "tab:S1", notes="$^{a}$ Engine basis: $1000\\,\\rho_f\\,k_{\\mathrm{PTO}}\\,SFC_v$.")

# ------------------------------------------------------------------ TABLE S2 — implied depth (Appendix)
rows = []
for f, d in D.items():
    if SET[f] == "rotary": continue
    w = (d.phase == "work").to_numpy() & (d.WheelBasedVehicleSpeed.to_numpy() > 1)
    df = pd.DataFrame({"g": np.arange(len(d))[w] // (WIN_S * FS), "P": PE[f][w],
                       "v": d.WheelBasedVehicleSpeed.to_numpy()[w],
                       "dr": d[DRAFT_RIGHT].to_numpy()[w] if DRAFT_RIGHT in d else np.nan}).groupby("g").mean()
    Dn = df.P * K_PTO * K_DB / (df.v / 3.6) * 1000                       # N
    T = Dn / (DRAFT_F * (DRAFT_A + DRAFT_B * df.v + DRAFT_C * df.v ** 2) * W_PLOUGH)
    rows.append({"Field": f, "Set": SET[f][0].upper(), "Dkn": (Dn / 1000).median(), "T": T.median(),
                 "Tiqr": T.quantile(.75) - T.quantile(.25), "dr": df.dr.median()})
TS2 = pd.DataFrame(rows)
write_tex("TabS2_implied_depth", TS2, [
    ("Field", "Field", "", fS), ("Set", "Setting", "", fS), ("Dkn", "Implied draft", "[kN]", f_(2)),
    ("T", "Implied depth", "[cm]", f_(1)), ("Tiqr", "Depth IQR", "[cm]", f_(1)),
    ("dr", "Draft sensor (right)", "[\\%]", f_(0))],
    "Draft force and working depth implied by the D497 mouldboard-plough draft equation, compared with the right lower-link draft sensor (medians of 10\\,s working-pass windows).",
    "tab:S2")
ok = TS2.dropna(subset=["dr"])
if len(ok) > 2:
    KEY["implied_depth_corr"] = {"pearson": list(stats.pearsonr(ok["T"], ok.dr)),
                                 "spearman": list(stats.spearmanr(ok["T"], ok.dr))}

# =============================================================================
#  FIGURES
# =============================================================================
# ---------------------------------------------------------- Fig 2 — segmentation example
f_ex = 8 if 8 in D else max(D, key=lambda k: len(D[k]))
d = D[f_ex].iloc[: 40 * 60 * FS]; t = np.arange(len(d)) / FS / 60
fig, axs = plt.subplots(3, 1, figsize=(W2, 100 * MM), sharex=True, gridspec_kw={"hspace": .30})
sig = [(d.HtcPositionSensPerc, "Hitch position [%]", "k"),
       (d.ActualEngine_PercTorque.clip(lower=0) / 100 * M_R, "Engine torque [Nm]", "#2166ac"),
       (d.EngFuelRate, r"Fuel rate [L h$^{-1}$]", "#b2182b")]
for k, (ax, (y, yl, c)) in enumerate(zip(axs, sig)):
    for v, a, b in runs((d.phase == "work").to_numpy()):
        if v: ax.axvspan(t[a], t[min(b, len(t) - 1)], color="#d9f0d3", lw=0, zorder=0)
    ax.plot(t, y, c=c, lw=.5, zorder=2); ax.set_ylabel(yl); plabel(ax, f"({'abc'[k]})")
axs[0].axhline(D[f_ex].attrs["thr"], c="k", ls="--", lw=.7, zorder=3)
axs[-1].set_xlabel("Time [min]"); axs[-1].set_xlim(0, t[-1])
fig.legend([Patch(color="#d9f0d3"), Line2D([], [], c="k", ls="--", lw=.7)],
           ["Working pass", f"Otsu threshold ({D[f_ex].attrs['thr']:.1f}%)"],
           loc="upper center", ncol=2, bbox_to_anchor=(.5, .97))
save(fig, "Fig02_segmentation_example")

# ---------------------------------------------------------- Fig 3 — operating-point map
S = {k: pd.concat([pd.DataFrame({"n": D[f].EngineSpeed[D[f].phase == k].to_numpy(),
                                 "P": PE[f][(D[f].phase == k).to_numpy()]}) for f in D]).dropna()
     for k in ("work", "head")}
xb, yb = np.arange(800, 2451, 50), np.arange(-2.5, 92.5, 2.5)
H = {}
for k, s in S.items():
    h, _, _ = np.histogram2d(s.n, s.P, bins=[xb, yb]); H[k] = np.ma.masked_equal(h / h.sum(), 0)
vmin, vmax = min(H[k].min() for k in H), max(H[k].max() for k in H)
band = (T4["rpm"].min(), T4["rpm"].max())
cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("white")
fig, axs = plt.subplots(1, 2, figsize=(W2, 78 * MM), sharey=True, gridspec_kw={"wspace": .06})
for ax, k, lab in ((axs[0], "work", "(a) Working passes"), (axs[1], "head", "(b) Headland turns")):
    m = ax.pcolormesh(xb, yb, H[k].T, norm=LogNorm(vmin, vmax), cmap=cmap, shading="flat", rasterized=True)
    for X in (.25, .5, .75, 1.0):
        ax.axhline(X * P_N, c="0.4", ls=":", lw=.6); ax.text(815, X * P_N + .8, f"$X$ = {X:g}", fontsize=6.5, color="0.3")
    ax.axvspan(*band, color="k", alpha=.08, lw=0)
    ax.plot(N_RT, P_N, marker="*", ms=10, mfc="white", mec="k", ls="")
    ax.set(xlabel=r"Engine speed $n_e$ [rpm]", xlim=(800, 2450)); plabel(ax, lab)
s = S["work"]; env = s.groupby(pd.cut(s.n, xb), observed=True).P.agg(["size", lambda x: x.quantile(.99)])
env = env[env["size"] >= 50]
axs[0].plot([i.mid for i in env.index], env.iloc[:, 1], "k-", lw=1)
axs[0].set_ylabel(r"Engine power $P_e$ [kW]"); axs[0].set_ylim(-2.5, 92.5)
cb = fig.colorbar(m, ax=axs, pad=.01, fraction=.03); cb.set_label("Probability [–]")
fig.legend([Line2D([], [], c="k"), Patch(color="k", alpha=.08), Line2D([], [], marker="*", ms=9, mfc="white", mec="k", ls="")],
           ["Observed envelope, working passes (99th percentile)", "Range of session-mean working speed", "Rated point"],
           loc="lower center", ncol=3, bbox_to_anchor=(.45, -.08))
save(fig, "Fig03_operating_map")
hh = S["head"]
KEY["opmap"] = {"work_time_below_90pct_rated_%": 100 * (S["work"].n < .9 * N_RT).mean(),
                "P99_envelope_max_kW": float(env.iloc[:, 1].max()),
                "P99_envelope_max_%rated": 100 * float(env.iloc[:, 1].max()) / P_N,
                "head_time_ge2000rpm_%": 100 * (hh.n >= 2000).mean(),
                "head_time_ge2000rpm_10to40kW_%": 100 * ((hh.n >= 2000) & hh.P.between(10, 40)).mean(),
                "head_time_below1000rpm_%": 100 * (hh.n < 1000).mean(),
                "head_median_rpm": hh.n.median(), "head_median_P": hh.P.median()}

# ---------------------------------------------------------- Fig 4 — predicted vs measured
fig, axs = plt.subplots(1, 4, figsize=(W2, 54 * MM), sharex=True, sharey=True, gridspec_kw={"wspace": .08})
lim = (0, np.ceil(max(W.fuel.quantile(.999), 1) / 5) * 5 + 5)
hbs = []
for k, (ax, m) in enumerate(zip(axs, MODELS)):
    hb = ax.hexbin(W.fuel, W[m], gridsize=45, extent=lim + lim, bins="log", mincnt=1, cmap="viridis",
                   linewidths=0, rasterized=True); hbs.append(hb)
    xx = np.array(lim)
    ax.plot(xx, xx, "k-", lw=.7); ax.plot(xx, 1.15 * xx, "k:", lw=.6); ax.plot(xx, .85 * xx, "k:", lw=.6)
    mm = metrics(W.fuel, W[m])
    ax.text(.04, .96, f"RMSE = {mm['RMSE']:.2f} L h$^{{-1}}$\n" + um(f"MBE = {mm['MBE']:+.1f}%\n") + "$R^2$ = " + um(f"{mm['R2']:.2f}"),
            transform=ax.transAxes, va="top", fontsize=6.5)
    ax.set(xlim=lim, ylim=lim, aspect="equal"); ax.set_xlabel(r"Measured [L h$^{-1}$]")
    plabel(ax, f"({'abcd'[k]}) {MODELS[m]}")
axs[0].set_ylabel(r"Predicted [L h$^{-1}$]")
vmax = max(h.get_array().max() for h in hbs)
for h in hbs: h.set_norm(LogNorm(1, vmax))
cb = fig.colorbar(hbs[0], ax=axs, pad=.01, fraction=.02); cb.set_label("Windows [–]")
save(fig, "Fig04_pred_vs_meas")

# ---------------------------------------------------------- Fig 5 — Bland–Altman (PT, work)
mean_ = (wk.fuel + wk.PT) / 2; diff = wk.PT - wk.fuel
bias, sd = diff.mean(), diff.std(ddof=1); lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
sl, ic = np.polyfit(mean_, diff, 1)
fig, ax = plt.subplots(figsize=(W15, 70 * MM))
ax.hexbin(mean_, diff, gridsize=60, bins="log", mincnt=1, cmap="Greys", linewidths=0, rasterized=True)
ax.axhline(bias, c="#b2182b", lw=1); ax.axhline(lo, c="#b2182b", ls="--", lw=.8); ax.axhline(hi, c="#b2182b", ls="--", lw=.8)
xs = np.linspace(mean_.min(), mean_.max(), 10); ax.plot(xs, ic + sl * xs, c="#2166ac", lw=1)
ax.axhline(0, c="k", lw=.5)
ax.set(xlabel=r"Mean of measured and predicted fuel rate [L h$^{-1}$]",
       ylabel=r"Predicted $-$ measured [L h$^{-1}$]")
ax.legend([Line2D([], [], c="#b2182b"), Line2D([], [], c="#b2182b", ls="--"), Line2D([], [], c="#2166ac")],
          [um(f"Mean bias ({bias:+.2f} L h$^{{-1}}$)"), um(f"95% limits of agreement ({lo:.2f}, {hi:+.2f})"),
           um(f"Trend (slope {sl:.3f})")], loc="lower left")
save(fig, "Fig06_bland_altman_PT")
KEY["bland_altman_PT"] = {"bias": bias, "LoA": [lo, hi], "trend_slope": sl, "trend_intercept": ic,
                          "zero_cross_Lh": -ic / sl if sl else None}

# ---------------------------------------------------------- Fig 6 — bias vs speed ratio / load
F = np.array(FIELDS)
CNT = rng.multinomial(len(F), np.ones(len(F)) / len(F), size=N_BOOT)
def binned(df, by, edges, col, min_n=30):
    df = df.assign(b=pd.cut(df[by], edges)); out = []
    for b_, g in df.groupby("b", observed=True):
        if len(g) < min_n: continue
        num = (g[col] - g.fuel).groupby(g.field).sum().reindex(F, fill_value=0).to_numpy()
        den = g.fuel.groupby(g.field).sum().reindex(F, fill_value=0).to_numpy()
        dd = CNT @ den; bs = 100 * (CNT @ num) / np.where(dd > 0, dd, np.nan)
        out.append({"mid": float(b_.mid), "MBE": 100 * num.sum() / den.sum(),
                    "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5), "n": len(g)})
    return pd.DataFrame(out)
wk = wk.assign(sr=np.clip(wk.rpm / N_RT, .5, 1.0))
sr_e = np.round(np.arange(.66, 1.0001, .04), 2); x_e = np.round(np.arange(0, 1.2001, .1), 2)
Bs = {m: binned(wk, "sr", sr_e, m) for m in ("FT", "PT")}
Bx = {m: binned(W, "X", x_e, m) for m in ("4974", "FT", "PT")}
fig, axs = plt.subplots(1, 2, figsize=(W2, 72 * MM), gridspec_kw={"wspace": .45})
ax = axs[0]; ax2 = ax.twinx()
cnt = wk.groupby(pd.cut(wk.sr, sr_e), observed=True).size()
ax2.bar([i.mid for i in cnt.index], 100 * cnt / cnt.sum(), width=.034, color="0.85", lw=0)
ax2.set_ylabel("Share of working-pass windows [%]")
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
for m, st in (("FT", "-o"), ("PT", "--s")):
    b_ = Bs[m]; ax.plot(b_.mid, b_.MBE, st, c=MCOL[m], ms=3, label=MODELS[m])
    ax.fill_between(b_.mid, b_.lo, b_.hi, color=MCOL[m], alpha=.2, lw=0)
ax.axhspan(-10, 10, color="0.5", alpha=.07, lw=0); ax.axhline(0, c="k", lw=.5)
ax.set(xlim=(.66, 1.0), xlabel=r"Engine-speed ratio $N_{\mathrm{PT}}/N_{\mathrm{RT}}$ [–]", ylabel="Mean bias error [%]")
ax.legend(loc="upper right"); plabel(ax, "(a) Working passes")
ax = axs[1]
for m, st in (("4974", ":^"), ("FT", "-o"), ("PT", "--s")):
    b_ = Bx[m]; ax.plot(b_.mid, 1 + b_.MBE / 100, st, c=MCOL[m], ms=3, label=MODELS[m])
    ax.fill_between(b_.mid, 1 + b_.lo / 100, 1 + b_.hi / 100, color=MCOL[m], alpha=.15, lw=0)
ax.set_yscale("log"); tk = [.8, .9, 1, 1.25, 1.5, 2, 3]
ax.set_yticks(tk); ax.set_yticklabels([f"{v:g}" for v in tk]); ax.minorticks_off()
ax.axhspan(.9, 1.1, color="0.5", alpha=.07, lw=0); ax.axhline(1, c="k", lw=.5)
ax.set(xlabel=r"Engine load ratio $X$ [–]", ylabel="Predicted / measured fuel [–]")
ax.legend(loc="upper right"); plabel(ax, "(b) All windows")
save(fig, "Fig05_bias_mechanism")
def crossing(b_):
    y, x = b_.MBE.to_numpy(), b_.mid.to_numpy(); k = np.flatnonzero((y[:-1] > 0) & (y[1:] <= 0))
    return None if not len(k) else float(x[k[0]] + (x[k[0] + 1] - x[k[0]]) * y[k[0]] / (y[k[0]] - y[k[0] + 1]))
KEY["bias_mechanism"] = {"FT_by_sr": Bs["FT"].round(2).to_dict("records"), "PT_by_sr": Bs["PT"].round(2).to_dict("records"),
                         "PT_by_X": Bx["PT"].round(2).to_dict("records"), "FT_by_X": Bx["FT"].round(2).to_dict("records"),
                         "D4974_by_X": Bx["4974"].round(2).to_dict("records"),
                         "PT_zero_cross_X": crossing(Bx["PT"]), "FT_zero_cross_X": crossing(Bx["FT"])}
pd.concat([Bs[m].assign(model=m, by="speed_ratio") for m in Bs] + [Bx[m].assign(model=m, by="X") for m in Bx]).to_csv(
    os.path.join(TAB, "Fig05_binned_bias.csv"), index=False)

# ---------------------------------------------------------- Fig 7 — specific fuel consumption
xs = np.linspace(.08, 1.1, 200)
fig, ax = plt.subplots(figsize=(W15, 72 * MM))
ax.scatter(wk2.X, wk2.sfc, s=1, c="0.75", alpha=.25, lw=0, rasterized=True)
bb = wk2.groupby(pd.cut(wk2.X, bins), observed=True).sfc.agg(["median", lambda v: v.quantile(.25), lambda v: v.quantile(.75), "size"])
bb = bb[bb["size"] >= 20]; mids = [i.mid for i in bb.index]
ax.errorbar(mids, bb["median"], yerr=[bb["median"] - bb.iloc[:, 1], bb.iloc[:, 2] - bb["median"]],
            fmt="o", c="k", ms=3.5, capsize=2, lw=.8, label="Measured (median, IQR)")
ax.plot(xs, q_4974(xs, PP) / (xs * PP), c=MCOL["4974"], ls=":", label="ASAE D497.4")
ax.plot(xs, (0.22 * xs + 0.096) / xs, c=MCOL["FT"], label="D497.7 full throttle")
ax.plot(xs, (a_hat * xs + b_hat) / xs, c=MCOL["RC"], ls="--", label="Recalibrated (this study)")
ax.set(xlim=(0, 1.1), ylim=(0, 1.2), xlabel=r"Engine load ratio $X$ [–]",
       ylabel=r"Specific fuel consumption $SFC_v$ [L kWh$^{-1}$, PTO basis]")
sec = ax.secondary_yaxis("right", functions=(lambda v: v * 1000 * RHO_F * K_PTO, lambda v: v / (1000 * RHO_F * K_PTO)))
sec.set_ylabel(r"BSFC, engine basis [g kWh$^{-1}$]")
ax.legend(loc="upper right")
save(fig, "Fig07_SFC_vs_load")
KEY["sfc"] = {"plateau_X>0.6_median": wk2[wk2.X > .6].sfc.median(),
              "share_windows_X>0.6_%": 100 * (wk.X > .6).mean(), "share_windows_X>0.7_%": 100 * (wk.X > .7).mean()}

# ---------------------------------------------------------- Fig S1 — load-ratio violins
fig, ax = plt.subplots(figsize=(W2, 65 * MM))
data_v = [np.clip(XX[f][(D[f].phase == "work").to_numpy()], 0, 1.3) for f in FIELDS]
vp = ax.violinplot(data_v, showmedians=True, showextrema=False, widths=.8)
for body, f in zip(vp["bodies"], FIELDS):
    body.set_facecolor(SCOL[SET[f]]); body.set_edgecolor("k"); body.set_linewidth(.4); body.set_alpha(.7)
vp["cmedians"].set_color("k"); vp["cmedians"].set_linewidth(.8)
ax.set_xticks(range(1, len(FIELDS) + 1)); ax.set_xticklabels([f"F{f}" for f in FIELDS])
ax.set(ylabel=r"Engine load ratio $X$ [–]", ylim=(0, 1.3))
ax.legend([Patch(color=SCOL[k], alpha=.7) for k in SCOL], ["Deep plough", "Shallow plough", "Rotary tiller"],
          loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3)
save(fig, "FigS1_load_ratio_violins")

# =============================================================================
#  EXPORT
# =============================================================================
with pd.ExcelWriter(os.path.join(OUT, "BE_tables.xlsx")) as xw:
    for k, v in TABLES.items(): v.round(4).to_excel(xw, sheet_name=k[:31], index=False)
W.round(4).to_csv(os.path.join(TAB, "S_windows_10s.csv"), index=False)
def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer): return int(o)
    return o
with open(os.path.join(OUT, "key_numbers.json"), "w") as fh: json.dump(_clean(KEY), fh, indent=1)

pd.set_option("display.width", 200)
print("=" * 78, "\nCHECK vs manuscript (windows 6931/1171; work MBE 37.9/13.5/1.9; head 80.1/64.6/35.6)")
print(f"windows work/head: {len(wk)}/{len(hd)}")
print(T5[["Phase", "n", "MBE", "RMSE", "R2", "w10"]].assign(Model=[MODELS[m] for m in MODELS for _ in range(3)]).round(2).to_string(index=False))
print("\nSensitivity (headland MBE under friction is the key check):\n",
      T11[["Case", "MBE_FT", "MBE_PT", "H_FT", "H_PT", "a", "b"]].round(3).to_string(index=False))
print(f"\nRecalibrated: a={a_hat:.4f}, b={b_hat:.4f}  (intercept {b_hat*PP:.2f} L/h); "
      f"measured fuel at X<0.05 (headland) = {KEY['measured_near_idle_fuel_Lh']['median']:.2f} L/h")
print(f"\nSaved figures -> {FIG}\nSaved tables  -> {TAB}  (+ BE_tables.xlsx, key_numbers.json)")