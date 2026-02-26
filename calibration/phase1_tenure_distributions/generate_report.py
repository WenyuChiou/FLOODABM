"""
Generate Phase 1 Report: Owner/Renter Data Extraction & Distribution Fitting
Output: Word document with tables, embedded plots, and analysis.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import pandas as pd

OUT_DIR = Path(r"C:\Users\wenyu\OneDrive - Lehigh University\Desktop\Lehigh\NSF-project\ABM\Calibration\tenure_distributions")

doc = Document()

# ── Styles ──
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15

def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)
    return h

def add_table_from_df(df, col_widths=None):
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Light Shading Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header
    for j, col in enumerate(df.columns):
        cell = table.rows[0].cells[j]
        cell.text = str(col)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(9)
    # Data
    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for j, val in enumerate(row):
            row_cells[j].text = str(val)
            for p in row_cells[j].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9)
    return table

# ══════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════
title = doc.add_heading("Phase 1 Report: Owner/Renter Data Extraction\n& Distribution Fitting", level=0)
for run in title.runs:
    run.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)

doc.add_paragraph("Date: 2026-02-26 | ABM Restructuring: MG/NMG → Homeowner/Renter")
doc.add_paragraph("")

# ══════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════════════
add_heading("1. Overview", level=1)
doc.add_paragraph(
    "This report documents the restructuring of the ABM's primary grouping axis "
    "from MG (mitigation grant) vs NMG (non-mitigation grant) to homeowner vs renter "
    "(housing tenure). The classification rule is: Q3 = 2 → renter; all other values → owner."
)

add_heading("1.1 Data Sources", level=2)
doc.add_paragraph(
    "• Raw survey data: data_ori.xlsx (NMG sheet: 789 rows; MG sheet: 148 rows; total: 937)\n"
    "• TP decay calibration data: cal_data.xlsx (NMG_FE: 173 rows; MG_FE: 40 rows; total: 213)\n"
    "• Q3 encoding: 1 = homeowner, 2 = renter"
)

add_heading("1.2 Variable Formulas (verified from Excel)", level=2)
formula_data = pd.DataFrame({
    "Variable": ["TP_mean", "CP_mean", "SP_mean", "SC_mean", "PA_mean",
                  "FI", "EH", "BP", "RL"],
    "Formula": [
        "mean(Q22_1 : Q22_11) — 11 items",
        "mean(Q24_1, Q24_2, Q25_1, Q25_2, Q25_4, Q25_5, Q25_7, Q25_8) — 8 items",
        "mean(Q25_3, Q25_6, Q25_9) — 3 items",
        "mean(Q21_1 : Q21_6) — 6 items",
        "mean(Q21_7 : Q21_15) — 9 items",
        "Q27", "Q29", "Q31", "Q33"
    ],
    "Scale": ["1–5 Likert", "1–5 Likert", "1–5 Likert", "1–5 Likert", "1–5 Likert",
              "1–5 Likert", "1–5 Likert", "1–5 Likert", "1–5 Likert"]
})
add_table_from_df(formula_data)
doc.add_paragraph("")

add_heading("1.3 Important Bug Discovery", level=2)
p = doc.add_paragraph()
run = p.add_run("MG_variable sheet has broken Excel formulas: ")
run.bold = True
run.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
p.add_run(
    "The TP_mean, CP_mean, and SP_mean columns in MG_variable reference NMG sheet cells "
    "(e.g., =AVERAGE(NMG!BH2:BR2)) instead of MG sheet cells. This means all 148 MG "
    "respondents' psychological variable data was actually NMG data at the same row index. "
    "MG_syn_variable (1,989 rows used by Bayesian engine) inherits this error. "
    "The new owner/renter extraction recomputes all variables from raw data, bypassing this bug."
)

# ══════════════════════════════════════════════════════════════════
# 2. SAMPLE SIZES
# ══════════════════════════════════════════════════════════════════
add_heading("2. Sample Sizes After Split", level=1)

size_data = pd.DataFrame({
    "Dataset": ["Full Survey", "Full Survey", "TP Decay Calibration", "TP Decay Calibration"],
    "Group": ["Owner", "Renter", "Owner", "Renter"],
    "n": [694, 243, 170, 43],
    "Source MG": ["77", "71", "28", "12"],
    "Source NMG": ["617", "172", "142", "31"],
})
add_table_from_df(size_data)
doc.add_paragraph("")
doc.add_paragraph(
    "Note: Renter calibration sample (n=43) is small. "
    "This may require discussion on whether synthetic augmentation or pooled estimation is needed."
)

# ══════════════════════════════════════════════════════════════════
# 3. BETA DISTRIBUTION FITS
# ══════════════════════════════════════════════════════════════════
add_heading("3. Beta Distribution Parameters", level=1)
doc.add_paragraph(
    "Each psychological variable was normalized to [0, 1] by dividing by 5 (Likert 1–5 → 0.2–1.0), "
    "then fitted with a Beta(α, β) distribution using MLE (scipy.stats.beta.fit with floc=0, fscale=1). "
    "The KS test checks goodness-of-fit (p > 0.05 = good fit)."
)

beta_df = pd.read_csv(OUT_DIR / "beta_parameters_summary.csv")
add_table_from_df(beta_df)
doc.add_paragraph("")

add_heading("3.1 Key Observations", level=2)
doc.add_paragraph(
    "• TP (Threat Perception): Owner has slightly tighter distribution (α=6.12 vs 4.77). "
    "Renter TP shows excellent KS fit (p=0.667); owner TP marginal (p=0.037).\n"
    "• CP (Coping Perception): Very similar between groups (mean ~0.62).\n"
    "• SP (Social Perception): Renter has slightly higher mean (0.592 vs 0.570) and wider spread.\n"
    "• SC (Social Capital): Owner has higher mean (0.745 vs 0.717). Both distributions are "
    "strongly right-skewed (β < 2).\n"
    "• PA (Place Attachment): Nearly identical means (~0.659). Owner has tighter distribution.\n"
    "• KS test: Only TP passes for both groups. Other variables show significant deviation — "
    "likely due to discrete Likert data creating non-smooth distributions. "
    "The Beta approximation is standard practice for ABM initialization despite this."
)

add_heading("3.2 Comparison with Old MG/NMG Parameters", level=2)
old_params = pd.DataFrame({
    "Variable": ["TP", "CP", "SP", "SC", "PA"],
    "MG α": [4.441, 4.078, 1.367, 5.369, 2.555],
    "MG β": [2.887, 3.330, 1.694, 3.105, 2.165],
    "MG mean": [0.606, 0.550, 0.447, 0.634, 0.541],
    "NMG α": [5.346, 5.263, 1.725, 4.563, 4.007],
    "NMG β": [3.615, 4.178, 1.933, 2.389, 2.791],
    "NMG mean": [0.597, 0.557, 0.472, 0.656, 0.589],
    "Owner α": [6.123, 5.430, 2.667, 3.368, 3.575],
    "Owner β": [4.480, 3.333, 2.014, 1.155, 1.850],
    "Owner mean": [0.578, 0.620, 0.570, 0.745, 0.659],
    "Renter α": [4.765, 5.236, 2.142, 3.996, 2.762],
    "Renter β": [3.364, 3.155, 1.474, 1.576, 1.428],
    "Renter mean": [0.586, 0.624, 0.592, 0.717, 0.659],
})
add_table_from_df(old_params)
doc.add_paragraph("")
doc.add_paragraph(
    "Key differences: Owner/Renter CP and SP means are notably higher than old MG/NMG means "
    "(~0.62 vs ~0.55 for CP; ~0.58 vs ~0.46 for SP). This is partly because the old MG values "
    "were computed from incorrect data (NMG formulas). SC shows much higher values in the new "
    "split (0.72–0.74 vs 0.63–0.66), likely reflecting the correct computation from raw data."
)

# ══════════════════════════════════════════════════════════════════
# 4. ACTION ADOPTION RATES
# ══════════════════════════════════════════════════════════════════
add_heading("4. Action Adoption Rates", level=1)
action_df = pd.read_csv(OUT_DIR / "action_adoption_rates.csv")
add_table_from_df(action_df)
doc.add_paragraph("")

add_heading("4.1 Key Observations", level=2)
doc.add_paragraph(
    "• FI (Flood Insurance): 100% response rate for both groups. Renters have slightly higher "
    "mean intention (3.01 vs 2.92).\n"
    "• EH (Elevation/Hardening): Owners respond more (62.2% vs 51.4%), but renters who do "
    "respond show slightly higher intensity (2.75 vs 2.59). This makes sense — owners have "
    "more agency over structural modifications.\n"
    "• BP (Building Protection): Same pattern as EH — owners respond more but renters show "
    "slightly higher intensity.\n"
    "• RL (Relocation): Renters respond significantly more (46.1% vs 34.9%) with similar "
    "mean intensity (~3.13). This aligns with theory — renters have lower relocation barriers.\n\n"
    "Discussion point: In the original model, EH and BP were owner-only actions, and RL was "
    "available to all but emphasized for MG. In the new owner/renter framework, we should "
    "discuss whether EH/BP should still be available to renters (who may influence landlord "
    "decisions) or restricted to owners only."
)

# ══════════════════════════════════════════════════════════════════
# 5. PLOTS
# ══════════════════════════════════════════════════════════════════
add_heading("5. Distribution Plots", level=1)

add_heading("5.1 Overlay Comparison (Owner vs Renter)", level=2)
doc.add_picture(str(OUT_DIR / "overlay_comparison.png"), width=Inches(6.5))
last_paragraph = doc.paragraphs[-1]
last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

add_heading("5.2 Beta Fits with Histograms", level=2)
doc.add_picture(str(OUT_DIR / "beta_distributions_owner_renter.png"), width=Inches(6.0))
last_paragraph = doc.paragraphs[-1]
last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

add_heading("5.3 Action Response Rates", level=2)
doc.add_picture(str(OUT_DIR / "action_rates_owner_renter.png"), width=Inches(5.5))
last_paragraph = doc.paragraphs[-1]
last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

add_heading("5.4 Action Mean Likert Scores", level=2)
doc.add_picture(str(OUT_DIR / "action_means_owner_renter.png"), width=Inches(5.5))
last_paragraph = doc.paragraphs[-1]
last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ══════════════════════════════════════════════════════════════════
# 6. TP DECAY CALIBRATION DATA SUMMARY
# ══════════════════════════════════════════════════════════════════
add_heading("6. TP Decay Calibration Data", level=1)
doc.add_paragraph(
    "The calibration dataset contains only respondents with flood experience (Q15 not NaN). "
    "Variables include Q15 (flood recency ordinal), Q15_year (converted to approximate years), "
    "and the 5 psychological means."
)

cal_summary = pd.DataFrame({
    "Statistic": ["Count", "Q15 mean", "Q15_year mean", "SC mean", "PA mean", "TP mean", "CP mean", "SP mean"],
    "Owner": ["170", "2.27", "5.05", "3.67", "3.35", "3.34", "3.19", "2.83"],
    "Renter": ["43", "2.63", "7.19", "3.36", "3.25", "3.60", "3.38", "3.02"],
})
add_table_from_df(cal_summary)
doc.add_paragraph("")
doc.add_paragraph(
    "Renters have slightly longer average time since last flood (7.2 vs 5.1 years) and "
    "higher TP mean (3.60 vs 3.34). The renter sample (n=43) is small but usable for "
    "grid-search calibration. Bootstrap confidence intervals will be wider."
)

# ══════════════════════════════════════════════════════════════════
# 7. READY-TO-USE PARAMETERS
# ══════════════════════════════════════════════════════════════════
add_heading("7. Ready-to-Use PerceptionSampler Parameters", level=1)
doc.add_paragraph(
    "These Beta parameters can directly replace the hardcoded values in "
    "PerceptionSampler.py (line 20–35):"
)

code = doc.add_paragraph()
code.style = doc.styles["Normal"]
code_text = """params = {
    'owner': {
        'TP': {'alpha': 6.122639800, 'beta': 4.480251499},
        'CP': {'alpha': 5.430291126, 'beta': 3.333321897},
        'SP': {'alpha': 2.666477827, 'beta': 2.014197185},
        'SC': {'alpha': 3.367934467, 'beta': 1.154550838},
        'PA': {'alpha': 3.574674584, 'beta': 1.850491186},
    },
    'renter': {
        'TP': {'alpha': 4.765219193, 'beta': 3.364159852},
        'CP': {'alpha': 5.236202560, 'beta': 3.154797843},
        'SP': {'alpha': 2.142182744, 'beta': 1.474363312},
        'SC': {'alpha': 3.996213540, 'beta': 1.576348572},
        'PA': {'alpha': 2.761719297, 'beta': 1.428060094},
    }
}"""
run = code.add_run(code_text)
run.font.name = "Consolas"
run.font.size = Pt(9)

# ══════════════════════════════════════════════════════════════════
# 8. NEXT STEPS
# ══════════════════════════════════════════════════════════════════
add_heading("8. Next Steps", level=1)
doc.add_paragraph(
    "1. Professor reviews this report and approves distributions.\n"
    "2. Phase 2: Update Bayesian regression config → run with owner/renter data → "
    "produce new .pkl model files.\n"
    "3. Phase 3: Update TP decay calibration → run grid search → new α, β, τ₀, d, k parameters.\n"
    "4. Phase 4: Propagate owner/renter labels through simulation code "
    "(bayes_fast_predictors.py, mgmix_tp.py, mgmix_decision.py, mgmix_pipeline.py)."
)

add_heading("8.1 Discussion Points for Professor", level=2)
doc.add_paragraph(
    "• Renter calibration sample (n=43): sufficient or needs augmentation?\n"
    "• EH/BP availability: restrict to owners only, or allow renters?\n"
    "• Original plan: RL = renter-only. Current data shows owners also respond to RL (34.9%). "
    "Keep RL for both groups or restrict?\n"
    "• MG_variable bug: does this affect any previously published results?\n"
    "• KS test failures: acceptable for ABM initialization? (standard practice)"
)

# ── Save ──
out_path = OUT_DIR / "Phase1_Owner_Renter_Report.docx"
doc.save(str(out_path))
print(f"Report saved to: {out_path}")
