from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

params_1rc = pd.read_csv(BASE_DIR / "params_1rc.csv").sort_values("soc")
params_2rc = pd.read_csv(BASE_DIR / "params_2rc.csv").sort_values("soc")

required_1rc = {"soc", "R0", "RTh", "CTh", "tau1"}
required_2rc = {"soc", "R0", "R1", "C1", "tau1", "R2", "C2", "tau2"}
missing_1rc = required_1rc.difference(params_1rc.columns)
missing_2rc = required_2rc.difference(params_2rc.columns)
if missing_1rc or missing_2rc:
    raise ValueError(
        f"Parameter CSV columns missing: 1-RC={sorted(missing_1rc)}, "
        f"2-RC={sorted(missing_2rc)}"
    )

plots_1rc = [
    ("R0", "Resistance (mOhm)", 1000),
    ("RTh", "Resistance (mOhm)", 1000),
    ("CTh", "Capacitance (F)", 1),
    ("tau1", "Time constant (s)", 1),
]
fig_1rc, axs_1rc = plt.subplots(2, 2, figsize=(12, 8))
for ax, (column, ylabel, scale) in zip(axs_1rc.flat, plots_1rc):
    ax.plot(
        params_1rc["soc"] * 100,
        params_1rc[column] * scale,
        "o-",
        color="tab:blue",
    )
    ax.set_title(column)
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.5)

fig_1rc.suptitle("1-RC Fitted Parameters vs. SOC", fontsize=16)
fig_1rc.tight_layout(rect=(0, 0, 1, 0.96))

plots_2rc = [
    ("R0", "Resistance (mOhm)", 1000),
    ("R1", "Resistance (mOhm)", 1000),
    ("C1", "Capacitance (F)", 1),
    ("tau1", "Time constant (s)", 1),
    ("R2", "Resistance (mOhm)", 1000),
    ("C2", "Capacitance (F)", 1),
    ("tau2", "Time constant (s)", 1),
]
fig_2rc, axs_2rc = plt.subplots(4, 2, figsize=(12, 14))
for ax, (column, ylabel, scale) in zip(axs_2rc.flat, plots_2rc):
    ax.plot(
        params_2rc["soc"] * 100,
        params_2rc[column] * scale,
        "s--",
        color="tab:orange",
    )
    ax.set_title(column)
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.5)

for ax in axs_2rc.flat[len(plots_2rc):]:
    ax.set_visible(False)

fig_2rc.suptitle("2-RC Fitted Parameters vs. SOC", fontsize=16)
fig_2rc.tight_layout(rect=(0, 0, 1, 0.97))

output_1rc = BASE_DIR / "fitted_parameters_1rc_vs_soc.png"
output_2rc = BASE_DIR / "fitted_parameters_2rc_vs_soc.png"
fig_1rc.savefig(output_1rc, dpi=300)
fig_2rc.savefig(output_2rc, dpi=300)
print(f"Saved 1-RC parameter plots to: {output_1rc}")
print(f"Saved 2-RC parameter plots to: {output_2rc}")
plt.show()
