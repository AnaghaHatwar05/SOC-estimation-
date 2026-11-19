import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

BASE_DIR = Path(__file__).resolve().parent.parent
data = pd.read_csv(BASE_DIR / "hppc_data.csv")


# 2. Extract global OCV-SOC lookup table
# Option A: Take the voltage at the end of the initial 2-hour rest (t == 7200s)
# This avoids averaging transient polarization from the 40s post-pulse rests
rest_mask = (np.abs(data["current"]) < 0.01) & (data["time"] <= 7200)
ocv_summary = data[rest_mask].groupby("soc_point")["voltage"].last().reset_index()
ocv_summary.columns = ["soc", "ocv"]
ocv_summary = ocv_summary.sort_values("soc").reset_index(drop=True)

# Export the extracted OCV table for Chen2020
ocv_summary.to_csv(BASE_DIR / "ocv_soc_table.csv", index=False)
print(f"Saved extracted OCV table to: {BASE_DIR / 'ocv_soc_table.csv'}")
print(ocv_summary.to_string(index=False))

ocv_interp = interp1d(ocv_summary["soc"], ocv_summary["ocv"], kind="cubic", fill_value="extrapolate")
# 3. Define target SOC points for detailed pulse comparison
target_socs = [0.1, 0.3, 0.6, 0.9]
colors = {0.1: "tab:red", 0.3: "tab:orange", 0.6: "tab:green", 0.9: "tab:blue"}

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# ---------------------------------------------------------
# Plot 1: Terminal Voltage vs. Relative Time (Per SOC Pulse)
# ---------------------------------------------------------
ax1 = axs[0, 0]
for soc_val in target_socs:
    # Find closest matching soc_point in data
    soc_mask = np.isclose(data["soc_point"], soc_val, atol=0.05)
    df_soc = data[soc_mask].copy()
    
    if not df_soc.empty:
        pulse_data = df_soc[df_soc["time"] >= 7000]
        ax1.plot(
            pulse_data["time"],
            pulse_data["voltage"],
            label=f"SOC {int(soc_val*100)}%",
            color=colors[soc_val],
            lw=1.8,
        )

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Terminal Voltage (V)")
ax1.set_title("1. HPPC Pulse Voltage Response by SOC")
ax1.grid(True)
ax1.legend(loc="upper right")

# ---------------------------------------------------------
# Plot 2: Terminal Voltage vs. Current (V–I Slopes per SOC)
# ---------------------------------------------------------
ax2 = axs[0, 1]
for soc_val in target_socs:
    soc_mask = np.isclose(data["soc_point"], soc_val, atol=0.05)
    df_soc = data[soc_mask]
    
    if not df_soc.empty:
        # V-I line slope = Internal Resistance (R = dV/dI) at this SOC
        ax2.scatter(df_soc["current"], df_soc["voltage"], color=colors[soc_val], 
                    label=f"SOC {int(soc_val*100)}%", s=15, alpha=0.7)

ax2.set_xlabel("Applied Current (A)")
ax2.set_ylabel("Terminal Voltage (V)")
ax2.set_title("2. V–I Characteristic & Resistance Slopes per SOC")
ax2.grid(True)
ax2.legend(loc="best")

# ---------------------------------------------------------
# Plot 3: OCV Relaxation vs. Time Post-Discharge (Per SOC)
# ---------------------------------------------------------
ax3 = axs[1, 0]
for soc_val in target_socs:
    soc_mask = np.isclose(data["soc_point"], soc_val, atol=0.05)
    df_soc = data[soc_mask & rest_mask].copy()
    
    if not df_soc.empty:
        t_rest = df_soc["time"] - df_soc["time"].iloc[0] if "time" in df_soc.columns else df_soc.index - df_soc.index[0]
        ax3.plot(t_rest, df_soc["voltage"], color=colors[soc_val], label=f"SOC {int(soc_val*100)}%", lw=1.8)

ax3.set_xlabel("Relaxation Time (s)")
ax3.set_ylabel("Rest Voltage / OCV (V)")
ax3.set_title("3. Voltage Relaxation Dynamics per SOC")
ax3.grid(True)
ax3.legend(loc="lower right")

# ---------------------------------------------------------
# Plot 4: Global OCV vs. SOC Curve with Highlighted Points
# ---------------------------------------------------------
# Extract ground-truth OCV per soc_point for verification
ocv_gt = data.groupby("soc_point")["ocv_true"].first().reset_index().sort_values("soc_point")

ax4 = axs[1, 1]
soc_dense = np.linspace(ocv_summary["soc"].min(), ocv_summary["soc"].max(), 200)

# 1. Plot Ground-Truth OCV from PyBaMM physics
ax4.plot(
    ocv_gt["soc_point"] * 100,
    ocv_gt["ocv_true"],
    "b-",
    lw=2.5,
    alpha=0.6,
    label="Ground Truth OCV (Physics)",
)

# 2. Plot Interpolated Rest-Extracted OCV Curve
ax4.plot(
    soc_dense * 100,
    ocv_interp(soc_dense),
    "k--",
    lw=1.5,
    label="Extracted OCV Fit (Rest)",
)

# 3. Highlight Evaluated Target SOC Operating Points (0.1, 0.3, 0.6, 0.9)
for soc_val in target_socs:
    ocv_est_v = float(ocv_interp(soc_val))
    
    # Find closest ground-truth OCV value to compute error in mV
    idx_closest = (ocv_gt["soc_point"] - soc_val).abs().idxmin()
    ocv_gt_v = float(ocv_gt.loc[idx_closest, "ocv_true"])
    err_mv = (ocv_est_v - ocv_gt_v) * 1000

    ax4.plot(
        soc_val * 100,
        ocv_est_v,
        "o",
        color=colors[soc_val],
        ms=9,
        zorder=5,
        label=f"SOC {int(soc_val*100)}% (err: {err_mv:+.1f} mV)",
    )
    ax4.annotate(
        f"{ocv_est_v:.3f}V",
        (soc_val * 100, ocv_est_v),
        textcoords="offset points",
        xytext=(-15, 10),
        fontweight="bold",
        color=colors[soc_val],
    )

ax4.set_xlabel("State of Charge (%)")
ax4.set_ylabel("Equilibrium OCV (V)")
ax4.set_title("4. OCV–SOC Curve: Rest Extraction vs. Ground Truth")
ax4.grid(True)
ax4.legend(loc="lower right", fontsize=8.5)

plt.tight_layout()
plt.savefig(BASE_DIR / "hppc_soc_breakdown.png", dpi=300)
plt.show()