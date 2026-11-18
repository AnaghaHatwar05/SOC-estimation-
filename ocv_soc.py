import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 1. Load HPPC dataset
data = pd.read_csv("hppc_data.csv")

# 2. Extract global OCV-SOC lookup table
rest_mask = np.abs(data["current"]) < 0.01
ocv_summary = data[rest_mask].groupby("soc_point")["voltage"].mean().reset_index()
ocv_summary.columns = ["soc", "ocv"]
ocv_summary = ocv_summary.sort_values("soc").reset_index(drop=True)
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
        # Align time relative to start of the pulse sequence for this SOC
        t_rel = df_soc["time"] - df_soc["time"].iloc[0] if "time" in df_soc.columns else df_soc.index - df_soc.index[0]
        ax1.plot(t_rel, df_soc["voltage"], label=f"SOC {int(soc_val*100)}%", color=colors[soc_val], lw=1.8)

ax1.set_xlabel("Relative Pulse Time (s)")
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
ax4 = axs[1, 1]
soc_dense = np.linspace(ocv_summary["soc"].min(), ocv_summary["soc"].max(), 200)
ax4.plot(soc_dense * 100, ocv_interp(soc_dense), "k--", lw=1.5, label="OCV Curve Fit")

for soc_val in target_socs:
    ocv_v = ocv_interp(soc_val)
    ax4.plot(soc_val * 100, ocv_v, "o", color=colors[soc_val], ms=10, label=f"Target {int(soc_val*100)}%")
    ax4.annotate(f"{ocv_v:.3f}V", (soc_val * 100, ocv_v), textcoords="offset points", 
                 xytext=(-15, 10), fontweight="bold", color=colors[soc_val])

ax4.set_xlabel("State of Charge (%)")
ax4.set_ylabel("Equilibrium OCV (V)")
ax4.set_title("4. Full OCV Curve & Evaluated SOC Operating Points")
ax4.grid(True)
ax4.legend(loc="lower right")

plt.tight_layout()
plt.savefig("hppc_soc_breakdown.png", dpi=300)
plt.show()