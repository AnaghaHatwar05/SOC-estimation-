import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 1. Load dataset, identified 1-RC parameters, and OCV lookup table
data = pd.read_csv("hppc_data.csv")
params_df = pd.read_csv("params_1rc.csv")
ocv_lut = pd.read_csv("ocv_soc_table.csv")

ocv_fn = interp1d(
    ocv_lut["soc"], ocv_lut["ocv"], kind="cubic", fill_value="extrapolate"
)

# 2. Forward 1-RC ECM Simulator
def simulate_1rc_ecm(time, current, soc, R0, RTh, CTh, ocv_lookup):
    n = len(time)
    V_ecm = np.zeros(n)
    V_rc = 0.0
    tau = RTh * CTh

    V_ecm[0] = float(ocv_lookup(soc[0])) - current[0] * R0

    for k in range(1, n):
        dt = time[k] - time[k - 1]
        if dt <= 0:
            V_ecm[k] = V_ecm[k - 1]
            continue

        alpha = np.exp(-dt / tau) if tau > 1e-6 else 0.0
        V_rc = alpha * V_rc + RTh * (1.0 - alpha) * current[k - 1]
        V_oc = float(ocv_lookup(soc[k]))
        V_ecm[k] = V_oc - current[k] * R0 - V_rc

    return V_ecm


# 3. Validate and Compute Error Metrics Across All SOC Points
metrics = []
pulse_traces = {}

for _, row in params_df.iterrows():
    soc_pt = row["soc"]
    R0, RTh, CTh = row["R0"], row["RTh"], row["CTh"]

    grp = data[np.isclose(data["soc_point"], soc_pt, atol=0.01)].copy()
    if grp.empty or np.isnan(CTh):
        continue

    t = grp["time"].values
    I = grp["current"].values
    V_true = grp["voltage"].values
    soc_arr = grp["soc"].values

    V_pred = simulate_1rc_ecm(t, I, soc_arr, R0, RTh, CTh, ocv_fn)

    # Isolate active pulse & relaxation window (t >= 7190s)
    pulse_mask = t >= 7190.0
    err_mv = (V_pred[pulse_mask] - V_true[pulse_mask]) * 1000.0

    rmse_mv = np.sqrt(np.mean(err_mv**2))
    mae_mv = np.mean(np.abs(err_mv))
    max_err_mv = np.max(np.abs(err_mv))
    bias_mv = np.mean(err_mv)  # Mean signed error (shows systematic offset)

    metrics.append({
        "SOC (%)": int(round(soc_pt * 100)),
        "R0 (mOhm)": round(R0 * 1000, 3),
        "RTh (mOhm)": round(RTh * 1000, 3),
        "CTh (F)": round(CTh, 1),
        "RMSE (mV)": round(rmse_mv, 3),
        "MAE (mV)": round(mae_mv, 3),
        "Max Error (mV)": round(max_err_mv, 3),
        "Mean Bias (mV)": round(bias_mv, 3),
    })

    pulse_traces[round(soc_pt, 1)] = {
        "t_rel": t[pulse_mask] - t[pulse_mask][0],
        "V_true": V_true[pulse_mask],
        "V_pred": V_pred[pulse_mask],
        "err_mv": err_mv,
        "rmse": rmse_mv,
        "max_err": max_err_mv,
    }

# Print formatted error table and export CSV
metrics_df = pd.DataFrame(metrics)
print("\n" + "=" * 78)
print("                 1-RC ECM VALIDATION ERROR SUMMARY (PULSE WINDOW)")
print("=" * 78)
print(metrics_df.to_string(index=False))
print("-" * 78)
print(f"Overall Mean RMSE across all SOCs : {metrics_df['RMSE (mV)'].mean():.3f} mV")
print(f"Worst-Case Peak Error             : {metrics_df['Max Error (mV)'].max():.3f} mV (at {metrics_df.loc[metrics_df['Max Error (mV)'].idxmax(), 'SOC (%)']}% SOC)")
print("=" * 78 + "\n")

metrics_df.to_csv("validation_metrics.csv", index=False)

# 4. Plot Voltage Comparison + Zoomed Millivolt Residual Error Subplots
target_socs = [0.1, 0.3, 0.6, 0.9]
available_targets = [s for s in target_socs if s in pulse_traces]

# 4 rows x 2 columns: Top panel = Voltage (V), Bottom panel = Error (mV) for each SOC
fig, axs = plt.subplots(
    4, 2, figsize=(14, 12), gridspec_kw={"height_ratios": [2.2, 1, 2.2, 1]}
)

for idx, soc_val in enumerate(available_targets):
    col = idx % 2
    row_v = (idx // 2) * 2
    row_e = row_v + 1

    tr = pulse_traces[soc_val]
    ax_v = axs[row_v, col]
    ax_e = axs[row_e, col]

    # Top subplot: Terminal Voltage Comparison
    ax_v.plot(tr["t_rel"], tr["V_true"], "k-", lw=2, label="PyBaMM Ground Truth")
    ax_v.plot(tr["t_rel"], tr["V_pred"], "r--", lw=1.8, label="1-RC ECM Predicted")
    ax_v.set_title(
        f"SOC {int(soc_val*100)}% — RMSE: {tr['rmse']:.2f} mV | Peak Err: {tr['max_err']:.2f} mV",
        fontweight="bold",
    )
    ax_v.set_ylabel("Terminal Voltage (V)")
    ax_v.grid(True, alpha=0.6)
    ax_v.legend(loc="best", fontsize=8.5)
    ax_v.tick_params(labelbottom=False)  # Share x-axis with error plot below

    # Bottom subplot: Residual Error in Millivolts (mV)
    ax_e.plot(tr["t_rel"], tr["err_mv"], color="tab:purple", lw=1.5)
    ax_e.axhline(0, color="black", linestyle=":", lw=1)
    ax_e.fill_between(tr["t_rel"], tr["err_mv"], 0, color="tab:purple", alpha=0.2)
    ax_e.set_ylabel("Error (mV)")
    ax_e.set_xlabel("Pulse Window Time (s)")
    ax_e.grid(True, alpha=0.6)

plt.tight_layout()
plt.savefig("ecm_validation_with_errors.png", dpi=300)
plt.show()