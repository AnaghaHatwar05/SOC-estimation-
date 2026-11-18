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

# 2. Forward 1-RC ECM Simulator (Discrete-Time State-Space)
def simulate_1rc_ecm(time, current, soc, R0, RTh, CTh, ocv_lookup):
    """
    Simulates terminal voltage V_ecm(t) for a 1-RC Equivalent Circuit Model:
        V_rc[k]  = exp(-dt / tau) * V_rc[k-1] + RTh * (1 - exp(-dt / tau)) * I[k-1]
        V_ecm[k] = OCV(soc[k]) - I[k] * R0 - V_rc[k]
    """
    n = len(time)
    V_ecm = np.zeros(n)
    V_rc = 0.0  # Polarization voltage starts at 0 after initial 2-hour rest
    tau = RTh * CTh

    # Initial step
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


# 3. Validate across every SOC point in params_1rc.csv
metrics = []
pulse_traces = {}  # Store active pulse window traces for plotting

for _, row in params_df.iterrows():
    soc_pt = row["soc"]
    R0, RTh, CTh = row["R0"], row["RTh"], row["CTh"]

    # Filter HPPC data for this SOC point
    grp = data[np.isclose(data["soc_point"], soc_pt, atol=0.01)].copy()
    if grp.empty or np.isnan(CTh):
        continue

    t = grp["time"].values
    I = grp["current"].values
    V_true = grp["voltage"].values
    soc_arr = grp["soc"].values

    # Run 1-RC ECM model
    V_pred = simulate_1rc_ecm(t, I, soc_arr, R0, RTh, CTh, ocv_fn)

    # Evaluate error specifically during the active pulse & relaxation window (t >= 7190s)
    pulse_mask = t >= 7190.0
    err_mv = (V_pred[pulse_mask] - V_true[pulse_mask]) * 1000.0
    rmse_mv = np.sqrt(np.mean(err_mv**2))
    mae_mv = np.mean(np.abs(err_mv))
    max_err_mv = np.max(np.abs(err_mv))

    metrics.append({
        "soc": soc_pt,
        "R0 (Ohm)": round(R0, 5),
        "RTh (Ohm)": round(RTh, 5),
        "CTh (F)": round(CTh, 1),
        "RMSE (mV)": round(rmse_mv, 3),
        "MAE (mV)": round(mae_mv, 3),
        "Max Error (mV)": round(max_err_mv, 3),
    })

    # Save active pulse traces for visualization
    pulse_traces[round(soc_pt, 1)] = {
        "t_rel": t[pulse_mask] - t[pulse_mask][0],
        "V_true": V_true[pulse_mask],
        "V_pred": V_pred[pulse_mask],
        "err_mv": err_mv,
        "rmse": rmse_mv,
    }

metrics_df = pd.DataFrame(metrics)
print("\n=== 1-RC ECM Validation Results (Active HPPC Pulse Window) ===")
print(metrics_df.to_string(index=False))

# 4. Plot Comparison at Key SOC Points (0.1, 0.3, 0.6, 0.9)
target_socs = [0.1, 0.3, 0.6, 0.9]
available_targets = [s for s in target_socs if s in pulse_traces]

fig, axs = plt.subplots(2, 2, figsize=(14, 9))
axs = axs.flatten()

for idx, soc_val in enumerate(available_targets):
    ax = axs[idx]
    tr = pulse_traces[soc_val]

    ax.plot(tr["t_rel"], tr["V_true"], "k-", lw=2, label="PyBaMM Ground Truth")
    ax.plot(tr["t_rel"], tr["V_pred"], "r--", lw=1.8, label="1-RC ECM Predicted")

    ax.set_title(f"SOC {int(soc_val*100)}% — RMSE: {tr['rmse']:.2f} mV")
    ax.set_xlabel("Pulse Window Time (s)")
    ax.set_ylabel("Terminal Voltage (V)")
    ax.grid(True)
    ax.legend(loc="best")

plt.tight_layout()
plt.savefig("ecm_validation.png", dpi=300)
plt.show()