import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Load HPPC dataset, 1-RC, 2-RC parameters, and OCV lookup table
data = pd.read_csv(BASE_DIR / "hppc_data.csv")
params_1rc_df = pd.read_csv(BASE_DIR / "params_1rc.csv")
params_2rc_df = pd.read_csv(BASE_DIR / "params_2rc.csv")
ocv_lut = pd.read_csv(BASE_DIR / "ocv_soc_table.csv")

ocv_fn = interp1d(
    ocv_lut["soc"], ocv_lut["ocv"], kind="cubic", fill_value="extrapolate"
)


# 2. Forward ECM Simulators
def simulate_1rc_ecm(time, current, soc, R0, RTh, CTh, ocv_lookup):
    """Simulates 1-RC ECM terminal voltage."""
    n = len(time)
    V_ecm = np.zeros(n)
    v_rc = 0.0
    tau = RTh * CTh
    V_ecm[0] = float(ocv_lookup(soc[0])) - current[0] * R0

    for k in range(1, n):
        dt = time[k] - time[k - 1]
        if dt <= 0:
            V_ecm[k] = V_ecm[k - 1]
            continue
        alpha = np.exp(-dt / tau) if tau > 1e-6 else 0.0
        v_rc = alpha * v_rc + RTh * (1.0 - alpha) * current[k - 1]
        V_oc = float(ocv_lookup(soc[k]))
        V_ecm[k] = V_oc - current[k] * R0 - v_rc

    return V_ecm


def simulate_2rc_ecm(time, current, soc, R0, R1, C1, R2, C2, ocv_lookup):
    """Simulates 2-RC ECM terminal voltage."""
    n = len(time)
    V_ecm = np.zeros(n)
    v1 = 0.0
    v2 = 0.0
    tau1 = R1 * C1
    tau2 = R2 * C2
    V_ecm[0] = float(ocv_lookup(soc[0])) - current[0] * R0

    for k in range(1, n):
        dt = time[k] - time[k - 1]
        if dt <= 0:
            V_ecm[k] = V_ecm[k - 1]
            continue
        a1 = np.exp(-dt / tau1) if tau1 > 1e-5 else 0.0
        a2 = np.exp(-dt / tau2) if tau2 > 1e-5 else 0.0
        v1 = a1 * v1 + R1 * (1.0 - a1) * current[k - 1]
        v2 = a2 * v2 + R2 * (1.0 - a2) * current[k - 1]
        V_oc = float(ocv_lookup(soc[k]))
        V_ecm[k] = V_oc - current[k] * R0 - v1 - v2

    return V_ecm


# 3. Validate Across Every SOC Point
metrics = []
pulse_traces = {}

soc_points = sorted(params_1rc_df["soc"].unique())

for soc_pt in soc_points:
    row_1rc = params_1rc_df[np.isclose(params_1rc_df["soc"], soc_pt, atol=0.01)].iloc[0]
    row_2rc = params_2rc_df[np.isclose(params_2rc_df["soc"], soc_pt, atol=0.01)].iloc[0]

    grp = data[np.isclose(data["soc_point"], soc_pt, atol=0.01)].copy()
    if grp.empty:
        continue

    t = grp["time"].values
    I = grp["current"].values
    V_true = grp["voltage"].values
    soc_arr = grp["soc"].values

    # Run 1-RC and 2-RC Forward Simulations
    V_pred_1rc = simulate_1rc_ecm(
        t, I, soc_arr, row_1rc["R0"], row_1rc["RTh"], row_1rc["CTh"], ocv_fn
    )
    V_pred_2rc = simulate_2rc_ecm(
        t, I, soc_arr, row_2rc["R0"], row_2rc["R1"], row_2rc["C1"],
        row_2rc["R2"], row_2rc["C2"], ocv_fn
    )

    # Active pulse window (t >= 7190s)
    pulse_mask = t >= 7190.0
    err_1rc_mv = (V_pred_1rc[pulse_mask] - V_true[pulse_mask]) * 1000.0
    err_2rc_mv = (V_pred_2rc[pulse_mask] - V_true[pulse_mask]) * 1000.0

    rmse_1rc = np.sqrt(np.mean(err_1rc_mv**2))
    mae_1rc = np.mean(np.abs(err_1rc_mv))
    max_1rc = np.max(np.abs(err_1rc_mv))

    rmse_2rc = np.sqrt(np.mean(err_2rc_mv**2))
    mae_2rc = np.mean(np.abs(err_2rc_mv))
    max_2rc = np.max(np.abs(err_2rc_mv))

    metrics.append({
        "SOC (%)": int(round(soc_pt * 100)),
        "1-RC RMSE (mV)": round(rmse_1rc, 3),
        "2-RC RMSE (mV)": round(rmse_2rc, 3),
        "1-RC MAE (mV)": round(mae_1rc, 3),
        "2-RC MAE (mV)": round(mae_2rc, 3),
        "1-RC Max (mV)": round(max_1rc, 3),
        "2-RC Max (mV)": round(max_2rc, 3),
        "Improvement (mV)": round(rmse_1rc - rmse_2rc, 3),
    })

    pulse_traces[round(soc_pt, 1)] = {
        "t_rel": t[pulse_mask] - t[pulse_mask][0],
        "V_true": V_true[pulse_mask],
        "V_1rc": V_pred_1rc[pulse_mask],
        "V_2rc": V_pred_2rc[pulse_mask],
        "err_1rc": err_1rc_mv,
        "err_2rc": err_2rc_mv,
        "rmse_1rc": rmse_1rc,
        "rmse_2rc": rmse_2rc,
    }

metrics_df = pd.DataFrame(metrics)
print("\n" + "=" * 86)
print("       1-RC vs 2-RC  ECM PARAMETER VALIDATION COMPARISON")
print("=" * 86)
print(metrics_df.to_string(index=False))
print("-" * 86)
print(f"Overall Mean RMSE  ->  1-RC: {metrics_df['1-RC RMSE (mV)'].mean():.3f} mV  |  2-RC: {metrics_df['2-RC RMSE (mV)'].mean():.3f} mV")
print(f"Mean RMSE Reduction with 2-RC: {metrics_df['Improvement (mV)'].mean():.3f} mV")
print("=" * 86 + "\n")

metrics_df.to_csv(BASE_DIR / "validation_metrics.csv", index=False)
print(f"Saved comparison metrics to: {BASE_DIR / 'validation_metrics.csv'}")

# 4. Plot Comparison at Key SOC Points (0.1, 0.3, 0.6, 0.9)
target_socs = [0.1, 0.3, 0.6, 0.9]
available_targets = [s for s in target_socs if s in pulse_traces]

fig, axs = plt.subplots(2, 2, figsize=(14, 9))
axs = axs.flatten()

for idx, soc_val in enumerate(available_targets):
    ax = axs[idx]
    tr = pulse_traces[soc_val]

    ax.plot(tr["t_rel"], tr["V_true"], "k-", lw=2.2, label="PyBaMM Ground Truth")
    ax.plot(tr["t_rel"], tr["V_1rc"], "b--", lw=1.6, label=f"1-RC (RMSE: {tr['rmse_1rc']:.2f} mV)")
    ax.plot(tr["t_rel"], tr["V_2rc"], "r:", lw=2.0, label=f"2-RC (RMSE: {tr['rmse_2rc']:.2f} mV)")

    ax.set_title(f"SOC {int(soc_val*100)}% Pulse Response Comparison", fontweight="bold")
    ax.set_xlabel("Pulse Window Time (s)")
    ax.set_ylabel("Terminal Voltage (V)")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="best")

plt.tight_layout()
plt.savefig(BASE_DIR / "ecm_validation.png", dpi=300)
print(f"Saved validation comparison plot to: {BASE_DIR / 'ecm_validation.png'}")
plt.show()
