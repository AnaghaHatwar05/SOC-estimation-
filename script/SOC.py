import pybamm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Inspect your loaded tables first to confirm Chen2020 alignment
ocv_lut = pd.read_csv(BASE_DIR / "ocv_soc_table.csv").sort_values("soc")
params_df = pd.read_csv(BASE_DIR / "params_1rc.csv").sort_values("soc")

print("--- Loaded OCV Table ---")
print(ocv_lut.to_string(index=False))
print("\n--- Loaded 1-RC Parameters ---")
print(params_df.to_string(index=False))

ocv_fn = PchipInterpolator(ocv_lut["soc"], ocv_lut["ocv"], extrapolate=True)
docv_fn = ocv_fn.derivative()

R0_fn = interp1d(params_df["soc"], params_df["R0"], kind="linear", fill_value="extrapolate")
RTh_fn = interp1d(params_df["soc"], params_df["RTh"], kind="linear", fill_value="extrapolate")
CTh_fn = interp1d(params_df["soc"], params_df["CTh"], kind="linear", fill_value="extrapolate")

# 2. Simulate Chen2020 Dynamic Drive Cycle
model = pybamm.lithium_ion.SPM()
param = pybamm.ParameterValues("Chen2020")
q_nom_ah = param["Nominal cell capacity [A.h]"]  # 5.0 Ah

drive_steps = [
    "Rest for 30 seconds",
    "Discharge at 0.5C for 60 seconds",
    "Discharge at 1.5C for 25 seconds",
    "Charge at 0.8C for 20 seconds",
    "Rest for 15 seconds",
    "Discharge at 2.0C for 30 seconds",
    "Discharge at 0.7C for 90 seconds",
    "Charge at 1.0C for 25 seconds",
    "Discharge at 1.2C for 60 seconds",
    "Rest for 45 seconds",
] * 4

sol = pybamm.Simulation(model, experiment=pybamm.Experiment(drive_steps), parameter_values=param).solve(initial_soc=0.80)

Ts = 1.0
t_raw = sol["Time [s]"].entries
t_grid = np.arange(0, t_raw[-1], Ts)

V_true = np.interp(t_grid, t_raw, sol["Terminal voltage [V]"].entries)
I_true = np.interp(t_grid, t_raw, sol["Current [A]"].entries)
soc_true = 0.80 - (np.interp(t_grid, t_raw, sol["Discharge capacity [A.h]"].entries) / q_nom_ah)

rng = np.random.default_rng(42)
V_meas = V_true + rng.normal(0, 0.003, size=len(t_grid))
I_meas = I_true + rng.normal(0, 0.015, size=len(t_grid))

# 3. Properly Tuned EKF (Coulomb counting + gentle voltage correction)
def run_ekf_soc(time, current, voltage, q_nom_ah, soc_guess_init):
    n = len(time)
    soc_est = np.zeros(n)
    x = np.array([soc_guess_init, 0.0])
    soc_est[0] = x[0]

    # Key Fix: Q[0,0] = 1e-7 allows dynamic tracking; P[0,0] = 2e-2 fixes initial error fast
    P = np.diag([2e-2, 1e-4])
    Q = np.diag([1e-7, 1e-5])
    R = np.array([[1e-3]])

    q_nom_coulombs = q_nom_ah * 3600.0

    for k in range(1, n):
        dt = time[k] - time[k - 1]
        soc_prev, v_rc_prev = x[0], x[1]

        R0 = float(np.clip(R0_fn(soc_prev), 1e-4, 1.0))
        RTh = float(np.clip(RTh_fn(soc_prev), 1e-4, 1.0))
        CTh = float(np.clip(CTh_fn(soc_prev), 10.0, 1e5))
        tau = RTh * CTh
        alpha = np.exp(-dt / tau)

        # Predict
        soc_pred = float(np.clip(soc_prev - (dt / q_nom_coulombs) * current[k - 1], 0.01, 0.99))
        v_rc_pred = alpha * v_rc_prev + RTh * (1.0 - alpha) * current[k - 1]
        x_pred = np.array([soc_pred, v_rc_pred])

        A = np.array([[1.0, 0.0], [0.0, alpha]])
        P_pred = A @ P @ A.T + Q

        # Update
        V_hat = float(ocv_fn(soc_pred)) - current[k] * R0 - v_rc_pred
        dOCV = float(np.clip(docv_fn(soc_pred), 0.1, 5.0))
        H = np.array([[dOCV, -1.0]])

        y_err = voltage[k] - V_hat
        S = H @ P_pred @ H.T + R
        K = (P_pred @ H.T) / S[0, 0]

        x = x_pred + (K.flatten() * y_err)
        x[0] = float(np.clip(x[0], 0.01, 0.99))
        P = (np.eye(2) - K @ H) @ P_pred

        soc_est[k] = x[0]

    return soc_est

soc_estimated = run_ekf_soc(t_grid, I_meas, V_meas, q_nom_ah, soc_guess_init=0.65)
soc_err_pct = (soc_estimated - soc_true) * 100.0
ss_rmse = np.sqrt(np.mean(soc_err_pct[t_grid >= 120]**2))
print(f"\nSteady-State SOC RMSE (>2 min): {ss_rmse:.3f}% | Final Error: {soc_err_pct[-1]:+.3f}%")

# 4. Plot
fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axs[0].plot(t_grid / 60.0, soc_true * 100.0, "k-", lw=2.2, label="PyBaMM Ground-Truth SOC")
axs[0].plot(t_grid / 60.0, soc_estimated * 100.0, "r--", lw=2.0, label="EKF Estimated SOC (Init: 65%)")
axs[0].set_ylabel("SOC (%)")
axs[0].set_title(f"Chen2020 EKF SOC Estimation — Steady-State RMSE: {ss_rmse:.2f}%")
axs[0].grid(True, alpha=0.6)
axs[0].legend()

axs[1].plot(t_grid / 60.0, soc_err_pct, color="tab:purple", lw=1.5)
axs[1].axhline(0, color="black", ls=":", lw=1)
axs[1].set_ylabel("SOC Error (%)")
axs[1].set_xlabel("Time (minutes)")
axs[1].grid(True, alpha=0.6)
plt.tight_layout()
plt.savefig(BASE_DIR / "soc_estimation_verification.png", dpi=300)
print(f"Plot saved to: {BASE_DIR / 'soc_estimation_verification.png'}")
plt.show()