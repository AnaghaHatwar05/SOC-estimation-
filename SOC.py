import pybamm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 1. Inspect your loaded tables first to confirm Chen2020 alignment
ocv_lut = pd.read_csv(BASE_DIR / "ocv_soc_table.csv").sort_values("soc")
params_df = pd.read_csv(BASE_DIR / "params_1rc.csv").sort_values("soc")
params_2rc_df = pd.read_csv(BASE_DIR / "params_2rc.csv").sort_values("soc")

print("--- Loaded OCV Table ---")
print(ocv_lut.to_string(index=False))
print("\n--- Loaded 1-RC Parameters ---")
print(params_df.to_string(index=False))
print("\n--- Loaded 2-RC Parameters ---")
print(params_2rc_df.to_string(index=False))

ocv_fn = PchipInterpolator(ocv_lut["soc"], ocv_lut["ocv"], extrapolate=True)
docv_fn = ocv_fn.derivative()

R0_fn = interp1d(params_df["soc"], params_df["R0"], kind="linear", fill_value="extrapolate")
RTh_fn = interp1d(params_df["soc"], params_df["RTh"], kind="linear", fill_value="extrapolate")
CTh_fn = interp1d(params_df["soc"], params_df["CTh"], kind="linear", fill_value="extrapolate")
R1_fn = interp1d(params_2rc_df["soc"], params_2rc_df["R1"], kind="linear", fill_value="extrapolate")
C1_fn = interp1d(params_2rc_df["soc"], params_2rc_df["C1"], kind="linear", fill_value="extrapolate")
R2_fn = interp1d(params_2rc_df["soc"], params_2rc_df["R2"], kind="linear", fill_value="extrapolate")
C2_fn = interp1d(params_2rc_df["soc"], params_2rc_df["C2"], kind="linear", fill_value="extrapolate")

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


def run_2rc_ekf_soc(time, current, voltage, q_nom_ah, soc_guess_init):
    n = len(time)
    soc_est = np.zeros(n)
    x = np.array([soc_guess_init, 0.0, 0.0])
    soc_est[0] = x[0]

    P = np.diag([2e-2, 1e-4, 1e-4])
    Q = np.diag([1e-7, 1e-5, 1e-5])
    R = np.array([[1e-3]])
    q_nom_coulombs = q_nom_ah * 3600.0

    for k in range(1, n):
        dt = time[k] - time[k - 1]
        soc_prev, v1_prev, v2_prev = x

        R0 = float(np.clip(R0_fn(soc_prev), 1e-4, 1.0))
        R1 = float(np.clip(R1_fn(soc_prev), 1e-4, 1.0))
        C1 = float(np.clip(C1_fn(soc_prev), 10.0, 1e5))
        R2 = float(np.clip(R2_fn(soc_prev), 1e-4, 1.0))
        C2 = float(np.clip(C2_fn(soc_prev), 10.0, 1e5))
        alpha1 = np.exp(-dt / (R1 * C1))
        alpha2 = np.exp(-dt / (R2 * C2))

        current_prev = current[k - 1]
        soc_pred = float(np.clip(
            soc_prev - (dt / q_nom_coulombs) * current_prev, 0.01, 0.99
        ))
        v1_pred = alpha1 * v1_prev + R1 * (1.0 - alpha1) * current_prev
        v2_pred = alpha2 * v2_prev + R2 * (1.0 - alpha2) * current_prev
        x_pred = np.array([soc_pred, v1_pred, v2_pred])

        A = np.diag([1.0, alpha1, alpha2])
        P_pred = A @ P @ A.T + Q

        V_hat = (
            float(ocv_fn(soc_pred))
            - current[k] * R0
            - v1_pred
            - v2_pred
        )
        dOCV = float(np.clip(docv_fn(soc_pred), 0.1, 5.0))
        H = np.array([[dOCV, -1.0, -1.0]])

        y_err = voltage[k] - V_hat
        S = H @ P_pred @ H.T + R
        K = (P_pred @ H.T) / S[0, 0]

        x = x_pred + (K.flatten() * y_err)
        x[0] = float(np.clip(x[0], 0.01, 0.99))
        P = (np.eye(3) - K @ H) @ P_pred
        soc_est[k] = x[0]

    return soc_est


soc_estimated = run_ekf_soc(t_grid, I_meas, V_meas, q_nom_ah, soc_guess_init=0.65)
soc_estimated_2rc = run_2rc_ekf_soc(
    t_grid, I_meas, V_meas, q_nom_ah, soc_guess_init=0.65
)
soc_err_pct = (soc_estimated - soc_true) * 100.0
soc_err_2rc_pct = (soc_estimated_2rc - soc_true) * 100.0
ss_rmse = np.sqrt(np.mean(soc_err_pct[t_grid >= 120]**2))
ss_rmse_2rc = np.sqrt(np.mean(soc_err_2rc_pct[t_grid >= 120]**2))
print(
    f"\n1-RC Steady-State SOC RMSE (>2 min): {ss_rmse:.3f}% "
    f"| Final Error: {soc_err_pct[-1]:+.3f}%"
)
print(
    f"2-RC Steady-State SOC RMSE (>2 min): {ss_rmse_2rc:.3f}% "
    f"| Final Error: {soc_err_2rc_pct[-1]:+.3f}%"
)

# 4. Plot
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axs[0].plot(t_grid / 60.0, soc_true * 100.0, "k-", lw=2.2, label="PyBaMM Ground-Truth SOC")
axs[0].plot(t_grid / 60.0, soc_estimated * 100.0, "r--", lw=2.0, label="1-RC EKF Estimated SOC")
axs[0].plot(t_grid / 60.0, soc_estimated_2rc * 100.0, "b:", lw=2.0, label="2-RC EKF Estimated SOC")
axs[0].set_ylabel("SOC (%)")
axs[0].set_title(
    f"Chen2020 EKF SOC Estimation — 1-RC RMSE: {ss_rmse:.2f}%, "
    f"2-RC RMSE: {ss_rmse_2rc:.2f}%"
)
axs[0].grid(True, alpha=0.6)
axs[0].legend()

axs[1].plot(t_grid / 60.0, soc_err_pct, color="tab:purple", lw=1.5, label="1-RC SOC Error")
axs[1].plot(t_grid / 60.0, soc_err_2rc_pct, color="tab:blue", lw=1.5, ls=":", label="2-RC SOC Error")
axs[1].axhline(0, color="black", ls=":", lw=1)
axs[1].set_ylabel("SOC Error (%)")
axs[1].grid(True, alpha=0.6)
axs[1].legend()

axs[2].plot(t_grid / 60.0, I_true, color="tab:green", lw=1.3, label="Drive Cycle Current")
axs[2].set_ylabel("Current (A)")
axs[2].set_xlabel("Time (minutes)")
axs[2].grid(True, alpha=0.6)
axs[2].legend()

plt.tight_layout()
plt.savefig(BASE_DIR / "soc_estimation_verification.png", dpi=300)
print(f"Plot saved to: {BASE_DIR / 'soc_estimation_verification.png'}")
plt.show()