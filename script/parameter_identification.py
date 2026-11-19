import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Load HPPC dataset and extracted Chen2020 OCV table
data = pd.read_csv(BASE_DIR / "hppc_data.csv")
ocv_lut = pd.read_csv(BASE_DIR / "ocv_soc_table.csv")
ocv_fn = interp1d(ocv_lut["soc"], ocv_lut["ocv"], kind="cubic", fill_value="extrapolate")


# ==============================================================================
# 1-RC Parameter Estimation (1st Order Thevenin Model)
# ==============================================================================
def identify_1rc(V_L, I_L, V_oc, T):
    """
    Identifies 1-RC parameters (R0, RTh, CTh, tau1) using discrete ARX
    least-squares and Tustin bilinear transform.
    """
    E_L = V_oc - V_L

    Psi = np.column_stack([
        E_L[:-1],   # E_L(k-1)
        I_L[1:],    # I_L(k)
        I_L[:-1],   # I_L(k-1)
    ])
    U_target = E_L[1:]

    theta, _, _, _ = np.linalg.lstsq(Psi, U_target, rcond=None)
    a1, a2, a3 = theta

    # Bilinear transform (Tustin)
    R0 = (a2 - a3) / (1.0 + a1)
    tau1 = (T / 2.0) * ((1.0 + a1) / (1.0 - a1))
    RTh = ((a2 + a3) / (1.0 - a1)) - R0

    # Physical stability bounds
    R0 = float(np.clip(R0, 0.005, 0.15))
    tau1 = float(np.clip(tau1, 0.5, 300.0))
    RTh = float(np.clip(RTh, 0.001, 0.15))
    CTh = float(np.clip(tau1 / RTh, 50.0, 20000.0))

    return {"R0": R0, "RTh": RTh, "CTh": CTh, "tau1": tau1}


# ==============================================================================
# 2-RC Parameter Estimation (2nd Order Thevenin Model)
# Based on Das & Samanta (IEEE PEDES 2022 / IJCTA 2024):
# "Estimation of Equivalent Circuit Model Parameters for a Generic Battery
#  Using Least-Square Method"
# ==============================================================================
def identify_2rc(V_L, I_L, V_oc, T):
    """
    Identifies 2-RC parameters (R0, R1, C1, tau1, R2, C2, tau2) using
    second-order ARX Least Squares and analytical inverse Tustin transformation.
    
    Continuous Transfer Function:
        G(s) = (V_oc - V_L) / I_L = R0 + R1/(1 + s*tau1) + R2/(1 + s*tau2)
             = (b2*s^2 + b1*s + b0) / (a2*s^2 + a1*s + 1)
             
    Discrete ARX Form via Tustin (s = 2/T * (1 - z^-1)/(1 + z^-1)):
        E_L(k) = -alpha1*E_L(k-1) - alpha2*E_L(k-2) + beta0*I(k) + beta1*I(k-1) + beta2*I(k-2)
    """
    E_L = V_oc - V_L

    Psi = np.column_stack([
        -E_L[1:-1],  # -E_L(k-1)
        -E_L[:-2],   # -E_L(k-2)
        I_L[2:],     # I_L(k)
        I_L[1:-1],   # I_L(k-1)
        I_L[:-2],    # I_L(k-2)
    ])
    U = E_L[2:]

    theta, _, _, _ = np.linalg.lstsq(Psi, U, rcond=None)
    alpha1, alpha2, beta0, beta1, beta2 = theta

    denom_c = 1.0 + alpha1 + alpha2
    if abs(denom_c) < 1e-7:
        denom_c = 1e-7

    denom_inf = 1.0 - alpha1 + alpha2
    if abs(denom_inf) < 1e-7:
        denom_inf = 1e-7

    # Continuous polynomial denominator coefficients: 1 + a1*s + a2*s^2
    a1 = T * (1.0 - alpha2) / denom_c
    a2 = (T**2 / 4.0) * (denom_inf / denom_c)

    # Solve for roots (time constants tau1 and tau2): tau^2 - a1*tau + a2 = 0
    disc = a1**2 - 4.0 * a2
    if disc < 0:
        disc = 0.0

    tau_a = (a1 - np.sqrt(disc)) / 2.0
    tau_b = (a1 + np.sqrt(disc)) / 2.0

    # Branch 1 = fast electrochemical charge transfer (smaller tau)
    # Branch 2 = slow solid-state diffusion (larger tau)
    tau1 = min(tau_a, tau_b)
    tau2 = max(tau_a, tau_b)

    # Resistances:
    # High frequency limit (s -> infinity, z = -1) gives R0:
    R0 = (beta0 - beta1 + beta2) / denom_inf

    # DC steady-state limit (s -> 0, z = 1) gives R_total = R0 + R1 + R2:
    R_total = (beta0 + beta1 + beta2) / denom_c
    delta_R = R_total - R0

    # Numerator s-term gives b1 = R0*a1 + R1*tau2 + R2*tau1:
    b1 = T * (beta0 - beta2) / denom_c
    M = b1 - R0 * a1

    d_tau = tau2 - tau1
    if abs(d_tau) > 1e-5:
        R1 = (M - delta_R * tau1) / d_tau
        R2 = delta_R - R1
    else:
        R1 = delta_R / 2.0
        R2 = delta_R / 2.0

    # Physical stability safeguards:
    R0 = float(np.clip(R0, 0.005, 0.15))
    R1 = float(np.clip(R1, 0.0001, 0.08))
    R2 = float(np.clip(R2, 0.0005, 0.08))
    tau1 = float(np.clip(tau1, 0.1, 50.0))
    tau2 = float(np.clip(tau2, 1.0, 300.0))

    C1 = float(np.clip(tau1 / R1, 50.0, 50000.0))
    C2 = float(np.clip(tau2 / R2, 50.0, 50000.0))

    return {
        "R0": R0,
        "R1": R1,
        "C1": C1,
        "tau1": tau1,
        "R2": R2,
        "C2": C2,
        "tau2": tau2,
    }


# ==============================================================================
# Run Parameter Extraction Across All SOC Points
# ==============================================================================
results_1rc = []
results_2rc = []
Ts = 0.5  # Uniform resampling step (0.5 seconds)

for soc_pt, grp in data.groupby("soc_point"):
    grp_pulse = grp[grp["time"] >= 7195.0].copy()

    # Resample onto uniform grid for exact Ts
    t_uniform = np.arange(grp_pulse["time"].min(), grp_pulse["time"].max(), Ts)
    V_L = np.interp(t_uniform, grp_pulse["time"], grp_pulse["voltage"])
    I_L = np.interp(t_uniform, grp_pulse["time"], grp_pulse["current"])
    soc_uniform = np.interp(t_uniform, grp_pulse["time"], grp_pulse["soc"])
    V_oc = ocv_fn(soc_uniform)

    # 1-RC Identification
    p1 = identify_1rc(V_L, I_L, V_oc, T=Ts)
    p1["soc"] = soc_pt
    results_1rc.append(p1)

    # 2-RC Identification (Das & Samanta Method)
    p2 = identify_2rc(V_L, I_L, V_oc, T=Ts)
    p2["soc"] = soc_pt
    results_2rc.append(p2)

# Save 1-RC Parameters
df_1rc = pd.DataFrame(results_1rc)[["soc", "R0", "RTh", "CTh", "tau1"]]
df_1rc.to_csv(BASE_DIR / "params_1rc.csv", index=False)

# Save 2-RC Parameters
df_2rc = pd.DataFrame(results_2rc)[["soc", "R0", "R1", "C1", "tau1", "R2", "C2", "tau2"]]
df_2rc.to_csv(BASE_DIR / "params_2rc.csv", index=False)

print("=" * 78)
print("             1-RC EQUIVALENT CIRCUIT PARAMETERS")
print("=" * 78)
print(df_1rc.to_string(index=False))
print(f"Saved: {BASE_DIR / 'params_1rc.csv'}\n")

print("=" * 78)
print("     2-RC EQUIVALENT CIRCUIT PARAMETERS (DAS & SAMANTA METHOD)")
print("=" * 78)
print(df_2rc.to_string(index=False))
print(f"Saved: {BASE_DIR / 'params_2rc.csv'}")
print("=" * 78)

print(f"Saved: {BASE_DIR / 'params_1rc.csv'}\n")

print("=" * 78)
print("     2-RC EQUIVALENT CIRCUIT PARAMETERS (DAS & SAMANTA METHOD)")
print("=" * 78)
print(df_2rc.to_string(index=False))
print(f"Saved: {BASE_DIR / 'params_2rc.csv'}")
print("=" * 78)