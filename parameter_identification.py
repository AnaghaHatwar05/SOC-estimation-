# day3_param_identification.py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

data    = pd.read_csv("hppc_data.csv")
ocv_lut = pd.read_csv("ocv_soc_table.csv")
ocv_fn  = interp1d(ocv_lut["soc"], ocv_lut["ocv"],
                    kind="cubic", fill_value="extrapolate")

# ── 1RC Least-Squares (Das & Samanta Eq. 8, 11) ──────────────────────────
def identify_1rc(V_L, I_L, V_oc):
    """
    Builds the observation matrix Psi and solves theta = (Psi'Psi)^-1 Psi' U_L
    following Das & Samanta (2022) exactly.
    """
    E_L = V_oc - V_L          # impedance voltage drop, Eq.(2)
    m   = len(E_L)
    # Observation matrix: rows are [U_L(k-1), I_L(k), I_L(k-1)]
    # (from discrete difference Eq. 8 in Paper 2)
    Psi = np.column_stack([
        E_L[:-1],              # E_L(k-1)
        I_L[1:],               # I_L(k)
        I_L[:-1],              # I_L(k-1)
    ])
    U_target = E_L[1:]         # E_L(k), the output
    # Least-squares: theta = [a1, a2, a3]
    theta, _, _, _ = np.linalg.lstsq(Psi, U_target, rcond=None)
    a1, a2, a3 = theta
    # Convert back to physical parameters (Paper 2 Eq. after Eq.5)
    T  = np.mean(np.diff(data["time"]))   # sample time
    R0 = (a3 - a2) / (1 - a1)
    tau1 = T * (1 - a1) / (2 * (1 + a1))
    RTh  = (2*(a2 - a1*a3)*(1 - a1)) / (1 - a1**2)
    CTh  = tau1 / RTh if RTh > 0 else np.nan
    return {"R0": R0, "RTh": RTh, "CTh": CTh, "tau1": tau1}

# Run at each SOC point
results_1rc = []
for soc_pt, grp in data.groupby("soc_point"):
    V_L  = grp["voltage"].values
    I_L  = grp["current"].values
    V_oc = float(ocv_fn(soc_pt)) * np.ones(len(V_L))
    params = identify_1rc(V_L, I_L, V_oc)
    params["soc"] = soc_pt
    results_1rc.append(params)

df_1rc = pd.DataFrame(results_1rc)
df_1rc.to_csv("params_1rc.csv", index=False)
print(df_1rc)