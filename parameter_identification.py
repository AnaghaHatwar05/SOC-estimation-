import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

data = pd.read_csv("hppc_data.csv")
ocv_lut = pd.read_csv("ocv_soc_table.csv")
ocv_fn = interp1d(ocv_lut["soc"], ocv_lut["ocv"], kind="cubic", fill_value="extrapolate")

def identify_1rc(V_L, I_L, V_oc, T):
    # E_L > 0 during discharge (I_L > 0)
    E_L = V_oc - V_L

    Psi = np.column_stack([
        E_L[:-1],   # E_L(k-1)
        I_L[1:],    # I_L(k)
        I_L[:-1],   # I_L(k-1)
    ])
    U_target = E_L[1:]

    theta, _, _, _ = np.linalg.lstsq(Psi, U_target, rcond=None)
    a1, a2, a3 = theta

    # Bilinear transform (Tustin) with PyBaMM sign convention (I > 0 = discharge)
    R0 = (a2 - a3) / (1.0 + a1)
    tau1 = (T / 2.0) * ((1.0 + a1) / (1.0 - a1))
    RTh = ((a2 + a3) / (1.0 - a1)) - R0
    CTh = tau1 / RTh if RTh > 1e-6 else np.nan

    return {"R0": R0, "RTh": RTh, "CTh": CTh, "tau1": tau1}

results_1rc = []
Ts = 0.5  # Fine uniform resampling step (0.5 seconds)

for soc_pt, grp in data.groupby("soc_point"):
    # Slice strictly to the active pulse window (t >= 7195s)
    grp_pulse = grp[grp["time"] >= 7195.0].copy()

    # Resample onto uniform grid so Ts is exact
    t_uniform = np.arange(grp_pulse["time"].min(), grp_pulse["time"].max(), Ts)
    V_L = np.interp(t_uniform, grp_pulse["time"], grp_pulse["voltage"])
    I_L = np.interp(t_uniform, grp_pulse["time"], grp_pulse["current"])
    soc_uniform = np.interp(t_uniform, grp_pulse["time"], grp_pulse["soc"])
    V_oc = ocv_fn(soc_uniform)

    params = identify_1rc(V_L, I_L, V_oc, T=Ts)
    params["soc"] = soc_pt
    results_1rc.append(params)

df_1rc = pd.DataFrame(results_1rc)[["soc", "R0", "RTh", "CTh", "tau1"]]
df_1rc.to_csv("params_1rc.csv", index=False)
print(df_1rc.to_string(index=False))