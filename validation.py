import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

data    = pd.read_csv("hppc_data.csv")
ocv_lut = pd.read_csv("ocv_soc_table.csv")
ocv_fn  = interp1d(ocv_lut["soc"], ocv_lut["ocv"],
                    kind="cubic", fill_value="extrapolate")
def identify_1rc(V_L, I_L, V_oc):
    """Identifies 1-RC ECM parameters following Das & Samanta (2022)."""
    # PyBaMM current convention: + for discharge, - for charge
    # Inverse current sign for ECM standard drop formulation if needed:
    I_L_ecm = I_L  # Keep current sign consistent

    E_L = V_oc - V_L  # Impedance voltage drop: E_L = U_oc - U_L[cite: 2]

    # Regressor matrix Psi based on Equation 8[cite: 2]
    Psi = np.column_stack(
        [
            E_L[:-1],  # E_L(k-1)[cite: 2]
            I_L_ecm[1:],  # I_L(k)[cite: 2]
            I_L_ecm[:-1],  # I_L(k-1)[cite: 2]
        ]
    )
    U_target = E_L[1:]  # E_L(k)[cite: 2]

    # Least-squares solver[cite: 2]
    theta, _, _, _ = np.linalg.lstsq(Psi, U_target, rcond=None)
    a1, a2, a3 = theta

    T = np.mean(np.diff(data["time"]))  # Sampling time[cite: 2]

    # Physical parameter conversion (Eq. 5)[cite: 2]
    # Standard formula assumes positive load drop:
    R0 = abs((a3 - a2) / (1 - a1))  # Take abs() to enforce positive physical R0
    tau1 = (T * (1 - a1)) / (2 * (1 + a1))  #[cite: 2]

    # Correct formula for polarization resistance RTh[cite: 2]:
    RTh = (2 * (a2 - a1 * a3)) / (1 - a1**2)  #[cite: 2]
    RTh = abs(RTh)

    # Avoid divide-by-zero or negative capacitance
    CTh = tau1 / RTh if RTh > 1e-6 else np.nan

    return {"R0": R0, "RTh": RTh, "CTh": CTh, "tau1": tau1}