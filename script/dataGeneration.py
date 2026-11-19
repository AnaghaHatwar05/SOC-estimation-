import pybamm
import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Load model and parameters
model = pybamm.lithium_ion.SPM()

# Use "Prada2013" for LFP chemistry (2.3 Ah, ~3.2-3.3V plateau)
# Use "Chen2020" for NMC811 chemistry (5.0 Ah, 3.0-4.2V range)
param = pybamm.ParameterValues("Chen2020") 
q_nom = param["Nominal cell capacity [A.h]"]

# 2. Define HPPC pulse protocol
experiment = pybamm.Experiment([
    "Rest for 2 hours",
    "Discharge at 1C for 10 seconds",
    "Rest for 40 seconds",
    "Charge at 1C for 10 seconds",
    "Rest for 40 seconds",
])

soc_points = np.arange(0.9, 0.0, -0.1)
all_data = []

for soc_init in soc_points:
    sim = pybamm.Simulation(
        model, 
        experiment=experiment, 
        parameter_values=param
    )
    sol = sim.solve(initial_soc=soc_init)

    # Calculate true instantaneous SoC (0 to 1)
    discharge_ah = sol["Discharge capacity [A.h]"].entries
    true_soc = soc_init - (discharge_ah / q_nom)

    # Extract ground-truth physics OCV directly from PyBaMM
    try:
        ground_truth_ocv = sol["Battery open-circuit voltage [V]"].entries
    except KeyError:
        ground_truth_ocv = sol["Surface open-circuit voltage [V]"].entries

    df = pd.DataFrame({
        "time": sol["Time [s]"].entries,
        "voltage": sol["Terminal voltage [V]"].entries,
        "current": sol["Current [A]"].entries,
        "soc": true_soc,                   # True instantaneous SoC (0 to 1)
        "ocv_true": ground_truth_ocv,      # Ground-truth OCV from physics model
        "soc_point": round(float(soc_init), 1),
    })
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
data.to_csv(BASE_DIR / "hppc_data.csv", index=False)
print(data.head())