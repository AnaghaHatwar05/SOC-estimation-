# SOC Estimation

This project generates a synthetic HPPC data set with PyBaMM, extracts an OCV-SOC lookup table, identifies 1-RC and 2-RC equivalent-circuit parameters, and evaluates voltage and SOC estimates.

## Requirements

Use Python with these packages installed:

- NumPy
- pandas
- SciPy
- Matplotlib
- PyBaMM

Install any missing packages with:

```powershell
python -m pip install numpy pandas scipy matplotlib pybamm
```

Run commands from this project directory (`SOC-estimation-`). All scripts read and write their CSV and image files in this directory, even though the scripts are kept in `script\`.

## Run the workflow

Run these scripts in order when starting from scratch:

```powershell
python .\script\dataGeneration.py
python .\script\ocv_soc.py
python .\script\parameter_identification.py
```

Then run whichever analyses you need:

```powershell
python .\script\validation.py
python .\script\error.py
python .\script\plot_parameters.py
python .\script\SOC.py
```

`SOC.py` simulates a separate Chen2020 dynamic drive cycle and compares 1-RC and 2-RC EKF SOC estimates against PyBaMM ground truth.

## Scripts and outputs

| Script | Purpose | Main outputs |
|---|---|---|
| `script\dataGeneration.py` | Simulates HPPC data at SOC values 0.1–0.9 using Chen2020. The current protocol begins with a 2-hour rest. | `hppc_data.csv` |
| `script\ocv_soc.py` | Extracts the rest-voltage OCV-SOC table and creates HPPC/OCV plots. | `ocv_soc_table.csv`, `hppc_soc_breakdown.png` |
| `script\parameter_identification.py` | Identifies 1-RC and 2-RC parameters by SOC. | `params_1rc.csv`, `params_2rc.csv` |
| `script\validation.py` | Compares 1-RC and 2-RC voltage predictions on the HPPC pulse data. | `validation_metrics.csv`, `ecm_validation.png` |
| `script\error.py` | Reports 1-RC voltage prediction errors. | `ecm_validation_with_errors.png`, metrics CSV |
| `script\plot_parameters.py` | Plots each model's fitted parameters versus SOC. | `fitted_parameters_1rc_vs_soc.png`, `fitted_parameters_2rc_vs_soc.png` |
| `script\SOC.py` | Runs 1-RC and 2-RC EKF SOC estimates on a simulated dynamic drive cycle. | `soc_estimation_verification.png` |

## Notes

- The generated HPPC data and the SOC drive-cycle test are synthetic PyBaMM simulations, not measurements from a physical cell.
- `ocv_soc.py` extracts rest voltage at or before 7200 seconds, matching the current 2-hour initial rest. If the rest duration changes, update the extraction cutoff accordingly.
- Both `error.py` and `validation.py` currently write to `validation_metrics.csv`; running one overwrites the metrics written by the other.
