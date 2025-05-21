import os
import subprocess
import numpy as np

# Output directory
output_dir = "rossler_data_eps_range"
os.makedirs(output_dir, exist_ok=True)
ROSSLER_SCRIPT = "d:/Stat/Causal/Causality-in-Dynamic-Models/dynamical-systems/rossler_systems.py"

# Fixed parameters
base_command = [
    "python", ROSSLER_SCRIPT,
    "--omega_1", "1",
    "--omega_2", "1",
    "--nsamples", "10000",
    "--dt", "0.05",
    "--undersample_factor", "2",
    "--seed", "42",
    "--x_0", "10", "15", "30",
    "--integrator", "dop853"
]

# Loop over epsilon values
for eps in np.arange(0, 1.05, 0.05):  # up to and including 1
    eps_str = f"{eps:.2f}"
    output_filename = f"{output_dir}/rossler_eps_{eps_str}.p"
    
    full_command = base_command + ["--epsilon", eps_str, "--output", output_filename]
    print(f"Running: {' '.join(full_command)}")
    subprocess.run(full_command)
