"""4元阵 AEP 仿真"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
from antopt.utils import load_array_config
from hfss import run_aep_simulation

HERE = Path(__file__).resolve().parent
freq_ghz = 2.45
lam0 = 299792458.0 / (freq_ghz * 1e9)

x_m = np.array(load_array_config(
    "input/array_config/uniform_4_0p5wl_2.45Ghz.json", "xCenters", HERE))
Ne = len(x_m)
pos_wl = x_m / lam0
arr_len_wl = 2 * np.max(np.abs(pos_wl))

print(f"=== {Ne}元阵 AEP ===")
print(f"  arr_len_wl = {arr_len_wl:.2f}")

from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = HERE / "result" / f"aep_{Ne}elem_{ts}"
out_dir.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
csv_paths = run_aep_simulation(
    x_centers=x_m.tolist(),
    frequency_ghz=freq_ghz,
    results_dir=str(out_dir / "aep_csv"),
    close_after=False,
    array_width_wl=0.0,
    array_length_wl=arr_len_wl,
    array_margin_wl=0.5,
    airbox_margin_wl=0.25,
    theta_start=-90, theta_stop=90, theta_step=0.1,
    phi_start=0, phi_stop=0, phi_step=1.0,
    results_phi_sections=[0],
)
elapsed = time.perf_counter() - t0
print(f"Done. {len(csv_paths)} CSVs, {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Dir: {out_dir / 'aep_csv'}")
