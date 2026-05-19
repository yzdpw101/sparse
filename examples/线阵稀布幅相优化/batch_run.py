"""批量零陷优化 — 4频率 × 7角度 × 2phi面 × 3重复。

用法: cd examples/线阵稀布幅相优化2 && python batch_run.py
"""
import json, subprocess, sys, shutil, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "Config.json"

# 基准配置模板
BASE_CONFIG = {
    "$schema": "./Config.schema.json",
    "randomSeed": None,
    "aspectAngle": {"thetaDeg": [-90, 90, 0.1], "theta0sDeg": [0]},
    "amplitude": {"source": "optimize", "bounds": [0.0, 1.0], "optimizeHalf": True},
    "phase": {"source": "default", "optimizeHalf": True},
    "target": {
        "type": "null",
        "nullAngleDeg": 20,
        "nullTargetDb": -40,
        "sllTargetDb": -15,
        "nullWindowHalfDeg": 5,
        "nullMarginDb": 0,
        "weights": {"pointing": 0.5, "null_depth": 3, "valley": 1, "sll": 5},
    },
    "optimizer": {
        "sigma": 0.5,
        "pop_size": None,
        "max_iter": 1000,
        "n_jobs": -1,
        "verbose": True,
        "stopFitness": 0.0,
        "resume": False,
    },
}

FREQUENCIES = [9, 9.8, 10.6, 11.4]
NULL_ANGLES = [20, 30, 40, 50, 60, 70, 80]
PHI_PLANES = {
    "phi0": {
        "position_source": "input/array_config/uniform_526.json",
        "csv_dir": "input/element_pattern/kty/phi0",
        "theta_step": 0.01,
    },
    "phi90": {
        "position_source": "input/array_config/uniform_126.json",
        "csv_dir": "input/element_pattern/kty/phi90",
        "theta_step": 0.1,
    },
}
N_REPEAT = 3


def make_config(freq, phi, null_ang):
    """构建指定组合的 Config dict。"""
    phi_cfg = PHI_PLANES[phi]
    cfg = dict(BASE_CONFIG)
    cfg["frequenciesGHz"] = [freq]
    cfg["aspectAngle"] = dict(BASE_CONFIG["aspectAngle"])
    cfg["aspectAngle"]["thetaDeg"] = [-90, 90, phi_cfg["theta_step"]]
    cfg["position"] = {
        "source": phi_cfg["position_source"],
        "symmetric": False,
        "fixedAperture": True,
        "Ne": 17,
        "L_wavelength": 9.875,
        "dmin_wavelength": 0.5,
    }
    cfg["target"] = dict(BASE_CONFIG["target"])
    cfg["target"]["nullAngleDeg"] = null_ang
    cfg["ePattern"] = {
        "enabled": True,
        "csvDirectory": phi_cfg["csv_dir"],
        "thetaDeg": [-90, 90, phi_cfg["theta_step"]],
        "isGain": True,
        "inDB": True,
    }
    return cfg


def run_one(freq, phi, null_ang, repeat_idx):
    """运行单次优化并重命名结果目录。"""
    cfg = make_config(freq, phi, null_ang)

    # 写入 Config.json
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # 运行 main.py
    print(f"  [{freq}GHz, {phi}, null={null_ang}°, run {repeat_idx+1}/{N_REPEAT}] 开始...")
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(HERE / "main.py")],
        cwd=str(HERE),
        capture_output=True,
        text=True,
        timeout=3600,  # 1小时超时
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  !! 失败 (exit {result.returncode}): {result.stderr[-200:]}")
        return False

    # 找到最新创建的 result 子目录并重命名
    result_dir = HERE / "result"
    subdirs = sorted(
        [d for d in result_dir.iterdir() if d.is_dir() and d.name.startswith("20")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not subdirs:
        print(f"  !! 未找到结果目录")
        return False

    latest = subdirs[0]
    phi_num = "0" if phi == "phi0" else "90"
    new_name = f"{freq}-{phi_num}-{null_ang}-{repeat_idx+1}"
    target = result_dir / new_name
    if target.exists():
        shutil.rmtree(target)
    latest.rename(target)
    print(f"  完成: {target.name} ({elapsed:.0f}s)")
    return True


def main():
    total = len(FREQUENCIES) * len(NULL_ANGLES) * len(PHI_PLANES) * N_REPEAT
    print(f"=== 批量零陷优化 ===")
    print(f"  {len(FREQUENCIES)} 频率 × {len(NULL_ANGLES)} 角度 × {len(PHI_PLANES)} phi面 × {N_REPEAT} 重复 = {total} 组")
    print()

    done = 0
    failed = []
    for phi in PHI_PLANES:
        for freq in FREQUENCIES:
            for null_ang in NULL_ANGLES:
                for rep in range(N_REPEAT):
                    done += 1
                    print(f"[{done}/{total}]", end="")
                    ok = run_one(freq, phi, null_ang, rep)
                    if not ok:
                        failed.append((freq, phi, null_ang, rep))

    print(f"\n=== 完成: {done - len(failed)}/{total} 成功 ===")
    if failed:
        print(f"失败组: {failed}")


if __name__ == "__main__":
    main()
