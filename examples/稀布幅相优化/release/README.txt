============================================================
  稀布幅相优化 — 独立可执行程序 (Release)
============================================================

用法:
  1. 修改 Config.json 设置参数
  2. 双击 sparse_optimizer.exe
  3. 优化结果保存在 result/ 目录下（JSON + 方向图弹窗）

----------------------------
Config.json 字段说明
----------------------------

{
  "frequenciesGHz": [10.6],        // 工作频率 (GHz), 数组支持多频
  "randomSeed": 0,                 // 随机种子, 0=随机生成

  "aspectAngle": {
    "thetaStartDeg": -90,          // θ 扫描起始角
    "thetaEndDeg": 90,             // θ 扫描终止角
    "thetaStepDeg": 0.1,           // θ 采样步长
    "theta0sDeg": [0]              // 波束指向角 (度), 数组支持多角度
  },

  "target": {
    "mode": 201,                   // 优化模式 (3位编码)
                                   //   百位=位置: 1=优化, 2=导入
                                   //   十位=相位: 0=全0, 1=优化, 2=导入
                                   //   个位=幅度: 0=全1, 1=优化, 2=导入
                                   //   例: 100=纯稀布, 111=全优化, 201=导入位+优化幅
    "amplitudeBounds": [0.0, 1.0], // 幅度上下界 (仅 mode 个位=1 时有效)
    "targetHPBW": null,            // 目标半功率波束宽度, null=不限
    "mainLobePointingPenalty": true, // 是否惩罚主瓣指向偏差
    "import": {                    // 导入文件 (仅对应 mode 位为导入时有效)
      "positionsFile": "input/array_config/uniform_526.json",
      "phasesFile": null,          // null=不导入, 支持绝对/相对路径
      "amplitudesFile": null       // JSON格式: {"xCenters":[...], "phasesDeg":[...], "amplitudes":[...]}
    }
  },

  "ePattern": {                    // 单元方向图 (对应 C++ readFeMultiFreqFromCsvs)
    "enabled": false,              // true=启用, false=使用各向同性单元
    "csvDirectory": "input/element_pattern/kty",
    "degStep": 0.1,                // CSV 原始采样步长
    "isGain": true,                // true=增益值, false=场量
    "inDB": true,                  // true=dB值, false=线性值
    "thetaRange": [-90, 90]        // CSV 覆盖的 θ 范围 (度)
  },

  "antennaArray": {                // 仅 mode 百位=1(优化位置) 时需要
    "L_wavelength": 73,            // 孔径 (λ)
    "dmin_wavelength": 0.5,        // 最小间距 (λ)
    "dmax_wavelength": null,       // 最大间距, null/<=0=不限
    "Ne": 78,                      // 阵元数
    "isSymmetryArray": true,       // 是否对称阵列
    "isFixedAperture": true        // 是否固定孔径 (一般不开启)
  },

  "optimizer": {
    "method": "cma",               // 优化算法: cma | ngopt | cma_ng | de
    "cma":    { "pop_size": 100, "max_iter": 1000, "sigma": 1.0, "n_jobs": 0, "verbose": true, "stopFitness": null },
    "ngopt":  { "pop_size": 50,  "max_iter": 10000,                    "n_jobs": 0, "verbose": false, "stopFitness": null },
    "cma_ng": { "pop_size": 100, "max_iter": 300,                      "n_jobs": 0, "verbose": false, "stopFitness": null },
    "de":     { "pop_size": 50,  "max_iter": 10000,                    "n_jobs": 0, "verbose": false, "stopFitness": null }
    // pop_size: 种群大小, max_iter: 最大迭代, sigma: 初始步长
    // n_jobs: 并行线程 (-1=全部CPU, 0/1=单线程)
    // verbose: 打印迭代进度, stopFitness: 达到此值自动停止 (null=不停止)
  }
}

----------------------------
input/ 文件格式
----------------------------

array_config/*.json  —  导入的阵元配置:
  {
    "xCenters": [-4.2, -3.6, ..., 4.2],   // 绝对位置 (米), 主频转换
    "phasesDeg": [0, 0, ...],             // 相位 (度)
    "amplitudes": [1.0, 1.0, ...]         // 幅度 (线性)
  }

element_pattern/*.csv  —  单元方向图 (HFSS 导出):
  - 每行一个值 (dB增益或线性值, 由 isGain/inDB 控制)
  - θ 范围由 thetaRange 指定, 步长由 degStep 指定
  - 文件名: eGain_{freq}GHz.csv 或 {freq}GHz_*.csv
  - 多列文件可用 phiIdx 读取指定列 (默认第2列=phi=0)

----------------------------
result/ 输出格式
----------------------------

{时间戳}.json:
  {
    "settings": { ...Config.json... },
    "result": {
      "bestFitness": -25.14,
      "bestPositions": [...],
      "bestPhasesDeg": [...],
      "bestAmplitudes": [...],
      "optimizer": "cma",
      "elapsedSeconds": 28.9
    },
    "pattern": {
      "frequenciesGHz": [10.6],
      "theta0sDeg": [0],
      "thetaDeg": [-90, -89.9, ...],
      "value": [...]         // Fe*|AF| 实值 (未dB未归一化)
    }
  }
