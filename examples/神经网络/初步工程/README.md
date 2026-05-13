# 神经网络代理模型 — 初步工程

用 PyTorch 神经网络学习 `位置 → PSLL` 的映射。

## 文件

- `train.py` — 训练脚本
- `predict.py` — 预测脚本
- `model.py` — 网络定义

## 用法

```bash
# 训练 (生成 model.pt)
python train.py

# 预测
python predict.py --positions "0.1,0.2,..."
```
