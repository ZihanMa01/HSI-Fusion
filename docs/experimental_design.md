# 实验设计指南

## 标准实验协议

### 1. 数据准备

遵循 Wald's protocol：
- **训练**: 从 HR-HSI 合成 LR-HSI (PSF+下采样) 和 HR-MSI (SRF 积分)
- **测试**: 使用真实 LR-HSI + HR-MSI 进行推理

### 2. 评估维度

| 维度 | 指标 | 说明 |
|------|------|------|
| 光谱保真度 | SAM ↓ | 光谱角越小越好 |
| 空间质量 | PSNR ↑, SSIM ↑ | 峰值信噪比/结构相似性 |
| 全局质量 | ERGAS ↓, UIQI ↑ | 全局相对误差/通用质量 |
| 无参考 | HQNR ↑ | 无需 GT 的混合质量 |
| 效率 | Params, FLOPs, Runtime | 模型复杂度与速度 |

### 3. 对比实验

```
Method A (Ours)  vs  Method B (SOTA)  vs  Baseline
```

### 4. 消融实验

```
Full Model  vs  w/o Module A  vs  w/o Module B
```

### 5. 参数敏感性

测试关键超参数的影响范围。

## 结果报告模板

```markdown
| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | Params |
|--------|--------|--------|-------|---------|--------|
| Ours   | 38.5   | 0.982  | 3.2   | 1.8     | 0.5M   |
| SOTA   | 37.8   | 0.978  | 3.8   | 2.1     | 1.2M   |
| Baseline| 35.2  | 0.965  | 4.5   | 2.8     | 0.3M   |
```
