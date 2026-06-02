# 添加新融合方法指南

## 步骤

### 1. 复制模板

```bash
cp -r src/models/methods/template src/models/methods/your_method
```

### 2. 实现模型

编辑 `src/models/methods/your_method/model.py`：

```python
from src.models.registry import register_model
from src.models.base_model import BaseFusionModel

@register_model("your_method")
class YourMethod(BaseFusionModel):
    def __init__(self, config):
        super().__init__(config)
        # 构建网络...

    def forward(self, lr_hsi, hr_msi):
        # 实现融合逻辑
        return hr_hsi
```

### 3. 配置参数

在 `config/method/` 下创建 `your_method.yaml`：

```yaml
method:
  name: "your_method"

model:
  n_hsi_bands: 31
  n_msi_bands: 3
  # 自定义参数...

training:
  lr: 1e-4
  epochs: 200
```

### 4. 注册方法

编辑 `src/models/methods/__init__.py`，添加：

```python
from . import your_method
```

### 5. 验证

```python
from src.models import ModelRegistry
assert "your_method" in ModelRegistry.list_models()
```

## 接口要求

| 方法 | 说明 |
|------|------|
| `forward(lr_hsi, hr_msi)` | 必须实现，返回 HR-HSI |
| `get_loss(pred, target, lr_hsi, hr_msi)` | 可选，默认 L1 + SAM |
| `get_optimizer_params()` | 可选，分组 LR |

## 数据约定

| 变量 | 形状 | 范围 |
|------|------|------|
| `lr_hsi` | (B, C_hsi, H/s, W/s) | [0, 1] |
| `hr_msi` | (B, C_msi, H, W) | [0, 1] |
| `hr_hsi` (输出) | (B, C_hsi, H, W) | [0, 1] |
