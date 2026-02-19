# 貝氏模型版本說明

## 現有版本

### 舊版本（原始/第一輪）
- 位置：`C:\Users\user\OneDrive - Lehigh University\Desktop\FLOODABM\modules\actions\models`
- 快取：`C:\Users\user\OneDrive - Lehigh University\Desktop\FLOODABM\modules\actions\models_fast`
- 特性：beta_tp ≈ 1.9，較溫和的 TP 敏感度

### 新版本（第二輪優化）⭐
- 位置：`C:\Users\user\OneDrive - Lehigh University\Desktop\FLOODABM\modules\actions\models_optimized`
- 快取：`C:\Users\user\OneDrive - Lehigh University\Desktop\FLOODABM\modules\actions\models_optimized_fast`
- 特性：beta_tp ≈ 2.9，顯著提升的 TP 敏感度

## 如何切換版本

### 使用新優化版本

修改 `bayes_fast_predictors.py` 或在您的主程序中：

```python
# 方法 1：修改 actions_root 路徑
actions_root = Path("modules/actions")
# 改為載入 models_optimized
predictor_MG, predictor_NMG = build_fast_predictors(
    actions_root / "models_optimized"  # 使用新版本
)

# 方法 2：直接指定路徑
from pathlib import Path
new_models_path = Path("modules/actions/models_optimized")
predictor_MG, predictor_NMG = build_fast_predictors(new_models_path)
```

### 回退到舊版本

```python
# 使用原始路徑
actions_root = Path("modules/actions")
predictor_MG, predictor_NMG = build_fast_predictors(
    actions_root / "models"  # 使用舊版本
)
```

## 對比測試建議

1. **基線測試**（舊版本）
   - 使用 `models/` 資料夾
   - 運行場景並記錄結果

2. **實驗測試**（新版本）
   - 切換到 `models_optimized/`
   - 清理快取：刪除 `models_optimized_fast/*.npz`
   - 運行相同場景

3. **對比分析**
   - 政策介入效果
   - Agent 行為分化
   - TP 敏感度

## 預期差異

| 指標 | 舊版本 | 新版本 |
|------|--------|--------|
| 高 TP Agent 採納率 | ~45% | ~70% |
| 政策介入效果 | +15% | +35% |
| TP 敏感度 | 0.38 | 0.59 |

## 注意事項

⚠️ **快取管理**：
- 每次切換版本後，刪除對應的 `*_fast/*.npz` 文件
- 否則會使用舊快取，看不到模型變化

⚠️ **程式碼修改**：
- 需要修改載入模型的程式碼以切換版本
- 或創建配置文件來管理版本

---
生成時間：2025-12-04 09:16:31
