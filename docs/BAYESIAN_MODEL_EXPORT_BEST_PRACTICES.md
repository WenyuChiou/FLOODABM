# Bayesian Model Export Best Practices
## 避免 Pickle 模組依賴問題

---

## 問題描述

當使用 `dill.dump()` 或 `pickle.dump()` 儲存包含自定義類別的模型時，pickle 會記錄類別的完整模組路徑（如 `bayesian_engine.model.BayesianModel`）。這導致：

1. **跨環境不相容**：在沒有安裝 `bayesian_engine` 的環境中無法載入 pkl 檔案
2. **版本依賴**：即使安裝了模組，版本差異也可能導致 `AttributeError`
3. **部署困難**：生產環境需要安裝訓練時的完整依賴

---

## 解決方案：同時儲存 NPZ 權重檔案

### 修改 `bayesian_engine/model.py` 的 `save()` 方法：

```python
import numpy as np
from pathlib import Path

class BayesianModel:
    def save(self, output_path: str | Path):
        """
        Save model with BOTH .pkl (full model) AND .npz (weights only).
        
        The .npz file is portable across environments.
        """
        output_path = Path(output_path)
        
        # 1. Save full model as pkl (for retraining/analysis)
        import dill
        with open(output_path, 'wb') as f:
            dill.dump(self, f)
        
        # 2. ALSO save weights as portable npz
        npz_path = output_path.with_suffix('.npz')
        self._export_weights_npz(npz_path)
        
        print(f"[OK] Saved: {output_path}")
        print(f"[OK] Saved: {npz_path} (portable weights)")
    
    def _export_weights_npz(self, npz_path: Path):
        """Export weights to portable numpy format."""
        ps = self.posterior_samples
        
        # Extract mean weights (shape: 3 for TP, CP, SP)
        weights = np.mean(np.asarray(ps.get('beta', ps.get('w', np.zeros(3)))), axis=0)
        
        # Extract mean bias
        bias = float(np.mean(np.asarray(ps.get('intercept', ps.get('alpha', [0.0])))))
        
        # Save as simple numpy arrays
        np.savez(
            npz_path,
            w=weights.astype(np.float32),
            b=np.float32(bias)
        )
```

---

## 訓練腳本範例

```python
# train.py
from bayesian_engine import BayesianModel

# Train model
model = BayesianModel()
model.fit(X_train, y_train)

# Save BOTH formats automatically
model.save("models/optimized/MG_FI.pkl")
# This creates:
#   - models/optimized/MG_FI.pkl  (full model, requires bayesian_engine)
#   - models/optimized/MG_FI.npz  (weights only, portable)
```

---

## 載入端 (FLOODABM)

```python
# 在 FLOODABM 中只需要載入 npz
def load_weights(model_path: Path):
    npz_path = model_path.with_suffix('.npz')
    if npz_path.exists():
        data = np.load(npz_path)
        return data['w'], float(data['b'])
    else:
        raise FileNotFoundError(f"No npz file found: {npz_path}")
```

---

## 重點整理

| 檔案格式 | 用途 | 可攜性 |
|----------|------|--------|
| `.pkl` | 完整模型（重新訓練、分析） | ❌ 需要原始環境 |
| `.npz` | 權重向量（推論用） | ✅ 任何 numpy 環境 |

**規則：每次 `model.save()` 都應該同時產生 `.pkl` 和 `.npz` 檔案。**

---

## 未來任務

- [ ] 修改 `bayesian_engine/model.py` 中的 `save()` 方法
- [ ] 重新訓練所有 8 個模型 (MG/NMG × FI/EH/BP/RL)
- [ ] 驗證 `.npz` 檔案可在 FLOODABM 環境正確載入
