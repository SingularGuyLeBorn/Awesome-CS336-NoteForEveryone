# 代码深度解析: PowerLawFitter (Scaling Law 拟合器)

### 1. 核心功能与目标 (Core Function & Goal)
`PowerLawFitter` 模块的主要目标是基于一系列小规模实验的观测数据（例如：不同数据量下的 Test Loss），拟合出 Scaling Law 的关键参数。这使得工程师能够外推（Extrapolate）并预测在更大规模下的模型表现，从而做出是否继续扩大训练规模的决策。

### 2. 参数解析 (Parameters)
主要处理的方程形式为：$L(x) = a x^{-k} + b$ (其中 $b$ 为不可约误差)。

*   `x_data`: 观测到的资源量数组（如数据集大小 tokens 数或模型参数量）。
*   `y_data`: 观测到的性能指标数组（通常是 Test Loss）。
*   `p0`: 拟合算法的初始参数猜测 `[a, k, b]`，用于辅助优化器收敛。

### 3. 核心逻辑 (Core Logic)

```python
import numpy as np
from scipy.optimize import curve_fit

class PowerLawFitter:
    """
    用于拟合缩放定律曲线 L(x) = a * x^(-k) + b 的工具类。
    """
    def __init__(self):
        pass

    def _power_law_func(self, x, a, k, b):
        """
        定义幂律函数模型。
        x: 输入变量 (如数据量或参数量)
        a: 缩放系数
        k: 幂律指数 (Slope in log-log without irreducible error)
        b: 不可约误差 (Irreducible Error)
        """
        # 加上 1e-9 防止除零错误，但在 log 空间通常处理
        return a * np.power(x, -k) + b

    def fit(self, x_data, y_data):
        """
        拟合数据并返回参数。
        """
        # 将输入转换为 numpy 数组
        x = np.array(x_data)
        y = np.array(y_data)

        # 初始猜测 [scale, exponent, irreducible_error]
        # exponent 通常在 0.05 到 0.5 之间
        # irreducible_error 通常略小于观察到的最小 Loss
        initial_guess = [10.0, 0.1, np.min(y) * 0.9]

        # 约束条件: a>0, k>0, b>=0 (Loss 不可能为负)
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

        try:
            # 使用非线性最小二乘法拟合
            popt, pcov = curve_fit(
                self._power_law_func,
                x,
                y,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            return {
                "a": popt[0],
                "k": popt[1], # 这是核心关注的斜率
                "b": popt[2]  # 这是预测的极限性能
            }
        except RuntimeError:
            print("拟合未能收敛。可能数据点太少或不在幂律区域内。")
            return None

    def predict(self, x_query, params):
        """
        基于拟合参数预测新的 x 对应的 Loss。
        """
        return self._power_law_func(x_query, params['a'], params['k'], params['b'])

# --- 使用示例 ---
# 假设我们训练了三个小模型，数据量分别为 1B, 10B, 100B
data_sizes = [1e9, 1e10, 1e100]
# 对应的 Loss
losses = [3.5, 2.8, 2.4]

fitter = PowerLawFitter()
params = fitter.fit(data_sizes, losses)

if params:
    # 预测使用 1T (1000B) 数据时的 Loss
    predicted_loss = fitter.predict(1e12, params)
    print(f"Predicted slope (k): {params['k']:.4f}")
    print(f"Irreducible Error (b): {params['b']:.4f}")
    print(f"Predicted Loss at 1T tokens: {predicted_loss:.4f}")
```

### 4. 与理论的连接 (Connection to Theory)
*   **幂律验证**: 代码中的 `np.power(x, -k)` 直接对应了课堂上讲到的 **[Power Law](./Lecture9-Power-Law.md)** 数学形式。
*   **不可约误差**: 参数 `b` 明确建模了 **[Irreducible Error](./Lecture9-Irreducible-Error.md)**。如果不包含 `b`，在大规模数据下预测值会趋向于 0，这在物理上是不可能的（因为熵的存在）。
*   **斜率含义**: 拟合出的 `k` 值（Slope）在不同任务中不同（如课堂提到的 LM 约为 0.095，翻译约为 0.13）。这个值可以通过代码计算出来，作为评估任务 **[Intrinsic Dimensionality](./Lecture9-Intrinsic-Dimensionality.md)** 的代理指标。