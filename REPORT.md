# 量化项目报告（唯一指标：Daily-Weighted-Mean IC）

## 唯一评估指标（Daily-Weighted-Mean IC）
对每个交易日 *t*：

- **日内 IC**：
  \[ IC_t = weighted\_corr(y_t, \hat{y}_t, w_t) \]

- **日权重**：
  \[ W_t = \sum_i w_{t,i} \]

- **最终评分（唯一指标）**：
  \[ SCORE = \frac{\sum_t W_t \cdot IC_t}{\sum_t W_t} \]

**说明**：该 SCORE 是唯一指标，用于模型选择、调参、消融、特征重要性、`metrics.csv` 和全部图表。

---

## 1. 初步数据分析（特征类型、经济学含义、权重weight解释、基本统计图、缺失情况、y分布、简单单因子IC sanity check）
- **特征类型**：
  - `r_0..r_19`：半小时收益，反映短期动量/反转
  - `dv_0..dv_19`：成交额，反映流动性与冲击
  - `f_0..f_9`：慢变量基本面
  - `industry/beta/indbeta`：行业与风险暴露
- **权重解释**：`weight` 同时用于训练与评估中的加权相关，强调重要样本。
- **基本统计与缺失**：见 `res/data-statics/`（分布图、缺失率、`metadata.csv`）。
- **sanity check**：用简单单因子（如短期动量）计算日内 IC，确认方向合理。

**改进模板**：
“发现/证据 → 假设 → 改进方法 → 经济学意义 → 实验结果（引用 `res/experiments/{RUN_NAME}`）→ 小结”。

---

## 2. 数据预处理 & 特征工程（winsorize_by_date、按日标准化、标签处理y_csz/y_rankgauss的动机；构建mom_30m等多窗口特征及其经济学意义）
- **winsorize_by_date（非加权分位数）**：按日裁剪极端值，降低异常冲击。
- **按日标准化**：可选，用于降低尺度差异（非主实验轴）。
- **标签处理动机**：
  - `y_csz`/`rank-gauss` 有助于缓解分布偏态、提升稳定性（作为背景动机说明）。
- **当前仅保留 3 种标签模式（严格）**：
  - **raw**：`y_train = y_raw`，`y_score = y_raw`
  - **winsor_csz**：`y_train = cs_zscore_by_date(winsorize_by_date(y_raw))`，`y_score` 同 `y_train`
  - **neu_winsor_csz**：
    1) **label 中性化**（按日 WLS，行业+beta+indbeta）得到残差 `y_neu`
    2) `y_train = cs_zscore_by_date(winsorize_by_date(y_neu))`，`y_score` 同 `y_train`

**改进模板**：
“发现/证据 → 假设 → 改进方法 → 经济学意义 → 实验结果 → 小结”。

---

## 3. 初步模型构建（统一训练设置；对比 ridge/elasticnet/rf/extra_trees/lgbm/MLP；唯一指标= daily-weighted-mean IC）
- 模型：ridge / elasticnet / rf / extra_trees / lgbm / torch_mlp
- 训练使用样本权重；Torch 使用加权 MSE：
  \[ \mathcal{L} = \frac{\sum_i w_i (\hat{y}_i - y_i)^2}{\sum_i w_i} \]
- **唯一指标**：Daily-Weighted-Mean IC
- 结果查看：`res/experiments/{RUN_NAME}/metrics.csv`

**改进模板**：
“发现/证据 → 假设 → 改进方法 → 经济学意义 → 实验结果 → 小结”。

---

## 4. 特征重要性分析&消融实验（best model；gain/permutation；group-only和drop-one；解释Top特征的经济学意义）
- Best model 由 valid_score 选择
- Gain/Permutation importance
- Group-only 与 Drop-one 消融

**改进模板**：
“发现/证据 → 假设 → 改进方法 → 经济学意义 → 实验结果 → 小结”。

---

## 5. 深度学习方法（MLP基线 -> 1D-CNN/TCN -> RNN -> Transformer；同样做消融与指标对齐；说明为何先CNN后Transformer）
- 统一使用唯一指标 SCORE
- 先 CNN/TCN 捕捉局部结构，再考虑 Transformer 的全局依赖
- **推荐默认超参数（稳定基线）**：
  - `lr=3e-4`, `batch=2048`, `weight_decay=1e-4`, `dropout=0.2`, `grad_clip=1.0`
  - `epochs=30`, `early_stop_patience=6`, `ic_every=1`, `scheduler=plateau_ic`
  - `cnn_channels=[16,32]`, `kernel=3`, `rnn_hidden=32`

---

## 6. 不同数据集划分方式的影响（simple vs forward；解释为何forward更能反映regime shift）
- **simple**：2016-2018 训练，2019 验证，2020 测试
- **forward**：滚动训练，模拟 regime shift

---

## 7. 中性化/缺失值处理等（pred_neutralize_by_date、neutral_then_z顺序、行业/风险暴露剥离的意义；缺失指示/填补策略）
- **label 中性化（核心定义）**：
  - 每日构造暴露矩阵 \(X_t = [1, industry\_onehot, beta, indbeta]\)
  - 加权最小二乘：\(\arg\min \sum_i w_i (y_i - X_i\theta)^2\)
  - 残差作为中性化后标签
  - **缺失处理**：缺失暴露的样本不参与回归，但残差取原 y（fallback）
- **预测中性化** 保留为可选开关（非主实验轴）。

---

## 8. 总结（最终方案、关键结论、下一步改进方向）
- 唯一指标：Daily-Weighted-Mean IC
- 关键结论：标签处理与中性化顺序显著影响稳定性
- 下一步：更丰富的多窗口特征、时序模型与风险约束

---

## How to run
1) 运行主 notebook：`src/notebooks/run_pipeline.ipynb`
2) 生成脚本：`scripts/{RUN_NAME}/run_all.sh`
3) 执行：`bash scripts/{RUN_NAME}/run_all.sh`

输出目录：
- `res/experiments/{RUN_NAME}/metrics.csv`
- `res/experiments/{RUN_NAME}/plots/`
- `res/experiments/{RUN_NAME}/ic_series/`
