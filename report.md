# 项目报告
*张灏宇 20260128*

## 初步数据分析
- 数据格式：每个 `data_matrix.csv` `id`.
- 数据特征:
  - `id`, `DateTime`, `industry`, `weight`, `y`
  - `f_0~f_9`：正态化处理后的基本面特征（可以直接用于模型输入）
  - `beta`, `indbeta`
  - `r_0~r_19`, `dv_0~dv_19` (past 30-min returns and turnover)
- Missing values exist and should be handled in modeling.


## 数据预处理 & 特征工程
