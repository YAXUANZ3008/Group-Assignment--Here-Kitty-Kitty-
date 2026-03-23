# 数据集说明（Dataset Info）

## 1. 数据来源与代码入口
本项目使用蔬果图像数据集，目录按 `Fresh` / `Rotten` 组织。
项目代码中对数据集命名进行了解析与兼容（如部分类别拼写变体）。

当前仓库中，与数据集路径直接相关的脚本位置为：
- `02_Code/01_EDA/01_EDA.py`（`DATASET_PATH`）
- `02_Code/02_Preprocessing/02_Preprocessing.py`（`DATASET_PATH`）
- `02_Code/02_Preprocessing/create_dataset.py`（`BASE_PATH`）

## 2. 原始数据目录建议
建议原始数据按如下结构组织：

```text
Dataset/
├─ Fresh/
│  ├─ FreshBellpepper/
│  ├─ FreshBittergroud/
│  ├─ FreshCapsicum(or FreshCapciscum)/
│  ├─ FreshCarrot/
│  ├─ FreshCucumber/
│  ├─ FreshOkara(or FreshOkra)/
│  ├─ FreshPotato/
│  └─ FreshTomato/
└─ Rotten/
   ├─ RottenBellpepper/
   ├─ RottenBittergroud/
   ├─ RottenCapsicum/
   ├─ RottenCarrot/
   ├─ RottenCucumber/
   ├─ RottenOkra/
   ├─ RottenPotato/
   └─ RottenTomato/
```

建议将该原始数据目录放在你本机可访问位置，并在上述脚本中将路径变量改为本机实际路径。

## 3. 预处理输出文件
项目已提供以下预处理产物：
- `train_data.csv`
- `val_data.csv`
- `test_data.csv`
- `mean_std.npy`

这些文件已整理在：`03_Results/02_Preprocessing_Outputs/`。
对应生成脚本：`02_Code/02_Preprocessing/02_Preprocessing.py`。

## 4. 标签字段说明
- `freshness_label`: 0 表示 Fresh，1 表示 Rotten
- `vegetable_label`: 按脚本中的类别映射编码

## 5. 数据集链接
- Kaggle: https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset
