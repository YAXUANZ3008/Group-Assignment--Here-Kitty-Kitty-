# 数据集说明（Dataset Info）

## 1. 数据来源说明
本项目使用蔬果图像数据集，目录按 `Fresh` / `Rotten` 组织。
项目代码中对数据集命名进行了解析与兼容（如部分类别拼写变体）。

## 2. 数据目录组织（原始）
建议原始数据结构如下：

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

## 3. 预处理产物
项目已提供预处理输出文件：
- `train_data.csv`
- `val_data.csv`
- `test_data.csv`
- `mean_std.npy`

这些文件已放在提交包的 `03_Results/02_Preprocessing_Outputs/` 目录下。

## 4. 标签说明
- `freshness_label`: 0 表示 Fresh，1 表示 Rotten
- `vegetable_label`: 按脚本中的类别映射编码

## 5. 数据集获取链接
- Kaggle: https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset
