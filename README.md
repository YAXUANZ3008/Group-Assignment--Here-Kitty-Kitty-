# Group-Assignment--Here-Kitty-Kitty-
CDS524 - Group Assignment -- Here, Kitty Kitty!
# 小组课程项目 README
# 小组课程项目 README

## 1. 项目简介
本项目围绕“蔬果新鲜度与类别识别”任务开展，整体流程包括：
1. 数据探索与可视化（EDA）
2. 数据预处理与划分（生成训练/验证/测试集）
3. 模型训练与评估（ResNet18 与自建双任务 CNN）
4. 可解释性与对比实验（Grad-CAM、混淆矩阵、消融实验）

当前仓库中已包含我们阶段性代码、实验结果截图以及报告相关材料。

## 1.1 模型权重下载（GitHub Releases）
- 下载入口：https://github.com/YAXUANZ3008/Group-Assignment--Here-Kitty-Kitty-/releases/tag/v1.0.0
- 需要下载的文件：
  - `resnet_freeze_best.pth`
  - `resnet_finetune_best.pth`
  - `self_built_cnn_best.pth`
- 建议下载后放到：`04_Model_Weights/`


## 2. 目录结构说明
```text
groupass/
├─ 第三第四章/
│  ├─ 01_EDA.py
│  └─ eda_results/
│     ├─ vegetable_statistics.csv
│     ├─ 01_distribution_bar.png
│     ├─ 02_sample_grid.png
│     ├─ 03_rgb_histogram.png
│     └─ 04_blur_analysis.png
│
├─ 02_Preprocessing/
│  ├─ 02_Preprocessing.py
│  ├─ create_dataset.py
│  ├─ train_data.csv
│  ├─ val_data.csv
│  ├─ test_data.csv
│  └─ mean_std.npy
│
├─ part6-8/03_Model_Training&Evaluation/
│  ├─ 03_Modeling_Training&Evaluation.ipynb
│  ├─ notebook_extracted.py
│  ├─ resnet_freeze_best.pth
│  ├─ resnet_finetune_best.pth
│  ├─ self_built_cnn_best.pth
│  ├─ *收敛曲线/混淆矩阵/GradCAM/消融实验结果图*
│
└─ all_code_merged.py          # 本次整理的“所有代码合并版”
```

---

## 3. 运行环境
建议环境（Windows / Linux 均可）：
- Python 3.9 ~ 3.11（本机也可在 3.14 下运行部分流程）
- PyTorch（含 torchvision）
- 其余依赖：
  - numpy
  - pandas
  - opencv-python
  - matplotlib
  - seaborn
  - scikit-learn
  - jupyter

安装示例：
```bash
pip install numpy pandas opencv-python matplotlib seaborn scikit-learn jupyter
# PyTorch 请按官网对应 CUDA/CPU 版本安装
```

---

## 4. 数据准备说明
本项目脚本中路径是按我们本地目录写的，运行前需要检查并按自己电脑路径修改。

重点关注：
- `第三第四章/01_EDA.py` 中的 `DATASET_PATH`
- `02_Preprocessing/02_Preprocessing.py` 中的 `DATASET_PATH`
- `02_Preprocessing/create_dataset.py` 中的 `BASE_PATH`

原始数据目录建议结构（示例）：
```text
Dataset/
├─ Fresh/
│  ├─ FreshCarrot/
│  ├─ FreshTomato/
│  └─ ...
└─ Rotten/
   ├─ RottenCarrot/
   ├─ RottenTomato/
   └─ ...
```

---

## 5. 代码运行步骤（建议顺序）

### Step 1：运行 EDA（统计与可视化）
```bash
cd 第三第四章
python 01_EDA.py
```
输出结果在：`第三第四章/eda_results/`

### Step 2：运行预处理（清洗、划分、均值方差）
```bash
cd 02_Preprocessing
python 02_Preprocessing.py
```
输出：`train_data.csv`、`val_data.csv`、`test_data.csv`、`mean_std.npy`

### Step 3：按 CSV 组织训练目录（可选）
```bash
cd 02_Preprocessing
python create_dataset.py
```
该步骤会根据 CSV 将图片复制到 train/val/test 目录结构中。

### Step 4：模型训练与评估
```bash
cd part6-8/03_Model_Training&Evaluation
jupyter notebook
```
打开 `03_Modeling_Training&Evaluation.ipynb` 按单元顺序运行。

说明：
- 训练好的权重文件（`.pth`）请从 Releases 下载后放入 `04_Model_Weights/`。
- 如果只需查看代码，也可打开 `notebook_extracted.py`。

---

## 6. 主要实验结果（仓库已包含）

### EDA结果
- `第三第四章/eda_results/01_distribution_bar.png`
- `第三第四章/eda_results/02_sample_grid.png`
- `第三第四章/eda_results/03_rgb_histogram.png`
- `第三第四章/eda_results/04_blur_analysis.png`

### 模型训练与评估结果
- `part6-8/03_Model_Training&Evaluation/Self-built Dual CNN_Convergence_Curve.png`
- `part6-8/03_Model_Training&Evaluation/ResNet18 Fine-tuned_Convergence_Curve.png`
- `part6-8/03_Model_Training&Evaluation/ResNet18_Vegetable_Classification_Confusion_Matrix_(Test_Set).png`
- `part6-8/03_Model_Training&Evaluation/ResNet18_Freshness_Classification_Confusion_Matrix_(Test_Set).png`
- `part6-8/03_Model_Training&Evaluation/GradCAM_Visualization.png`
- `part6-8/03_Model_Training&Evaluation/GradCAM_Vegetable_Task.png`
- `part6-8/03_Model_Training&Evaluation/GradCAM_Freshness_Task.png`
- `part6-8/03_Model_Training&Evaluation/Misclassified_Samples_Visualization.png`
- `part6-8/03_Model_Training&Evaluation/Ablation_Experiment_Result.png`

---

## 7. 合并代码文件说明
本仓库已新增：
- `all_code_merged.py`

该文件把以下代码按顺序拼接在一个文件中：
1. `第三第四章/01_EDA.py`
2. `02_Preprocessing/02_Preprocessing.py`
3. `02_Preprocessing/create_dataset.py`
4. `part6-8/03_Model_Training&Evaluation/notebook_extracted.py`

说明：
- 合并文件主要用于“统一提交与查看”，尽量保持原代码内容不变。
- 因为多个脚本都带有各自执行入口和路径配置，实际运行时仍建议按原目录分别运行。

---

## 8. 备注
- 如果运行时报路径错误，优先检查脚本开头的路径变量是否改成自己电脑上的绝对/相对路径。
- 若显存不足，可在 Notebook 中调小 `batch_size`。
- 若只做结果复现，可优先加载已有 `.pth` 权重并直接执行评估与可视化部分。
