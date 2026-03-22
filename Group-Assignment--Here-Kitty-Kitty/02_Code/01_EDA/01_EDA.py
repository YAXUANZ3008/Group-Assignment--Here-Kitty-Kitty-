import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 配置信息
# ==========================================
DATASET_PATH = './archive/Dataset'
OUTPUT_DIR = './eda_results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 核心蔬菜关键字清单 (用于过滤和匹配)
# 代码会匹配文件夹名中包含这些词的部分，如 "FreshCarrot" 匹配 "carrot"
VEGETABLE_KEYS = [
    'bellpepper', 'bellpeper', 'bittergroud', 'capciscum',
    'capsicum', 'carrot', 'cucumber', 'okara', 'okra',
    'potato', 'tomato'
]


# ==========================================
# 2. 数据采集逻辑 (精准适配双重前缀)
# ==========================================
def collect_stats_fixed(root_path, target_keys):
    data = []

    # 遍历顶级目录: ['Fresh', 'Rotten']
    for status in ['Fresh', 'Rotten']:
        status_path = os.path.join(root_path, status)
        if not os.path.exists(status_path):
            print(f"⚠️ 跳过缺失路径: {status_path}")
            continue

        # 遍历次级目录: ['FreshCarrot', 'FreshApple', ...]
        sub_folders = os.listdir(status_path)
        for folder in sub_folders:
            folder_lower = folder.lower()

            # 过滤逻辑：判断该子文件夹名是否包含蔬菜关键字
            matched_veg = None
            for key in target_keys:
                if key in folder_lower:
                    # 归一化处理：解决拼写变体
                    if key in ['okara', 'okra']:
                        matched_veg = 'Okra'
                    elif key in ['capciscum', 'capsicum']:
                        matched_veg = 'Capsicum'
                    elif key in ['bellpepper', 'bellpeper']:
                        matched_veg = 'Bellpepper'
                    else:
                        matched_veg = key.capitalize()
                    break

            # 如果匹配到蔬菜关键字，则记录数据
            if matched_veg:
                img_dir = os.path.join(status_path, folder)
                images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                if len(images) > 0:
                    data.append({
                        'Category': matched_veg,
                        'Status': status,  # 使用顶级目录名作为状态
                        'Count': len(images),
                        'Path': img_dir,
                        'Image_Files': images
                    })

    return pd.DataFrame(data)


df_stats = collect_stats_fixed(DATASET_PATH, VEGETABLE_KEYS)

if df_stats.empty:
    print("❌ 错误：未识别到任何蔬菜样本。请检查 DATASET_PATH 是否正确。")
    exit()

# 导出统计文件
df_summary = df_stats.pivot_table(index='Category', columns='Status', values='Count', fill_value=0).reset_index()
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'vegetable_statistics.csv'), index=False)
print(f"✅ 成功提取 {len(df_stats['Category'].unique())} 种蔬菜数据。")

# ==========================================
# 3. 标准化图表输出
# ==========================================

# --- 3.1 样本分布图 ---
plt.figure(figsize=(12, 6))
sns.barplot(data=df_stats, x='Category', y='Count', hue='Status', palette='mako')
plt.title('Vegetable Sample Distribution (Cleaned Data)', fontsize=14)
plt.savefig(os.path.join(OUTPUT_DIR, '01_distribution_bar.png'), dpi=300, bbox_inches='tight')
plt.close()


# --- 3.2 样本网格可视化图 (按要求的网格展示) ---
def plot_grid(df):
    categories = sorted(df['Category'].unique())
    n_rows = len(categories)
    fig, axes = plt.subplots(n_rows, 6, figsize=(18, 3 * n_rows))

    for i, cat in enumerate(categories):
        # 依次取 Fresh 和 Rotten 的样本
        for j, status in enumerate(['Fresh', 'Rotten']):
            subset = df[(df['Category'] == cat) & (df['Status'] == status)]
            if subset.empty: continue

            folder = subset['Path'].values[0]
            img_list = subset['Image_Files'].values[0]
            selected = random.sample(img_list, min(3, len(img_list)))

            for k, img_name in enumerate(selected):
                col_idx = j * 3 + k
                img = cv2.imread(os.path.join(folder, img_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i, col_idx].imshow(img)
                axes[i, col_idx].axis('off')
                if i == 0: axes[i, col_idx].set_title(f"{status} {k + 1}")

        # 侧边添加品类标签
        axes[i, 0].set_ylabel(cat, rotation=0, size='large', labelpad=50, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_sample_grid.png'), dpi=300)
    plt.close()


plot_grid(df_stats)


# --- 3.3 RGB直方图 ---
def plot_rgb(df):
    sample_veg = df['Category'].iloc[0]
    plt.figure(figsize=(15, 5))
    for i, status in enumerate(['Fresh', 'Rotten']):
        subset = df[(df['Category'] == sample_veg) & (df['Status'] == status)]
        img_name = random.choice(subset['Image_Files'].values[0])
        img = cv2.imread(os.path.join(subset['Path'].values[0], img_name))
        plt.subplot(1, 2, i + 1)
        for j, col in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([img], [j], None, [256], [0, 256])
            plt.plot(hist, color=col, linewidth=2)
        plt.title(f'{sample_veg} ({status}) RGB Profile')
    plt.savefig(os.path.join(OUTPUT_DIR, '03_rgb_histogram.png'), dpi=300)
    plt.close()


plot_rgb(df_stats)


# --- 3.4 模糊检测图 ---
def plot_blur(df):
    scores = []
    for folder in df['Path']:
        imgs = random.sample(os.listdir(folder), min(10, len(os.listdir(folder))))
        for n in imgs:
            img = cv2.imread(os.path.join(folder, n), cv2.COLOR_BGR2GRAY)
            if img is not None:
                scores.append(cv2.Laplacian(img, cv2.CV_64F).var())

    plt.figure(figsize=(10, 5))
    sns.histplot(scores, kde=True, color='teal')
    plt.axvline(100, color='red', linestyle='--')
    plt.title('Image Sharpness Analysis')
    plt.savefig(os.path.join(OUTPUT_DIR, '04_blur_analysis.png'), dpi=300)
    plt.close()


plot_blur(df_stats)

print(f"🚀 数据处理与图表生成已全部完成！文件位于: {OUTPUT_DIR}")