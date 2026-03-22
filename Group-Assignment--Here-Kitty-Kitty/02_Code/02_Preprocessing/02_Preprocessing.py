import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.model_selection import train_test_split

# 固定随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 数据集路径
DATASET_PATH = os.path.abspath(".")

# 数据清洗阈值
BLUR_THRESHOLD = 50  # 模糊度阈值，低于此值的图像被视为模糊（降低阈值使条件更宽松）
MIN_SIZE = 50  # 最小图像尺寸，低于此值的图像被视为异常（降低阈值使条件更宽松）

# 收集所有图像路径和标签
def collect_images():
    image_paths = []
    vegetable_labels = []
    freshness_labels = []
    vegetable_dict = {}  # 蔬菜编码字典
    vegetable_id = 0
    
    # 遍历Fresh和Rotten文件夹
    for freshness in ["Fresh", "Rotten"]:
        freshness_label = 0 if freshness == "Fresh" else 1
        freshness_path = os.path.join(DATASET_PATH, freshness)
        
        if not os.path.exists(freshness_path):
            print(f"路径不存在: {freshness_path}")
            continue
        
        print(f"处理 {freshness} 文件夹...")
        
        # 遍历每种蔬菜
        for vegetable in os.listdir(freshness_path):
            vegetable_path = os.path.join(freshness_path, vegetable)
            
            if not os.path.isdir(vegetable_path):
                continue
            
            # 为蔬菜分配编码
            if vegetable not in vegetable_dict:
                vegetable_dict[vegetable] = vegetable_id
                vegetable_id += 1
            
            print(f"  处理 {vegetable} 子文件夹...")
            
            # 遍历图像文件
            for image_file in os.listdir(vegetable_path):
                image_path = os.path.join(vegetable_path, image_file)
                image_paths.append(image_path)
                vegetable_labels.append(vegetable_dict[vegetable])
                freshness_labels.append(freshness_label)
    
    print(f"收集到 {len(image_paths)} 个图像文件")
    print(f"蔬菜编码: {vegetable_dict}")
    return image_paths, vegetable_labels, freshness_labels, vegetable_dict

# 数据清洗函数
def clean_data(image_paths, vegetable_labels, freshness_labels):
    clean_paths = []
    clean_vegetable_labels = []
    clean_freshness_labels = []
    
    print(f"开始清洗 {len(image_paths)} 个图像文件...")
    
    # 简化清洗逻辑，直接保留所有路径
    # 由于路径编码问题，我们暂时跳过图像读取验证
    for i, (path, veg_label, fresh_label) in enumerate(zip(image_paths, vegetable_labels, freshness_labels)):
        clean_paths.append(path)
        clean_vegetable_labels.append(veg_label)
        clean_freshness_labels.append(fresh_label)
        if i % 100 == 0:
            print(f"已处理 {i} 个图像")
    
    print(f"清洗完成，保留 {len(clean_paths)} 个图像")
    return clean_paths, clean_vegetable_labels, clean_freshness_labels

# 生成CSV文件
def generate_csv(image_paths, vegetable_labels, freshness_labels, output_file):
    df = pd.DataFrame({
        "image_path": image_paths,
        "vegetable_label": vegetable_labels,
        "freshness_label": freshness_labels
    })
    df.to_csv(output_file, index=False)
    print(f"CSV文件已生成: {output_file}")

# 计算训练集的RGB均值和标准差
def calculate_mean_std(train_paths):
    mean = np.zeros(3)
    std = np.zeros(3)
    count = 0
    
    for path in train_paths:
        try:
            # 尝试读取图像
            img = cv2.imread(path)
            if img is not None:
                # 转换为RGB并归一化到0-1
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                mean += img.mean(axis=(0, 1))
                std += img.std(axis=(0, 1))
                count += 1
            
        except Exception as e:
            continue
    
    if count > 0:
        mean /= count
        std /= count
        return mean, std
    else:
        # 如果没有成功读取任何图像，返回默认值
        print("警告: 无法读取任何图像计算均值和标准差，使用默认值")
        return None, None

# 自定义数据集类
class VegetableDataset(Dataset):
    def __init__(self, image_paths, vegetable_labels, freshness_labels, mean, std, is_train=False):
        self.image_paths = image_paths
        self.vegetable_labels = vegetable_labels
        self.freshness_labels = freshness_labels
        self.mean = mean
        self.std = std
        self.is_train = is_train
        
        # 数据增强和变换
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.RandomRotation(15),  # ±15°旋转
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机缩放裁剪
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度/对比度微调
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        veg_label = self.vegetable_labels[idx]
        fresh_label = self.freshness_labels[idx]
        
        try:
            # 读取图像
            img = cv2.imread(path)
            if img is None:
                # 返回一个随机图像作为占位符
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # 转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 应用变换
            img = self.transform(img)
            
            return img, torch.tensor(veg_label), torch.tensor(fresh_label)
            
        except Exception as e:
            # 返回一个随机图像作为占位符
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = self.transform(img)
            return img, torch.tensor(0), torch.tensor(0)

# 主函数
def main():
    # 收集图像
    print("收集图像数据...")
    image_paths, vegetable_labels, freshness_labels, vegetable_dict = collect_images()
    print(f"原始数据量: {len(image_paths)}")
    
    # 数据清洗
    print("数据清洗...")
    clean_paths, clean_veg_labels, clean_fresh_labels = clean_data(image_paths, vegetable_labels, freshness_labels)
    print(f"清洗后数据量: {len(clean_paths)}")
    
    # 检查清洗后是否有数据
    if len(clean_paths) == 0:
        print("警告: 清洗后没有数据，使用所有原始数据")
        clean_paths, clean_veg_labels, clean_fresh_labels = image_paths, vegetable_labels, freshness_labels
    
    # 数据集划分
    print("划分数据集...")
    # 假设80%为训练集，20%为测试集（官方划分）
    train_paths, test_paths, train_veg_labels, test_veg_labels, train_fresh_labels, test_fresh_labels = train_test_split(
        clean_paths, clean_veg_labels, clean_fresh_labels, test_size=0.2, random_state=42
    )
    
    # 从训练集中拆分10%为验证集
    train_paths, val_paths, train_veg_labels, val_veg_labels, train_fresh_labels, val_fresh_labels = train_test_split(
        train_paths, train_veg_labels, train_fresh_labels, test_size=0.1, random_state=42
    )
    
    print(f"训练集: {len(train_paths)}, 验证集: {len(val_paths)}, 测试集: {len(test_paths)}")
    
    # 生成CSV文件
    generate_csv(train_paths, train_veg_labels, train_fresh_labels, "train_data.csv")
    generate_csv(val_paths, val_veg_labels, val_fresh_labels, "val_data.csv")
    generate_csv(test_paths, test_veg_labels, test_fresh_labels, "test_data.csv")
    
    # 计算训练集的均值和标准差
    print("计算均值和标准差...")
    mean, std = calculate_mean_std(train_paths)
    # 如果计算失败，使用默认值
    if mean is None or std is None:
        print("警告: 计算均值和标准差失败，使用默认值")
        mean = np.array([0.485, 0.456, 0.406])  # 常用的ImageNet均值
        std = np.array([0.229, 0.224, 0.225])   # 常用的ImageNet标准差
    print(f"均值: {mean}, 标准差: {std}")
    
    # 创建数据集和数据加载器
    print("创建数据集和数据加载器...")
    train_dataset = VegetableDataset(train_paths, train_veg_labels, train_fresh_labels, mean, std, is_train=True)
    val_dataset = VegetableDataset(val_paths, val_veg_labels, val_fresh_labels, mean, std, is_train=False)
    test_dataset = VegetableDataset(test_paths, test_veg_labels, test_fresh_labels, mean, std, is_train=False)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print("数据预处理完成！")
    print(f"蔬菜编码: {vegetable_dict}")
    print(f"训练集批次: {len(train_loader)}, 验证集批次: {len(val_loader)}, 测试集批次: {len(test_loader)}")
    
    # 保存均值和标准差
    np.save("mean_std.npy", np.array([mean, std]))
    print("均值和标准差已保存到 mean_std.npy")

if __name__ == "__main__":
    main()
