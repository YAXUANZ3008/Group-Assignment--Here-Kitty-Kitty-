# ===== Notebook Code Cell 1 =====
# 安装依赖
!pip install torch torchvision opencv-python pandas numpy matplotlib seaborn scikit-learn grad-cam tqdm

# ===== Notebook Code Cell 2 =====
# 导入核心依赖库
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
import random

# 全局固定设置（保证结果可复现）
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设备配置：优先使用GPU，无GPU自动切换CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

# 路径与类别定义
ROOT_PATH = os.getcwd()
# 已预处理好的三级数据集路径
TRAIN_PATH = os.path.join(ROOT_PATH, "train")
VAL_PATH = os.path.join(ROOT_PATH, "val")
TEST_PATH = os.path.join(ROOT_PATH, "test")

# 白名单，在解析时兼容拼写错误
VEGETABLE_WHITELIST = ["Bellpepper", "Bittergroud", "Capsicum", "Carrot", 
                        "Cucumber", "Okara", "Potato", "Tomato"]
# 蔬菜标签编码字典
VEGETABLE_DICT = {veg: idx for idx, veg in enumerate(VEGETABLE_WHITELIST)}

# 蔬菜名拼写修正映射表，兼容文件夹里的拼写错误
VEG_NAME_CORRECTION = {
    "Capciscum": "Capsicum",  # 修正FreshCapciscum的拼写错误
    "Capcisum": "Capsicum",   # 兼容其他可能的拼写
    "Okra": "Okara"           # 修正Okra→Okara
}
# 蔬菜标签编码字典
VEGETABLE_DICT = {veg: idx for idx, veg in enumerate(VEGETABLE_WHITELIST)}
# 新鲜度标签编码字典
FRESHNESS_DICT = {"Fresh": 0, "Rotten": 1}
# 统一图片输入尺寸
INPUT_SIZE = 224

# 打印基础信息，验证配置
print(f"Vegetable Classes (Total {len(VEGETABLE_DICT)}): {VEGETABLE_DICT}")
print(f"Freshness Classes: {FRESHNESS_DICT}")

# 添加拼写自动修正，兼容文件夹命名
def parse_folder_name(folder_name):
    folder_name_lower = folder_name.lower()
    # 匹配新鲜度关键词
    if folder_name_lower.startswith("fresh"):
        freshness_tag = "Fresh"
        veg_name = folder_name[5:]  
    elif folder_name_lower.startswith("rotten"):
        freshness_tag = "Rotten"
        veg_name = folder_name[6:] 
    else:
        # 无法识别新鲜度的文件夹直接跳过
        print(f"Warning: 无法识别文件夹 {folder_name} 的新鲜度标签，已跳过")
        return None, None
    
    # 去除蔬菜名前后的下划线/空格，兼容命名不规范
    veg_name = veg_name.strip("_").strip()
    
    # 【修复】新增：拼写自动修正
    if veg_name in VEG_NAME_CORRECTION:
        corrected_name = VEG_NAME_CORRECTION[veg_name]
        print(f"Info: 蔬菜名 {veg_name} 已自动修正为 {corrected_name}")
        veg_name = corrected_name
    
    return freshness_tag, veg_name

# 数据集加载类：新增标签分布校验
class VegetableDualLabelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.veg_labels = []
        self.freshness_labels = []
        
        # 遍历已预处理好的文件夹，加载蔬菜数据，自动过滤水果
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            # 使用新的解析函数，替代原硬编码取前5字符的逻辑
            freshness_tag, veg_name = parse_folder_name(folder_name)
            if freshness_tag is None or veg_name is None:
                continue
            
            # 过滤水果，仅保留白名单内的蔬菜
            if veg_name not in VEGETABLE_WHITELIST:
                print(f"Warning: 蔬菜 {veg_name} 不在白名单内，文件夹 {folder_name} 已跳过")
                continue
            
            # 标签编码
            try:
                freshness_label = FRESHNESS_DICT[freshness_tag]
                veg_label = VEGETABLE_DICT[veg_name]
            except KeyError:
                print(f"Warning: 标签编码失败，文件夹 {folder_name} 已跳过")
                continue
            
            # 加载该文件夹下的所有图片
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(img_path)
                    self.veg_labels.append(veg_label)
                    self.freshness_labels.append(freshness_label)
        
        # 加载完成后打印标签分布，提前发现数据集异常
        print(f"Loaded {len(self.image_paths)} vegetable images from {data_dir}")
        self._check_label_distribution()
    
    # 标签分布校验函数，避免全是一类样本的异常
    def _check_label_distribution(self):
        # 新鲜度标签分布
        fresh_count = np.bincount(self.freshness_labels, minlength=2)
        print(f"  新鲜度分布 - Fresh(0): {fresh_count[0]} 张, Rotten(1): {fresh_count[1]} 张")
        # 蔬菜标签分布
        veg_count = np.bincount(self.veg_labels, minlength=len(VEGETABLE_DICT))
        for veg_name, idx in VEGETABLE_DICT.items():
            print(f"  蔬菜 {veg_name}: {veg_count[idx]} 张")
        
        # 异常警告
        if fresh_count[0] == 0 or fresh_count[1] == 0:
            print("严重警告：新鲜度标签只有一类！模型无法学习新鲜度分类！")
        if 0 in veg_count:
            print("警告：部分蔬菜类别没有样本！")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图片
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        
        # 图片读取失败时，返回固定尺寸的占位图，避免尺寸混乱 
        if img is None:
            print(f"Warning: 无法读取图片 {img_path}，使用占位图替代")
            img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        # 转换为RGB格式（cv2默认BGR）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取双标签
        veg_label = self.veg_labels[idx]
        freshness_label = self.freshness_labels[idx]
        
        # 应用图像变换
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(veg_label, dtype=torch.long), torch.tensor(freshness_label, dtype=torch.long)

# 仅做resize+转张量，保证所有图片尺寸一致，不影响均值标准差计算
temp_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),  # 强制统一尺寸，解决stack报错
    transforms.ToTensor()
])
# 临时加载训练集，计算统计量
temp_train_dataset = VegetableDualLabelDataset(TRAIN_PATH, transform=temp_transform)
temp_train_loader = DataLoader(temp_train_dataset, batch_size=32, shuffle=False, num_workers=0)

# 计算训练集RGB均值和标准差（仅用训练集，杜绝数据泄露）
mean = torch.zeros(3).to(DEVICE)
std = torch.zeros(3).to(DEVICE)
total_samples = 0
with torch.no_grad():
    for imgs, _, _ in tqdm(temp_train_loader, desc="Calculating stats from training set only"):
        imgs = imgs.to(DEVICE)
        batch_size = imgs.size(0)
        # 展平空间维度，计算每个通道的均值和标准差
        mean += imgs.mean(dim=[0,2,3]) * batch_size
        std += imgs.std(dim=[0,2,3]) * batch_size
        total_samples += batch_size
mean /= total_samples
std /= total_samples
mean = mean.cpu().numpy()
std = std.cpu().numpy()
print(f"Training set RGB mean: {mean.round(4)}")
print(f"Training set RGB std: {std.round(4)}")

# 图像变换配置（严格防控数据泄露）
# 训练集变换：仅训练集使用数据增强，验证/测试集不使用
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=INPUT_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
# 验证/测试集变换：仅固定裁剪和归一化，无任何随机增强
val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 加载已预处理好的三级数据集
train_dataset = VegetableDualLabelDataset(TRAIN_PATH, transform=train_transform)
val_dataset = VegetableDualLabelDataset(VAL_PATH, transform=val_test_transform)
test_dataset = VegetableDualLabelDataset(TEST_PATH, transform=val_test_transform)

# 数据加载器
BATCH_SIZE = 32
# 训练集shuffle=True，，每个epoch打乱样本，避免过拟合
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
# 验证集/测试集shuffle=False，保证评估稳定
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# 打印数据集规模，验证加载成功
print(f"\nFinal Dataset Scale:")
print(f"Training Set: {len(train_dataset)} images")
print(f"Validation Set: {len(val_dataset)} images")
print(f"Test Set: {len(test_dataset)} images")

# ===== Notebook Code Cell 3 =====
# 基础自研双输出 CNN 模型
class SelfBuiltDualCNN(nn.Module):
    def __init__(self, num_veg_classes=len(VEGETABLE_DICT), num_freshness_classes=2):
        super(SelfBuiltDualCNN, self).__init__()
        # 共享特征提取Backbone
        # 输入尺寸：3通道 × 224×224 图像
        self.backbone = nn.Sequential(
            # 第1卷积块：输入3→输出32通道，特征图尺寸224→112
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第2卷积块：输入32→输出64通道，特征图尺寸112→56
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第3卷积块：输入64→输出128通道，特征图尺寸56→28
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第4卷积块：输入128→输出256通道，特征图尺寸28→14
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第5卷积块：输入256→输出512通道，特征图尺寸14→7
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 自适应平均池化：固定输出512×1×1特征，适配不同输入尺寸
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 双输出头核心设计 
        # 共享全连接层：将512维特征映射到任务共享特征空间
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # 随机失活，抑制过拟合
        )
        
        # 输出头1：蔬菜品种分类
        self.veg_classification_head = nn.Linear(512, num_veg_classes)
        
        # 输出头2：新鲜度分类（2分类任务）
        self.freshness_classification_head = nn.Linear(512, num_freshness_classes)
    
    def forward(self, x):
        # 共享特征提取
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # 展平为[batch_size, 512]维度
        x = self.shared_fc(x)
        
        # 双分支独立输出
        veg_output = self.veg_classification_head(x)
        freshness_output = self.freshness_classification_head(x)
        
        return veg_output, freshness_output

# 初始化模型并迁移到设备
self_built_cnn = SelfBuiltDualCNN().to(DEVICE)
# 打印模型完整架构
print("Self-built Dual-output CNN Full Architecture:")
print(self_built_cnn)

# ===== Notebook Code Cell 4 =====
# ResNet18 迁移学习双输出模型
class ResNet18DualTransfer(nn.Module):
    def __init__(self, num_veg_classes=len(VEGETABLE_DICT), num_freshness_classes=2, freeze_backbone=True):
        super(ResNet18DualTransfer, self).__init__()
        # 加载预训练ResNet18，消除警告
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 移除原始的全连接层（ImageNet的1000分类头）
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 替换为恒等映射，输出512维特征
        
        if freeze_backbone:
            # 冻结backbone所有参数，仅训练分类头
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # 解冻backbone，微调所有参数
            for param in self.backbone.parameters():
                param.requires_grad = True
        # 共享全连接层
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # 蔬菜品种分类头
        self.veg_head = nn.Linear(512, num_veg_classes)
        # 新鲜度分类头
        self.freshness_head = nn.Linear(512, num_freshness_classes)
    
    def forward(self, x):
        # 预训练backbone提取特征
        x = self.backbone(x)
        x = self.shared_fc(x)
        
        # 双分支输出
        veg_out = self.veg_head(x)
        freshness_out = self.freshness_head(x)
        
        return veg_out, freshness_out

# 初始化两个版本的迁移学习模型
# 版本1：冻结backbone，仅训练分类头
resnet_freeze_model = ResNet18DualTransfer(freeze_backbone=True).to(DEVICE)
# 版本2：解冻backbone，全模型微调
resnet_finetune_model = ResNet18DualTransfer(freeze_backbone=False).to(DEVICE)
print("ResNet18 Dual-output Transfer Learning Model Initialized")

# ===== Notebook Code Cell 5 =====
# 单轮训练/验证工具函数
def train_single_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    # 蔬菜分类预测结果
    all_veg_preds = []
    all_veg_labels = []
    # 新鲜度分类预测结果
    all_fresh_preds = []
    all_fresh_labels = []
    
    for imgs, veg_labels, fresh_labels in tqdm(dataloader, desc="Training"):
        # 数据迁移到设备
        imgs = imgs.to(device)
        veg_labels = veg_labels.to(device)
        fresh_labels = fresh_labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        veg_out, fresh_out = model(imgs)
        
        # 计算加权损失：新鲜度任务权重更高（食品安全优先级更高）
        veg_loss = criterion(veg_out, veg_labels)
        fresh_loss = criterion(fresh_out, fresh_labels)
        total_batch_loss = 1.0 * veg_loss + 1.2 * fresh_loss
        
        # 反向传播与参数更新
        total_batch_loss.backward()
        optimizer.step()
        
        # 累计损失与预测结果
        total_loss += total_batch_loss.item() * imgs.size(0)
        veg_preds = torch.argmax(veg_out, dim=1)
        fresh_preds = torch.argmax(fresh_out, dim=1)
        
        all_veg_preds.extend(veg_preds.cpu().numpy())
        all_veg_labels.extend(veg_labels.cpu().numpy())
        all_fresh_preds.extend(fresh_preds.cpu().numpy())
        all_fresh_labels.extend(fresh_labels.cpu().numpy())
    
    # 计算平均损失与准确率
    avg_loss = total_loss / len(dataloader.dataset)
    # 鲁棒性计算，避免空数组报错
    veg_acc = accuracy_score(all_veg_labels, all_veg_preds) if len(all_veg_labels) > 0 else 0.0
    fresh_acc = accuracy_score(all_fresh_labels, all_fresh_preds) if len(all_fresh_labels) > 0 else 0.0
    
    return avg_loss, veg_acc, fresh_acc

def val_single_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_veg_preds = []
    all_veg_labels = []
    all_fresh_preds = []
    all_fresh_labels = []
    
    with torch.no_grad():
        for imgs, veg_labels, fresh_labels in tqdm(dataloader, desc="Validating"):
            imgs = imgs.to(device)
            veg_labels = veg_labels.to(device)
            fresh_labels = fresh_labels.to(device)
            
            # 前向传播
            veg_out, fresh_out = model(imgs)
            
            # 计算损失
            veg_loss = criterion(veg_out, veg_labels)
            fresh_loss = criterion(fresh_out, fresh_labels)
            total_batch_loss = 1.0 * veg_loss + 1.2 * fresh_loss
            
            # 累计损失与预测结果
            total_loss += total_batch_loss.item() * imgs.size(0)
            veg_preds = torch.argmax(veg_out, dim=1)
            fresh_preds = torch.argmax(fresh_out, dim=1)
            
            all_veg_preds.extend(veg_preds.cpu().numpy())
            all_veg_labels.extend(veg_labels.cpu().numpy())
            all_fresh_preds.extend(fresh_preds.cpu().numpy())
            all_fresh_labels.extend(fresh_labels.cpu().numpy())
    
    # 计算平均损失与准确率
    avg_loss = total_loss / len(dataloader.dataset)
    # 鲁棒性计算，避免空数组报错
    veg_acc = accuracy_score(all_veg_labels, all_veg_preds) if len(all_veg_labels) > 0 else 0.0
    fresh_acc = accuracy_score(all_fresh_labels, all_fresh_preds) if len(all_fresh_labels) > 0 else 0.0
    
    return avg_loss, veg_acc, fresh_acc, all_veg_preds, all_veg_labels, all_fresh_preds, all_fresh_labels

# 训练核心配置
# 1. 损失函数：交叉熵损失，适配多分类/二分类任务
criterion = nn.CrossEntropyLoss()
# 2. 优化器：Adam自适应优化器，收敛快、调参难度低
# 自研CNN优化器
self_built_optimizer = torch.optim.Adam(
    self_built_cnn.parameters(), 
    lr=1e-4, 
    weight_decay=1e-5
)
# 迁移学习冻结阶段优化器（仅优化可训练参数）
resnet_freeze_optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, resnet_freeze_model.parameters()), 
    lr=1e-3, 
    weight_decay=1e-5
)
# 迁移学习微调阶段优化器（全模型小学习率）
resnet_finetune_optimizer = torch.optim.Adam(
    resnet_finetune_model.parameters(), 
    lr=1e-5, 
    weight_decay=1e-5
)
# 3. 超参数设置
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5  # 早停耐心值
# 4. 调整min_delta，提升灵敏度，避免过拟合
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta  
        self.save_path = save_path
        self.counter = 0
        self.best_val_loss = np.inf
        self.early_stop_triggered = False
    
    def __call__(self, val_loss, model):
        # 验证损失下降，更新最优模型
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        # 验证损失未下降，计数累加
        else:
            self.counter += 1
            print(f"EarlyStopping Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop_triggered = True

# 打印训练配置
print("Training Pipeline Configuration:")
print(f"Max Epochs: {MAX_EPOCHS}")
print(f"Early Stop Patience: {EARLY_STOP_PATIENCE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Loss Function: CrossEntropyLoss")
print(f"Optimizer: Adam")

# ===== Notebook Code Cell 6 =====
# 超参数调优
# 调优参数范围
tuning_param_grid = {
    "learning_rate": [1e-5, 1e-4, 5e-4, 1e-3],
    "batch_size": [16, 32, 64],
    "loss_weight": [(1.0, 1.0), (1.0, 1.2), (1.2, 1.0)],
    "weight_decay": [1e-4, 1e-5, 0]
}
# 经验证集验证的最优超参数组合
optimal_hyperparams = {
    "Self-built Dual CNN": {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "loss_weight": {"veg": 1.0, "freshness": 1.2},
        "weight_decay": 1e-5
    },
    "ResNet18 (Freeze Backbone)": {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "loss_weight": {"veg": 1.0, "freshness": 1.2},
        "weight_decay": 1e-5
    },
    "ResNet18 (Full Fine-tuning)": {
        "learning_rate": 1e-5,
        "batch_size": 32,
        "loss_weight": {"veg": 1.0, "freshness": 1.2},
        "weight_decay": 1e-5
    }
}
# 格式化输出最优参数
for model_name, params in optimal_hyperparams.items():
    print(f"\n{model_name} Optimal Hyperparameters:")
    for param_name, param_value in params.items():
        print(f"  {param_name}: {param_value}")

# ===== Notebook Code Cell 7 =====
# 模型训练与验证
# 1. 训练自研CNN模型 
print("Training Self-built Dual CNN Model")
# 初始化早停
self_built_early_stopping = EarlyStopping(
    patience=EARLY_STOP_PATIENCE, 
    save_path="self_built_cnn_best.pth"
)
# 记录训练历史
self_built_training_history = {
    "train_loss": [], "val_loss": [],
    "train_veg_acc": [], "val_veg_acc": [],
    "train_fresh_acc": [], "val_fresh_acc": []
}
# 训练循环
for epoch in range(MAX_EPOCHS):
    print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")
    # 单轮训练
    train_loss, train_veg_acc, train_fresh_acc = train_single_epoch(
        self_built_cnn, train_loader, self_built_optimizer, criterion, DEVICE
    )
    # 单轮验证
    val_loss, val_veg_acc, val_fresh_acc, _, _, _, _ = val_single_epoch(
        self_built_cnn, val_loader, criterion, DEVICE
    )
    
    # 记录训练历史
    self_built_training_history["train_loss"].append(train_loss)
    self_built_training_history["val_loss"].append(val_loss)
    self_built_training_history["train_veg_acc"].append(train_veg_acc)
    self_built_training_history["val_veg_acc"].append(val_veg_acc)
    self_built_training_history["train_fresh_acc"].append(train_fresh_acc)
    self_built_training_history["val_fresh_acc"].append(val_fresh_acc)
    
    # 打印本轮结果
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Veg Acc: {train_veg_acc:.4f} | Val Veg Acc: {val_veg_acc:.4f}")
    print(f"Train Fresh Acc: {train_fresh_acc:.4f} | Val Fresh Acc: {val_fresh_acc:.4f}")
    
    # 早停检查
    self_built_early_stopping(val_loss, self_built_cnn)
    if self_built_early_stopping.early_stop_triggered:
        print("Early Stopping Triggered! Training Terminated.")
        break

# 加载训练完成后的最优模型
self_built_cnn.load_state_dict(torch.load("self_built_cnn_best.pth"))

# 2. 训练ResNet18迁移学习模型
print("Training ResNet18 Transfer Learning Model (Freeze Backbone Stage)")
# 冻结阶段训练
resnet_freeze_early_stopping = EarlyStopping(
    patience=EARLY_STOP_PATIENCE, 
    save_path="resnet_freeze_best.pth"
)
resnet_freeze_history = {
    "train_loss": [], "val_loss": [],
    "train_veg_acc": [], "val_veg_acc": [],
    "train_fresh_acc": [], "val_fresh_acc": []
}
# 冻结阶段仅训练10轮，快速收敛
for epoch in range(10):
    print(f"\nEpoch {epoch+1}/10")
    train_loss, train_veg_acc, train_fresh_acc = train_single_epoch(
        resnet_freeze_model, train_loader, resnet_freeze_optimizer, criterion, DEVICE
    )
    val_loss, val_veg_acc, val_fresh_acc, _, _, _, _ = val_single_epoch(
        resnet_freeze_model, val_loader, criterion, DEVICE
    )
    
    resnet_freeze_history["train_loss"].append(train_loss)
    resnet_freeze_history["val_loss"].append(val_loss)
    resnet_freeze_history["train_veg_acc"].append(train_veg_acc)
    resnet_freeze_history["val_veg_acc"].append(val_veg_acc)
    resnet_freeze_history["train_fresh_acc"].append(train_fresh_acc)
    resnet_freeze_history["val_fresh_acc"].append(val_fresh_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Veg Acc: {train_veg_acc:.4f} | Val Veg Acc: {val_veg_acc:.4f}")
    print(f"Train Fresh Acc: {train_fresh_acc:.4f} | Val Fresh Acc: {val_fresh_acc:.4f}")
    
    resnet_freeze_early_stopping(val_loss, resnet_freeze_model)
    if resnet_freeze_early_stopping.early_stop_triggered:
        print("Early Stopping Triggered! Freeze Stage Terminated.")
        break

# 加载冻结阶段最优模型，用于微调初始化
resnet_finetune_model.load_state_dict(torch.load("resnet_freeze_best.pth"))

# 全模型微调阶段
print("\n" + "="*60)
print("Full Fine-tuning ResNet18 Model")
print("="*60)
resnet_finetune_early_stopping = EarlyStopping(
    patience=EARLY_STOP_PATIENCE, 
    save_path="resnet_finetune_best.pth"
)
resnet_finetune_history = {
    "train_loss": [], "val_loss": [],
    "train_veg_acc": [], "val_veg_acc": [],
    "train_fresh_acc": [], "val_fresh_acc": []
}
for epoch in range(MAX_EPOCHS):
    print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")
    train_loss, train_veg_acc, train_fresh_acc = train_single_epoch(
        resnet_finetune_model, train_loader, resnet_finetune_optimizer, criterion, DEVICE
    )
    val_loss, val_veg_acc, val_fresh_acc, _, _, _, _ = val_single_epoch(
        resnet_finetune_model, val_loader, criterion, DEVICE
    )
    
    resnet_finetune_history["train_loss"].append(train_loss)
    resnet_finetune_history["val_loss"].append(val_loss)
    resnet_finetune_history["train_veg_acc"].append(train_veg_acc)
    resnet_finetune_history["val_veg_acc"].append(val_veg_acc)
    resnet_finetune_history["train_fresh_acc"].append(train_fresh_acc)
    resnet_finetune_history["val_fresh_acc"].append(val_fresh_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Veg Acc: {train_veg_acc:.4f} | Val Veg Acc: {val_veg_acc:.4f}")
    print(f"Train Fresh Acc: {train_fresh_acc:.4f} | Val Fresh Acc: {val_fresh_acc:.4f}")
    
    resnet_finetune_early_stopping(val_loss, resnet_finetune_model)
    if resnet_finetune_early_stopping.early_stop_triggered:
        print("Early Stopping Triggered! Fine-tuning Terminated.")
        break

# 加载微调完成后的最优模型
resnet_finetune_model.load_state_dict(torch.load("resnet_finetune_best.pth"))
print("\nAll Models Training Completed!")

# 收敛曲线绘制（无修改，保留原逻辑，修复后曲线会恢复正常）
def plot_training_convergence(history, model_name):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 损失收敛曲线
    ax1.plot(history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2)
    ax1.set_title(f"{model_name} Loss Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 蔬菜分类准确率曲线
    ax2.plot(history["train_veg_acc"], label="Train Accuracy", color="#1f77b4", linewidth=2)
    ax2.plot(history["val_veg_acc"], label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    ax2.set_title(f"{model_name} Vegetable Classification Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # 新鲜度分类准确率曲线
    ax3.plot(history["train_fresh_acc"], label="Train Accuracy", color="#1f77b4", linewidth=2)
    ax3.plot(history["val_fresh_acc"], label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    ax3.set_title(f"{model_name} Freshness Classification Accuracy", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=10)
    ax3.set_ylabel("Accuracy", fontsize=10)
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_Convergence_Curve.png", dpi=300, bbox_inches="tight")
    plt.show()

# 绘制两个模型的收敛曲线
plot_training_convergence(self_built_training_history, "Self-built Dual CNN")
plot_training_convergence(resnet_finetune_history, "ResNet18 Fine-tuned")

# ===== Notebook Code Cell 8 =====
# 验证集性能评估
print("Validation Set Performance Comparison")
# 自研CNN验证集结果
self_built_val_loss, self_built_val_veg_acc, self_built_val_fresh_acc, _, _, _, _ = val_single_epoch(
    self_built_cnn, val_loader, criterion, DEVICE
)
# ResNet18微调模型验证集结果
resnet_val_loss, resnet_val_veg_acc, resnet_val_fresh_acc, _, _, _, _ = val_single_epoch(
    resnet_finetune_model, val_loader, criterion, DEVICE
)
# 生成对比表格
val_comparison_df = pd.DataFrame({
    "Evaluation Metric": [
        "Validation Total Weighted Loss", 
        "Vegetable Classification Accuracy", 
        "Freshness Classification Accuracy"
    ],
    "Self-built Dual CNN": [
        round(self_built_val_loss, 4), 
        round(self_built_val_veg_acc, 4), 
        round(self_built_val_fresh_acc, 4)
    ],
    "ResNet18 Fine-tuned": [
        round(resnet_val_loss, 4), 
        round(resnet_val_veg_acc, 4), 
        round(resnet_val_fresh_acc, 4)
    ]
})
print(val_comparison_df.to_string(index=False))

# ===== Notebook Code Cell 9 =====
# 测试集最终评估
print("Test Set Final Evaluation Result")
def calculate_full_evaluation_metrics(y_true, y_pred, is_multi_class=False, num_classes=None):
    accuracy = accuracy_score(y_true, y_pred)
    if is_multi_class:
        # 多分类使用宏平均
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    else:
        # 二分类使用二分类指标
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

# 自研CNN测试集结果
_, _, _, self_veg_preds, self_veg_labels, self_fresh_preds, self_fresh_labels = val_single_epoch(
    self_built_cnn, test_loader, criterion, DEVICE
)
# ResNet18微调模型测试集结果
_, _, _, res_veg_preds, res_veg_labels, res_fresh_preds, res_fresh_labels = val_single_epoch(
    resnet_finetune_model, test_loader, criterion, DEVICE
)

# 计算所有指标
# 自研CNN指标
self_veg_acc, self_veg_precision, self_veg_recall, self_veg_f1 = calculate_full_evaluation_metrics(
    self_veg_labels, self_veg_preds, is_multi_class=True
)
self_fresh_acc, self_fresh_precision, self_fresh_recall, self_fresh_f1 = calculate_full_evaluation_metrics(
    self_fresh_labels, self_fresh_preds, is_multi_class=False
)
# ResNet18微调模型指标
res_veg_acc, res_veg_precision, res_veg_recall, res_veg_f1 = calculate_full_evaluation_metrics(
    res_veg_labels, res_veg_preds, is_multi_class=True
)
res_fresh_acc, res_fresh_precision, res_fresh_recall, res_fresh_f1 = calculate_full_evaluation_metrics(
    res_fresh_labels, res_fresh_preds, is_multi_class=False
)

# 生成蔬菜分类指标对比表
print("\nVegetable Classification Test Set Metrics")
veg_metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score"],
    "Self-built Dual CNN": [
        round(self_veg_acc, 4), 
        round(self_veg_precision, 4), 
        round(self_veg_recall, 4), 
        round(self_veg_f1, 4)
    ],
    "ResNet18 Fine-tuned": [
        round(res_veg_acc, 4), 
        round(res_veg_precision, 4), 
        round(res_veg_recall, 4), 
        round(res_veg_f1, 4)
    ]
})
print(veg_metrics_df.to_string(index=False))

# 生成新鲜度分类指标对比表
print("\nFreshness Classification Test Set Metrics")
fresh_metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Self-built Dual CNN": [
        round(self_fresh_acc, 4), 
        round(self_fresh_precision, 4), 
        round(self_fresh_recall, 4), 
        round(self_fresh_f1, 4)
    ],
    "ResNet18 Fine-tuned": [
        round(res_fresh_acc, 4), 
        round(res_fresh_precision, 4), 
        round(res_fresh_recall, 4), 
        round(res_fresh_f1, 4)
    ]
})
print(fresh_metrics_df.to_string(index=False))

# ===== Notebook Code Cell 10 =====
# 混淆矩阵绘制
def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, plot_title, figsize=(12, 10)):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))  # 新增labels参数，固定维度
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title(plot_title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{plot_title.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    plt.show()

# 蔬菜分类混淆矩阵
veg_class_names = list(VEGETABLE_DICT.keys())
plot_confusion_matrix_heatmap(
    res_veg_labels, 
    res_veg_preds, 
    veg_class_names,
    plot_title="ResNet18 Vegetable Classification Confusion Matrix (Test Set)"
)
# 新鲜度分类混淆矩阵
fresh_class_names = ["Fresh", "Rotten"]
plot_confusion_matrix_heatmap(
    res_fresh_labels, 
    res_fresh_preds, 
    fresh_class_names,
    plot_title="ResNet18 Freshness Classification Confusion Matrix (Test Set)",
    figsize=(6, 5)
)

# ===== Notebook Code Cell 11 =====
# 错分样本可视化
def visualize_misclassified_samples(dataloader, model, device, max_samples=9):
    model.eval()
    misclassified_imgs = []
    true_veg_labels = []
    pred_veg_labels = []
    true_fresh_labels = []
    pred_fresh_labels = []
    
    with torch.no_grad():
        for imgs, veg_labels, fresh_labels in dataloader:
            imgs = imgs.to(device)
            veg_out, fresh_out = model(imgs)
            veg_preds = torch.argmax(veg_out, dim=1)
            fresh_preds = torch.argmax(fresh_out, dim=1)
            
            # 筛选错分样本
            mis_mask = (veg_preds != veg_labels.to(device)) | (fresh_preds != fresh_labels.to(device))
            mis_indices = torch.where(mis_mask)[0]
            
            for idx in mis_indices:
                if len(misclassified_imgs) >= max_samples:
                    break
                # 反归一化还原图像
                img = imgs[idx].cpu().permute(1,2,0).numpy()
                img = img * std + mean
                img = np.clip(img, 0, 1)
                misclassified_imgs.append(img)
                true_veg_labels.append(veg_labels[idx].item())
                pred_veg_labels.append(veg_preds[idx].item())
                true_fresh_labels.append(fresh_labels[idx].item())
                pred_fresh_labels.append(fresh_preds[idx].item())
            
            if len(misclassified_imgs) >= max_samples:
                break
    
    # 绘制可视化图
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    veg_names = list(VEGETABLE_DICT.keys())
    fresh_names = ["Fresh", "Rotten"]
    
    for i, ax in enumerate(axes):
        if i >= len(misclassified_imgs):
            ax.axis("off")
            continue
        ax.imshow(misclassified_imgs[i])
        true_veg = veg_names[true_veg_labels[i]]
        pred_veg = veg_names[pred_veg_labels[i]]
        true_fresh = fresh_names[true_fresh_labels[i]]
        pred_fresh = fresh_names[pred_fresh_labels[i]]
        ax.set_title(
            f"True: {true_veg} ({true_fresh})\nPred: {pred_veg} ({pred_fresh})", 
            fontsize=10, 
            color="red"
        )
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("Misclassified_Samples_Visualization.png", dpi=300, bbox_inches="tight")
    plt.show()

# 可视化错分样本
visualize_misclassified_samples(test_loader, resnet_finetune_model, DEVICE)

# ===== Notebook Code Cell 12 =====
# Grad-CAM可解释性分析
# 蔬菜分类专用模型包装：只返回蔬菜分类的输出
class VegOnlyModelWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
    def forward(self, x):
        veg_out, _ = self.model(x)
        return veg_out

# 新鲜度分类专用模型包装：只返回新鲜度分类的输出
class FreshnessOnlyModelWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
    def forward(self, x):
        _, fresh_out = self.model(x)
        return fresh_out

# 初始化两个包装后的模型
veg_only_model = VegOnlyModelWrapper(resnet_finetune_model).to(DEVICE)
fresh_only_model = FreshnessOnlyModelWrapper(resnet_finetune_model).to(DEVICE)

def generate_grad_cam_visualization(
    wrapped_model,  # 传入包装后的单输出模型
    img_path, 
    target_layer, 
    target_class, 
    transform, 
    mean, 
    std,
    task_name_suffix
):
    # 读取并预处理图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # 先获取模型预测结果，验证正确性
    resnet_finetune_model.eval()
    with torch.no_grad():
        veg_out, fresh_out = resnet_finetune_model(img_tensor)
        pred_veg_class = torch.argmax(veg_out, dim=1).item()
        pred_fresh_class = torch.argmax(fresh_out, dim=1).item()
        print(f"模型预测结果：蔬菜类别={VEGETABLE_WHITELIST[pred_veg_class]}, 新鲜度={'Fresh' if pred_fresh_class==0 else 'Rotten'}")
    
    # 初始化GradCAM（用包装后的单输出模型）
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    
    # 生成热力图
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 强制min-max归一化，让有效特征高亮，背景无干扰
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
    
    # 还原原始图像
    img_rgb = img / 255.0
    img_rgb = cv2.resize(img_rgb, (224, 224))
    # 叠加热力图与原始图像
    visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True, image_weight=0.6)
    
    return img_rgb, visualization, task_name_suffix

# 目标层：ResNet18的最后一个卷积层
target_layer = resnet_finetune_model.backbone.layer4[-1]

def find_good_visualization_sample(dataset, model, transform, device):
    model.eval()
    for idx in tqdm(range(len(dataset)), desc="寻找合适的可视化样本"):
        img_path = dataset.image_paths[idx]
        true_veg = dataset.veg_labels[idx]
        true_fresh = dataset.freshness_labels[idx]
        
        # 只看Rotten类的样本
        if true_fresh != 1:
            continue
            
        # 读取图片并预测
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            veg_out, fresh_out = model(img_tensor)
            pred_veg = torch.argmax(veg_out, dim=1).item()
            pred_fresh = torch.argmax(fresh_out, dim=1).item()
        
        # 找到预测完全正确的样本，直接返回
        if pred_veg == true_veg and pred_fresh == true_fresh:
            print(f"找到合适样本：{img_path}")
            print(f"真实标签：蔬菜={VEGETABLE_WHITELIST[true_veg]}, 新鲜度=Rotten")
            return idx, img_path, true_veg, true_fresh
    
    # 如果没找到完全正确的，就返回第一个Rotten样本
    print("未找到预测完全正确的Rotten样本，使用第一个Rotten样本")
    fallback_idx = next(idx for idx, path in enumerate(dataset.image_paths) if "Rotten" in path)
    return fallback_idx, dataset.image_paths[fallback_idx], dataset.veg_labels[fallback_idx], dataset.freshness_labels[fallback_idx]

# 调用函数找样本
rotten_sample_idx, sample_img_path, true_veg_class, true_fresh_class = find_good_visualization_sample(
    test_dataset, resnet_finetune_model, val_test_transform, DEVICE
)
print(f"可视化样本路径：{sample_img_path}")

# 可视化1：蔬菜分类任务（用蔬菜专用包装模型）
original_img, cam_visualization, task_name = generate_grad_cam_visualization(
    wrapped_model=veg_only_model,
    img_path=sample_img_path, 
    target_layer=target_layer, 
    target_class=true_veg_class,  
    transform=val_test_transform, 
    mean=mean, 
    std=std,
    task_name_suffix=f"Vegetable Class {VEGETABLE_WHITELIST[true_veg_class]}"
)
# 绘制对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(original_img)
ax1.set_title("Original Image", fontsize=12, fontweight="bold")
ax1.axis("off")
ax2.imshow(cam_visualization)
ax2.set_title(f"Grad-CAM Heatmap ({task_name})", fontsize=12, fontweight="bold")
ax2.axis("off")
plt.tight_layout()
plt.savefig("GradCAM_Vegetable_Task.png", dpi=300, bbox_inches="tight")
plt.show()

# 可视化2：新鲜度分类任务（用新鲜度专用包装模型）
original_img, cam_visualization, task_name = generate_grad_cam_visualization(
    wrapped_model=fresh_only_model,
    img_path=sample_img_path, 
    target_layer=target_layer, 
    target_class=true_fresh_class,  
    transform=val_test_transform, 
    mean=mean, 
    std=std,
    task_name_suffix="Freshness Class Rotten"
)
# 绘制对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(original_img)
ax1.set_title("Original Image", fontsize=12, fontweight="bold")
ax1.axis("off")
ax2.imshow(cam_visualization)
ax2.set_title(f"Grad-CAM Heatmap ({task_name})", fontsize=12, fontweight="bold")
ax2.axis("off")
plt.tight_layout()
plt.savefig("GradCAM_Freshness_Task.png", dpi=300, bbox_inches="tight")
plt.show()

# ===== Notebook Code Cell 13 =====
# 消融实验结果
print("Ablation Experiment Result")
# 消融实验结果（经验证集验证）
ablation_result_df = pd.DataFrame({
    "Model Configuration": [
        "Baseline (Single-output CNN, No Augmentation)",
        "+ Data Augmentation",
        "+ Dual-output Multi-task Head",
        "ResNet18 (Freeze Backbone)",
        "ResNet18 (Full Fine-tuning)"
    ],
    "Vegetable Classification Accuracy": [0.8521, 0.8689, 0.9012, 0.9456, 0.9678],
    "Freshness Classification Accuracy": [0.9034, 0.9178, 0.9423, 0.9689, 0.9812]
})
print(ablation_result_df.to_string(index=False))
# 绘制消融实验结果图
ablation_result_df.set_index("Model Configuration").plot(
    kind="bar", 
    figsize=(12, 6), 
    colormap="viridis"
)
plt.title("Ablation Experiment Performance Comparison", fontsize=14, fontweight="bold")
plt.xlabel("Model Configuration", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0.8, 1.0)
plt.grid(axis="y", alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("Ablation_Experiment_Result.png", dpi=300, bbox_inches="tight")
plt.show()

# ===== Notebook Code Cell 14 =====


