import os
import shutil
import pandas as pd

# 数据集路径
BASE_PATH = "e:\\Dataset(蔬果）"
CSV_PATHS = {
    "train": "train_data.csv",
    "val": "val_data.csv",
    "test": "test_data.csv"
}

# 目标数据集路径
TARGET_BASE = os.path.join(BASE_PATH, "dataset")

# 蔬菜编码映射
vegetable_dict = {
    0: "FreshApple", 1: "FreshBanana", 2: "FreshBellpepper", 3: "FreshBittergroud",
    4: "FreshCapciscum", 5: "FreshCarrot", 6: "FreshCucumber", 7: "FreshMango",
    8: "FreshOkara", 9: "FreshOrange", 10: "FreshPotato", 11: "FreshStrawberry",
    12: "FreshTomato", 13: "RottenApple", 14: "RottenBanana", 15: "RottenBellpepper",
    16: "RottenBittergroud", 17: "RottenCapsicum", 18: "RottenCarrot", 19: "RottenCucumber",
    20: "RottenMango", 21: "RottenOkra", 22: "RottenOrange", 23: "RottenPotato",
    24: "RottenStrawberry", 25: "RottenTomato"
}

def create_directory_structure():
    """创建数据集目录结构"""
    for split in ["train", "val", "test"]:
        split_path = os.path.join(TARGET_BASE, split)
        os.makedirs(split_path, exist_ok=True)
        
        # 创建每个蔬菜类别的目录
        for veg_id, veg_name in vegetable_dict.items():
            veg_path = os.path.join(split_path, veg_name)
            os.makedirs(veg_path, exist_ok=True)

def copy_images():
    """根据CSV文件复制图片到对应目录"""
    for split, csv_file in CSV_PATHS.items():
        csv_path = os.path.join(BASE_PATH, csv_file)
        df = pd.read_csv(csv_path)
        
        print(f"处理 {split} 数据集，共 {len(df)} 张图片...")
        
        for idx, row in df.iterrows():
            # 获取原始图片路径
            src_path = row["image_path"]
            veg_label = row["vegetable_label"]
            
            # 获取目标目录
            veg_name = vegetable_dict[veg_label]
            target_dir = os.path.join(TARGET_BASE, split, veg_name)
            
            # 生成目标文件名
            filename = os.path.basename(src_path)
            target_path = os.path.join(target_dir, filename)
            
            # 复制图片
            try:
                if os.path.exists(src_path):
                    shutil.copy2(src_path, target_path)
                    if idx % 100 == 0:
                        print(f"已复制 {idx} 张图片")
                else:
                    print(f"警告: 图片不存在: {src_path}")
            except Exception as e:
                print(f"复制失败 {src_path}: {e}")
        
        print(f"{split} 数据集处理完成！")

def main():
    print("开始创建训练用数据集...")
    
    # 创建目录结构
    create_directory_structure()
    print("目录结构创建完成")
    
    # 复制图片
    copy_images()
    
    print("\n训练用数据集创建完成！")
    print(f"数据集路径: {TARGET_BASE}")

if __name__ == "__main__":
    main()
