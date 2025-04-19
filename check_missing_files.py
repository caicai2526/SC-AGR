import os
import pandas as pd
import glob

def check_missing_files(split_dir, data_dir):
    # 获取所有split文件
    split_files = glob.glob(os.path.join(split_dir, "splits_*.csv"))
    if not split_files:
        print(f"No split files found in {split_dir}")
        return
    
    # 获取所有pt文件的名称
    existing_pt_files = set(os.path.basename(f) for f in glob.glob(os.path.join(data_dir, "*.pt")))
    
    for split_file in split_files:
        print(f"Checking split file: {split_file}")
        splits = pd.read_csv(split_file)
        for split_type in ['train', 'val', 'test']:
            if split_type not in splits.columns:
                continue
            slide_ids = splits[split_type].dropna().tolist()
            for slide_id in slide_ids:
                # 假设slide_id对应的文件名为 slide_id + .pt
                # 需要根据实际情况调整文件名格式
                pt_filename = f"{slide_id}.pt"
                if pt_filename not in existing_pt_files:
                    print(f"Missing file for slide_id {slide_id}: {pt_filename}")

if __name__ == "__main__":
    split_dir = "D:/论文/CLAM/CLAM-master/splits/TCGA_BC_R50_100"  # 根据实际情况修改
    data_dir = "D:/论文/CLAM/CLAM-master/RRT_data/tcga-subtyping/TCGA-BRCA R50/pt_files/"
    check_missing_files(split_dir, data_dir)
