import csv
import os
import pandas as pd
def verify_slides(csv_file, data_dir):
    if not os.path.exists(csv_file):
        print(f"CSV 文件不存�? {csv_file}")
        return
    missing_files = []
    with open(csv_file, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_slide_id = row['train']
            val_slide_id = row['val']
            test_slide_id = row['test']
            
            # 检查训练集 slide_id
            if train_slide_id and not pd.isna(train_slide_id):
                pt_file = os.path.join(data_dir, f"{train_slide_id}.pt")
                if not os.path.exists(pt_file):
                    missing_files.append(train_slide_id)
            
            # 检查验证集 slide_id
            if val_slide_id and not pd.isna(val_slide_id):
                pt_file = os.path.join(data_dir, f"{val_slide_id}.pt")
                if not os.path.exists(pt_file):
                    missing_files.append(val_slide_id)
            
            # 检查测试集 slide_id
            if test_slide_id and not pd.isna(test_slide_id):
                pt_file = os.path.join(data_dir, f"{test_slide_id}.pt")
                if not os.path.exists(pt_file):
                    missing_files.append(test_slide_id)
    
    if missing_files:
        print(f"以下 slide_id 对应�?.pt 文件缺失 ({csv_file}):")
        for slide_id in missing_files:
            print(f"- {slide_id}.pt")
    else:
        print(f"所�?slide_id 对应�?.pt 文件�?{csv_file} 中均存在�?")

if __name__ == "__main__":
    tasks = {
        'tumor_vs_normal_plip': {
            'csv_dir': 'D:/论文/CLAM/CLAM-master/splits/tumor_vs_normal_plip_100',
            'data_dir': 'D:/论文/CLAM/CLAM-master/RRT_data/camelyon16-diagnosis/combined_pt_plip'
        },
        'tumor_vs_normal_R50': {
            'csv_dir': 'D:/论文/CLAM/CLAM-master/splits/tumor_vs_normal_R50_100',
            'data_dir': 'D:/论文/CLAM/CLAM-master/RRT_data/camelyon16-diagnosis/combined_pt_R50'
        }
    }
    
    for task, paths in tasks.items():
        csv_dir = paths['csv_dir']
        data_dir = paths['data_dir']
        
        if not os.path.exists(csv_dir):
            print(f"分割目录不存�? {csv_dir}")
            continue
        
        # 遍历所有分割文�?
        for fold in range(10):
            split_file = os.path.join(csv_dir, f'splits_{fold}.csv')
            verify_slides(split_file, data_dir)
