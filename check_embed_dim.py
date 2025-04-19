import torch
import os
import glob

def verify_embed_dim(data_dir, expected_dim=1024):
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    for file in pt_files:
        try:
            features = torch.load(file, map_location='cpu')
            if isinstance(features, dict):
                # 如果 .pt 文件存储为字典，尝试获取特征张量
                if 'features' in features:
                    features = features['features']
                else:
                    print(f"{file}: 存储为字典但缺少 'features' 键")
                    continue
            if not isinstance(features, torch.Tensor):
                print(f"{file}: 加载的对象不是 torch.Tensor")
                continue
            print(f"{file}: {features.shape}")
            assert features.shape[1] == expected_dim, f"Feature dimension mismatch in {file}: {features.shape[1]} != {expected_dim}"
        except Exception as e:
            print(f"Error loading {file}: {e}")

if __name__ == "__main__":
    data_dir = "D:/论文/CLAM/CLAM-master/RRT_data/tcga-subtyping/TCGA-BRCA R50/pt_files/"
    verify_embed_dim(data_dir)
