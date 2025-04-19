import torch
import os
import glob

data_dir = 'D:/论文/CLAM/CLAM-master/RRT_data/tcga-subtyping/TCGA-BRCA PLIP/pt_files/'
slide_id = 'TCGA-EW-A1P6'

pattern = os.path.join(data_dir, f"{slide_id}*.pt")
matched_files = glob.glob(pattern)

if len(matched_files) == 1:
    full_path = matched_files[0]
    features = torch.load(full_path, weights_only=True)
    print(f"Features shape: {features.shape}")
elif len(matched_files) == 0:
    print(f"No .pt file found for slide_id: {slide_id}")
else:
    print(f"Multiple .pt files found for slide_id: {slide_id}")
