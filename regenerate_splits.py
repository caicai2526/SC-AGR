import os
import numpy as np
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, save_splits

def regenerate_splits():
    # 配置参数
    task = 'TCGA_BC_R50'
    n_classes = 2
    seed = 1
    k = 5
    label_frac = 1.0
    val_frac = 0.1
    test_frac = 0.1

    # 文件路径和标签映射
    csv_path = os.path.join('RRT_data', 'tcga-subtyping', 'TCGA-BRCA R50', 'label.csv')
    label_dict = {'IDC': 0, 'ILC': 1}
    label_col = 'label'

    print(f"Task: {task}")
    print(f"CSV path: {csv_path}")
    print(f"Label dictionary: {label_dict}")
    print(f"Label column: {label_col}")

    # 初始化数据集
    print("Initializing Generic_WSI_Classification_Dataset with patient_voting='maj'")
    dataset = Generic_WSI_Classification_Dataset(
        csv_path=csv_path,
        shuffle=False,
        seed=seed,
        print_info=True,
        label_dict=label_dict,
        patient_strat=True,
        patient_voting='maj',
        ignore=[],
        label_col=label_col
    )

    # 计算验证集和测试集数量
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.round(num_slides_cls * val_frac).astype(int)
    test_num = np.round(num_slides_cls * test_frac).astype(int)

    # 创建 splits
    split_dir = os.path.join('splits', f"{task}_{int(label_frac * 100)}")
    os.makedirs(split_dir, exist_ok=True)

    dataset.create_splits(k=k, val_num=val_num, test_num=test_num, label_frac=label_frac)

    for i in range(k):
        dataset.set_splits()
        descriptor_df = dataset.test_split_gen(return_descriptor=True)
        splits = dataset.return_splits(from_id=True)

        # 保存 splits
        save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, f'splits_{i}.csv'))
        save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, f'splits_{i}_bool.csv'), boolean_style=True)
        descriptor_df.to_csv(os.path.join(split_dir, f'splits_{i}_descriptor.csv'))

    print(f"Splits for task {task} have been regenerated and saved in {split_dir}")

if __name__ == '__main__':
    regenerate_splits()
