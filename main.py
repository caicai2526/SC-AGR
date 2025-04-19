# main.py

from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.utils import collate_MIL_padded
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

import logging
import sys

# Import CLAM model
from models.model_clam import CLAM_SB, CLAM_MB

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log")
    ]
)

logger = logging.getLogger('main')


def main(args, dataset):
    # Create results directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
        
    # Determine fold range
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    
    # Initialize model
    if args.model_type == 'clam_sb':
        clam_model = CLAM_SB(
            gate=not args.no_inst_cluster,
            size_arg=args.model_size,
            dropout=args.drop_out,
            k_sample=args.B,
            n_classes=args.n_classes,
            instance_loss_fn=get_loss_function(args.inst_loss),
            subtyping=args.subtyping,
            embed_dim=args.embed_dim
        )
    elif args.model_type == 'clam_mb':
        clam_model = CLAM_MB(
            gate=not args.no_inst_cluster,
            size_arg=args.model_size,
            dropout=args.drop_out,
            k_sample=args.B,
            n_classes=args.n_classes,
            instance_loss_fn=get_loss_function(args.inst_loss),
            subtyping=args.subtyping,
            embed_dim=args.embed_dim
        )
    else:
        raise NotImplementedError(f"Model type '{args.model_type}' is not implemented.")

    clam_model.to(device)
    
    # Define optimizer
    optimizer = get_optim(clam_model, args)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    for fold in folds:
        seed_torch(args.seed)
        # Load splits
        split_path = os.path.join(args.split_dir, f"splits_{fold}.csv")
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, 
            csv_path=split_path
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_MIL_padded)
        val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_MIL_padded)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_MIL_padded)
        
        for epoch in range(args.max_epochs):
            logger.info(f"Epoch [{epoch+1}/{args.max_epochs}] Fold [{fold}]")
            clam_model.train()
            
            for batch in train_loader:
                patch_features = batch['features'].to(device)  # shape: (B, N, D)
                slide_labels = batch['labels'].to(device)
                attn_mask = batch['mask'].to(device)
                
                # Model forward pass
                outputs, _, _, _, _ = clam_model(patch_features, label=slide_labels, instance_eval=False, attn_mask=attn_mask)
                loss = criterion(outputs, slide_labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log loss
                logger.info(f"Fold [{fold}] Epoch [{epoch+1}/{args.max_epochs}] Loss: {loss.item():.4f}")
            
            # Validation and testing (you need to implement evaluate function)
            # For the purpose of this code, we assume evaluate function exists
            # and returns val_results, val_auc, val_acc
            # Implement evaluate function accordingly
            # For now, we'll just append dummy values
            # Remove the following lines and implement actual evaluation
            val_auc = 0.0
            val_acc = 0.0
            test_auc = 0.0
            test_acc = 0.0
            all_test_auc.append(test_auc)
            all_val_auc.append(val_auc)
            all_test_acc.append(test_acc)
            all_val_acc.append(val_acc)
            
            # Save results
            filename = os.path.join(args.results_dir, f'split_{fold}_results.pkl')
            save_pkl(filename, {
                'val_auc': val_auc,
                'test_auc': test_auc,
                'val_acc': val_acc,
                'test_acc': test_acc
            })
    
    # Summarize results
    final_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_test_auc,
        'val_auc': all_val_auc,
        'test_acc': all_test_acc,
        'val_acc': all_val_acc
    })

    if len(folds) != args.k:
        save_name = f'summary_partial_{start}_{end}.csv'
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name), index=False)

    logger.info("################# Final Results ###################")
    logger.info(final_df)
    logger.info("Finished!")
    logger.info("End script")

# General training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')

# Update --task parameter to include new task options
parser.add_argument('--task', type=str, choices=[
    'camelyon16_plip',
    'camelyon16_R50',
    'TCGA_BC_PLIP',
    'TCGA_BC_R50',
    'TCGA_NSCLC_PLIP',
    'TCGA_NSCLC_R50'
], required=True, help='Task type')

parser.add_argument('--data_root_dir', type=str, required=True, 
                    help='Data directory (project root)')
parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='Maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='Fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='Weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='Start fold (default: -1, first fold)')
parser.add_argument('--k_end', type=int, default=-1, help='End fold (default: -1, last fold)')
parser.add_argument('--results_dir', default='./results', help='Results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='Manually specify the set of splits to use, ' 
                    +'instead of inferring from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='Log data using TensorBoard')
parser.add_argument('--testing', action='store_true', default=False, help='Debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam', help='Optimizer (default: adam)')
parser.add_argument('--drop_out', type=float, default=0.25, help='Dropout rate')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='Slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='Type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, required=True, help='Experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='Enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='Size of model, does not affect mil')

# CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='Disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', 'None'], default='None',
                     help='Instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='Subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='CLAM: Weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='Number of positive/negative patches to sample for CLAM')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

settings = {
    'num_splits': args.k, 
    'k_start': args.k_start,
    'k_end': args.k_end,
    'task': args.task,
    'max_epochs': args.max_epochs, 
    'results_dir': args.results_dir, 
    'lr': args.lr,
    'experiment': args.exp_code,
    'reg': args.reg,
    'label_frac': args.label_frac,
    'bag_loss': args.bag_loss,
    'seed': args.seed,
    'model_type': args.model_type,
    'model_size': args.model_size,
    "use_drop_out": args.drop_out,
    'weighted_sample': args.weighted_sample,
    'opt': args.opt
}

if args.model_type in ['clam_sb', 'clam_mb']:
    settings.update({
        'bag_weight': args.bag_weight,
        'inst_loss': args.inst_loss,
        'B': args.B
    })

print('\nLoad Dataset')

# Configure datasets for each task
if args.task == 'camelyon16_plip':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path=os.path.join(args.data_root_dir, 'dataset_csv', 'camelyon16_plip.csv'),
        data_dir=os.path.join(args.data_root_dir, 'RRT_data', 'camelyon16-diagnosis', 'CAMELYON16 PLIP', 'pt'),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={0: 0, 1: 1},  # Integer keys
        label_col='label',
        ignore=[]
    )
# ... [Additional task configurations] ...

# Create results directory
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# Update results directory path
args.results_dir = os.path.join(args.results_dir, f"{args.exp_code}_s{args.seed}")
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# Set split directory
if args.split_dir is None:
    args.split_dir = os.path.join('splits', f"{args.task}_{int(args.label_frac*100)}")
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir), f"Split directory does not exist: {args.split_dir}"

settings.update({'split_dir': args.split_dir})

# Save experiment settings to text file
experiment_settings_path = os.path.join(args.results_dir, f'experiment_{args.exp_code}.txt')
with open(experiment_settings_path, 'w') as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print(f"{key}:  {val}")        

if __name__ == "__main__":
    results = main(args, dataset) 
    print("Finished!")
    print("End script")
