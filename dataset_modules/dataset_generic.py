import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
import glob

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth
import logging

# Get logger
logger = logging.getLogger(__name__)

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])

    df.to_csv(filename, index=False)  # Ensure not to save index column
    logger.info(f"Splits saved to {filename}")

class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_dict={},
                 filter_dict={},
                 ignore=[],
                 patient_strat=False,
                 label_col=None,
                 patient_voting='max'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        if not label_col:
            label_col = 'label'
        self.label_col = label_col

        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

        ### Shuffle data
        if shuffle:
            np.random.seed(seed)
            slide_data = slide_data.sample(frac=1).reset_index(drop=True)
            logger.info(f"Data shuffled with seed {seed}")

        self.slide_data = slide_data

        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        # Store ids corresponding each class at the patient or case level
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        # Store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self, patient_voting='max'):
        logger.debug(f"Processing patient_voting: {patient_voting}")
        # Replace 'case_id' with 'slide_id'
        patients = np.unique(np.array(self.slide_data['slide_id']))  # Use slide_id instead of case_id
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['slide_id'] == p].index.tolist()
            assert len(locations) > 0, f"No data found for slide_id: {p}"
            label = self.slide_data['label'][locations].values
            logger.debug(f"Patient: {p}, Labels before voting: {label}")
            if patient_voting == 'max':
                label = label.max()  # Get patient label (MIL rule)
            elif patient_voting == 'maj':
                # Use np.unique instead of stats.mode
                values, counts = np.unique(label, return_counts=True)
                label = values[np.argmax(counts)]
            else:
                raise NotImplementedError(f"Unsupported patient_voting method: {patient_voting}")
            logger.debug(f"Patient: {p}, Label after voting: {label}")
            patient_labels.append(label)

        self.patient_data = {'slide_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()  # Create 'label' column if needed
        # Filter out any rows where label is 'label' to avoid misreading header as data
        data = data[data['label'] != 'label']
        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            assert key in label_dict, f"Label '{key}' not found in label_dict"
            data.at[i, 'label'] = label_dict[key]  # Map labels to values in label_dict
        # Add debug info
        logger.debug(f"Unique labels after mapping: {data['label'].unique()}")
        logger.debug(f"Data type of labels: {data['label'].dtype}")
        data['label'] = data['label'].astype(int)  # Ensure labels are integer type
        return data

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
            logger.debug(f"Filtered data with {filter_dict}, resulting in {len(df)} records.")
        return df

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['slide_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        logger.info(f"Label column: {self.label_col}")
        logger.info(f"Label dictionary: {self.label_dict}")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Slide-level counts:\n{self.slide_data['label'].value_counts(sort=False)}")
        for i in range(self.num_classes):
            logger.info(f'Patient-LVL; Number of samples registered in class {i}: {self.patient_cls_ids[i].shape[0]}')
            logger.info(f'Slide-LVL; Number of samples registered in class {i}: {self.slide_cls_ids[i].shape[0]}')

    def create_splits(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0, custom_test_ids=None):
        settings = {
            'n_splits': k,
            'val_num': val_num,
            'test_num': test_num,
            'label_frac': label_frac,
            'seed': self.seed,
            'custom_test_ids': custom_test_ids
        }

        if self.patient_strat:
            settings.update({'cls_ids': self.patient_cls_ids, 'samples': len(self.patient_data['slide_id'])})
        else:
            settings.update({'cls_ids': self.slide_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)
        logger.debug(f"Created splits with settings: {settings}")

    def set_splits(self, start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)
        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))]

            for split in range(len(ids)):
                for idx in ids[split]:
                    slide_id = self.patient_data['slide_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['slide_id'] == slide_id].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
            logger.debug(f"Set splits: train_ids={len(self.train_ids)}, val_ids={len(self.val_ids)}, test_ids={len(self.test_ids)}")
        else:
            self.train_ids, self.val_ids, self.test_ids = ids
            logger.debug(f"Set splits: train_ids={len(self.train_ids)}, val_ids={len(self.val_ids)}, test_ids={len(self.test_ids)}")

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
            logger.debug(f"Created {split_key} split with {len(split)} samples.")
        else:
            split = None
            logger.warning(f"No samples found for split: {split_key}")

        return split

    def get_merged_split_from_df(self, all_splits, split_keys=['train']):
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)

        if len(merged_split) > 0:
            mask = self.slide_data['slide_id'].isin(merged_split)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
            logger.debug(f"Merged split with {len(split)} samples.")
        else:
            split = None
            logger.warning(f"No samples found for merged splits: {split_keys}")

        return split

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)
                logger.debug(f"Returned train split with {len(train_split)} samples.")
            else:
                train_split = None
                logger.warning("No training samples available.")

            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)
                logger.debug(f"Returned validation split with {len(val_split)} samples.")
            else:
                val_split = None
                logger.warning("No validation samples available.")

            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
                logger.debug(f"Returned test split with {len(test_split)} samples.")
            else:
                test_split = None
                logger.warning("No test samples available.")
        else:
            assert csv_path, "csv_path must be provided if from_id is False."
            all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
            train_split = self.get_split_from_df(all_splits, 'train')
            val_split = self.get_split_from_df(all_splits, 'val')
            test_split = self.get_split_from_df(all_splits, 'test')

        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def test_split_gen(self, return_descriptor=False):
        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index=index, columns=columns)

        count = len(self.train_ids)
        logger.info(f'\nNumber of training samples: {count}')
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            logger.info(f'Number of samples in class {unique[u]}: {counts[u]}')
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]

        count = len(self.val_ids)
        logger.info(f'\nNumber of validation samples: {count}')
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            logger.info(f'Number of samples in class {unique[u]}: {counts[u]}')
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids)
        logger.info(f'\nNumber of test samples: {count}')
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            logger.info(f'Number of samples in class {unique[u]}: {counts[u]}')
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1)
        df.to_csv(filename, index=False)  # Ensure not to save index column
        logger.info(f"Split saved to {filename}")

class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, data_dir, **kwargs):
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        if isinstance(self.data_dir, dict):
            source = self.slide_data['source'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if data_dir:
                # Use glob to find matching .pt files
                pattern = os.path.join(data_dir, f"{slide_id}*.pt")
                matched_files = glob.glob(pattern)

                if len(matched_files) == 0:
                    logger.error(f"No .pt file found for slide_id: {slide_id}")
                    raise FileNotFoundError(f"No .pt file found for slide_id: {slide_id}")
                elif len(matched_files) > 1:
                    logger.warning(f"Multiple .pt files found for slide_id: {slide_id}, loading and concatenating them.")
                    features_list = []
                    for file in matched_files:
                        try:
                            features = torch.load(file, weights_only=True)
                            assert features.shape[1] == 1024 or features.shape[1] == 512, f"Feature dimension mismatch in {file}"
                            features_list.append(features)
                        except Exception as e:
                            logger.error(f"Error loading file {file}: {e}")
                            raise e
                    try:
                        features = torch.cat(features_list, dim=0)
                    except Exception as e:
                        logger.error(f"Error concatenating features for slide_id: {slide_id}: {e}")
                        raise e
                else:
                    full_path = matched_files[0]
                    try:
                        features = torch.load(full_path, weights_only=True)
                    except Exception as e:
                        logger.error(f"Error loading file {full_path}: {e}")
                        raise e

                return {'features': features, 'labels': label}
            else:
                return {'features': slide_id, 'labels': label}
        else:
            full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
            try:
                with h5py.File(full_path, 'r') as hdf5_file:
                    features = hdf5_file['features'][:]
                    coords = hdf5_file['coords'][:]
            except Exception as e:
                logger.error(f"Error loading h5 file {full_path}: {e}")
                raise e

            features = torch.from_numpy(features)
            return {'features': features, 'labels': label, 'coords': coords}

class Generic_Split(Dataset):
    def __init__(self, slide_data, data_dir=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        if isinstance(self.data_dir, dict):
            source = self.slide_data['source'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if data_dir:
                # Use glob to find matching .pt files
                pattern = os.path.join(data_dir, f"{slide_id}*.pt")
                matched_files = glob.glob(pattern)

                if len(matched_files) == 0:
                    logger.error(f"No .pt file found for slide_id: {slide_id}")
                    raise FileNotFoundError(f"No .pt file found for slide_id: {slide_id}")
                elif len(matched_files) > 1:
                    logger.warning(f"Multiple .pt files found for slide_id: {slide_id}, loading and concatenating them.")
                    features_list = []
                    for file in matched_files:
                        try:
                            features = torch.load(file, weights_only=True)
                            assert features.shape[1] == 1024 or features.shape[1] == 512, f"Feature dimension mismatch in {file}"
                            features_list.append(features)
                        except Exception as e:
                            logger.error(f"Error loading file {file}: {e}")
                            raise e
                    try:
                        features = torch.cat(features_list, dim=0)
                    except Exception as e:
                        logger.error(f"Error concatenating features for slide_id: {slide_id}: {e}")
                        raise e
                else:
                    full_path = matched_files[0]
                    try:
                        features = torch.load(full_path, weights_only=True)
                    except Exception as e:
                        logger.error(f"Error loading file {full_path}: {e}")
                        raise e

                return {'features': features, 'labels': label}
            else:
                return {'features': slide_id, 'labels': label}
        else:
            full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
            try:
                with h5py.File(full_path, 'r') as hdf5_file:
                    features = hdf5_file['features'][:]
                    coords = hdf5_file['coords'][:]
            except Exception as e:
                logger.error(f"Error loading h5 file {full_path}: {e}")
                raise e

            features = torch.from_numpy(features)
            return {'features': features, 'labels': label, 'coords': coords}
