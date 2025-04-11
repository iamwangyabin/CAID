import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import hydra
from typing import List, Tuple, Any, Optional
from omegaconf import DictConfig


def build_train_val_dataloader(source_list: List[DictConfig],
                               loader_conf: DictConfig,
                               trsf: Any) -> Tuple[DataLoader, DataLoader]:

    datasets = []
    for source_conf in source_list:
        data_root = source_conf.data_root
        for sub_set in source_conf.sub_sets:
            dataset_class = hydra.utils.get_class(source_conf.target)
            dataset = dataset_class(data_root, trsf, 
                                    subset=sub_set, split=source_conf.split)
            datasets.append(dataset)

    concat_dataset = ConcatDataset(datasets)
    total_size = len(concat_dataset)
    
    size1 = int(total_size * loader_conf.split_ratio)
    size2 = total_size - size1
    dataset1, dataset2 = random_split(concat_dataset, [size1, size2],
                                        generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(dataset1,
                            batch_size=loader_conf.batch_size,
                            shuffle=True, 
                            num_workers=loader_conf.loader_workers,
                            pin_memory=True)
    

    val_batch_size = getattr(loader_conf, 'val_batch_size', loader_conf.batch_size)
    val_dataloader = DataLoader(dataset2,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=loader_conf.loader_workers,
                            pin_memory=True)
    return train_dataloader, val_dataloader


def build_test_dataloader(source_conf: DictConfig,
                          loader_conf: DictConfig,
                          trsf: Any) -> DataLoader:
    datasets = []
    dataset_class = hydra.utils.get_class(source_conf.target)
    for sub_set in source_conf.sub_sets:
        dataset = dataset_class(source_conf.data_root, trsf, subset=sub_set, 
                                split=source_conf.split)
        datasets.append(dataset)
    concat_dataset = ConcatDataset(datasets)
    test_dataloader = DataLoader(concat_dataset,
                                batch_size=loader_conf.batch_size,
                                shuffle=False, 
                                num_workers=loader_conf.loader_workers,
                                pin_memory=True)
    return test_dataloader

