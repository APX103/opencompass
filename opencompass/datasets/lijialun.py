import csv
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class LijialunDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        for split in ['dev']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    print("row: ", row)
                    raw_data.append({'input': [row], 'target': ["ok"]})
            dataset[split] = Dataset.from_list(raw_data)
        dataset['test'] = Dataset.from_list([{'input': ['1'], 'target': ["2"]}])
        return dataset
