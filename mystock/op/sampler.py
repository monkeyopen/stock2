import random

from dotenv import load_dotenv
import os

load_dotenv()
CONF_PATH = os.getenv('CONF_PATH')
DATA_PATH = os.getenv('DATA_PATH')
ROOT_PATH = os.getenv('ROOT_PATH')
import sys

# print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([ROOT_PATH])

from torch.utils.data import Sampler


class CustomSampler(Sampler):
    def __init__(self, dataset, neg_sample_prob=0.5):
        self.dataset = dataset
        self.neg_sample_prob = neg_sample_prob
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        random.shuffle(self.indices)  # 随机打乱索引
        for idx, (data, label) in enumerate(self.dataset):
            if label == 0:  # 如果是负样本，按照指定的概率抽取
                if random.random() < self.neg_sample_prob:
                    yield idx
            else:  # 如果是正样本，全部抽取
                yield idx

    def __len__(self):
        return len(self.dataset)
