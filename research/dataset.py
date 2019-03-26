
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf



class GenderCoref(object):

    def __init__(self):
        self.data_dir = "/home/guo/Github/xiaoguoguo/research/data/gap-coreference/"
        self.tmp_dir = self.data_dir + "tmp/"

    def text_preprocess(self, text):
        return text
        
    def generate_dataset(self):
        train_df = pd.read_csv(self.data_dir + "gap-development.tsv", delimiter="\t")
        nrow, ncol = train_df.shape
        for ind in range(nrow):
            example_raw = train_df.iloc[ind,:]
            print(example)
            break
        

if __name__ == "__main__":
    gc = GenderCoref()
    gc.generate_dataset()

