import glob
import os
import random

import pandas as pd
import pickle
from tqdm import tqdm

from compy.datasets import dataset


def get_all_src_files(content_dir, file_ending='.c'):
    ret = []
    for root, dirs, files in os.walk(content_dir):
        for file in files:
            if file.endswith(file_ending):
                ret.append(os.path.join(root, file))
    return ret


class AnghabenchDataset(dataset.Dataset):
    def __init__(self):
        super().__init__()

        uri = "https://github.com/alexanderb14/AnghaBench.git"
        self.clone_git(uri)

    def get_size(self):
        print(self.content_dir)
        filenames = get_all_src_files(self.content_dir)
        return len(filenames)

    def load_graphs(self):
        graphs = []
        pickles = os.listdir(self.content_dir)
        for file in pickles:
            filename = f"{self.content_dir}{file}"
            with open(filename, "rb") as f:
                collection = pickle.load(f)
                for sample in collection:
                    graphs.append(sample)

        return {
            "samples": graphs,
            "num_types": 1
        }

    def load_graphs_whole(self):
        graphs = []
        pickle_files = os.listdir(self.content_dir)
        for pickle_file in pickle_files:
            file_path = f"{self.content_dir}{pickle_file}"
            with open(file_path, "rb") as file:
                graph = pickle.load(file)
                graphs.append(graph)

        print("done")
        return {
            "samples": [
                {
                    "x": {"code_rep": sample["code_rep"]},
                }
                for sample in graphs
            ]
        }

    def preprocess(self, builder, visitor, start_at=False, num_samples=None, randomly_select_samples=False):
        samples = {}

        filenames = get_all_src_files(self.content_dir, file_ending='.c')

        if start_at:
            filenames = filenames[start_at:]

        if num_samples:
            if randomly_select_samples:
                filenames = random.sample(filenames, num_samples)
            else:
                filenames = filenames[0:num_samples]

        for filename in tqdm(filenames, desc="Source Code -> IR+ -> Graph"):
            with open(filename, "rb") as f:
                source_code = f.read()

                try:
                    extractionInfo = builder.string_to_info(source_code)
                    sample = builder.info_to_representation(extractionInfo, visitor)

                    samples[filename] = sample
                except Exception as e:
                   print(e)
                   print('WARNING: Exception occurred while preprocessing sample: %s' % filename)

        return {
            "samples": [
                {
                    "info": info,
                    "x": {"code_rep": sample},
                }
                for info, sample in samples.items()
            ],
            "num_types": builder.num_tokens(),
        }
