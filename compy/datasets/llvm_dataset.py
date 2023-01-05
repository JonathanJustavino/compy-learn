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


class LLVMDataset(dataset.Dataset):
    def __init__(self):
        super(LLVMDataset, self).__init__()

        uri = "https://github.com/llvm/llvm-project.git"
        self.clone_git(uri)
        self.additional_include_basepath = os.path.join(self.content_dir, "/tmp/pgo_clang/stage2-prof-gen")

    def get_size(self):
        print(self.content_dir)
        filenames = get_all_src_files(self.content_dir)
        return len(filenames)

    def set_content_dir(self, content_dir):
        print("Previous content dir:", self.content_dir)
        self.content_dir = content_dir
        print("Current content dir:", self.content_dir)

    @staticmethod
    def get_pickle_dir():
        pickle_dir = os.environ.get('LLVM_DATASET_DIR')
        return pickle_dir

    def load_graphs(self):
        graphs = []
        pickle_path = self.get_pickle_dir()
        debug_max_len = 500
        pickles = os.listdir(pickle_path)
        pickles = pickles[:debug_max_len]
        num_types = 0
        for file in tqdm(pickles, desc="Accumulating graphs"):
            filename = f"{pickle_path}{file}"
            with open(filename, "rb") as f:
                collection = pickle.load(f)
                if num_types < collection["num_types"]:
                    num_types = collection["num_types"]
                for sample in collection["samples"]:
                    graphs.append(sample)

        return {
            "samples": graphs,
            "num_types": num_types
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

    def get_include_paths(self):
        includes = []
        header_ext = ".h"
        for root, dirs, files in os.walk(self.additional_include_basepath):
            if any(header_ext in file for file in files):
                includes.append(root)
        return includes
