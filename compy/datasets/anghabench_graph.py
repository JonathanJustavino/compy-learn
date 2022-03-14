import os
import re
import torch
import pickle
import numpy as np
from tqdm import tqdm
from compy.datasets import IndexCache
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric import transforms as T


class AnghabenchGraphDataset(Dataset):
    """
    attributes:
    - content_dir: directory of pickle files
    - sample_inventory: array of file infos
        - file_idx: index of pickle file
        - num_samples: amount of samples in pickle file
        - sample_offset: sum of amount of samples of previous files
    - total_num_samples: amount of entire samples in dataset
    """

    def __init__(self):
        self.root = os.environ.get('ANGHA_PICKLE_DIR')
        self._setup_directories()
        self.graph_indexes_file_path = f"{self.dataset_info_path}/graphs_info.pickle"
        self.max_num_types_file_path = f"{self.dataset_info_path}/max_num_types.pickle"
        self.graph_indexes = []
        self.file_indexes = []
        self.compute_indexes()
        self.num_types = self.get_num_types()
        self.total_num_samples = len(self.graph_indexes)
        self.queue_size = 10
        self.index_cache = IndexCache(self.load_file)
        super().__init__(self.root, self.__process_data)

    def len(self):
        return self.total_num_samples

    def get(self, idx):
        file_index = self.graph_indexes[idx]
        file_offset, _ = self.file_indexes[file_index]
        data = self.index_cache.get(file_index)
        return data["samples"][idx - file_offset]

    def process(self):
        pass

    def download(self):
        pass

    @property
    def raw_file_names(self):
        raw_file_names = os.listdir(self.content_dir)
        if len(raw_file_names) < 1:
            raise Exception("Raw Files are missing.", raw_file_names)
        return raw_file_names

    @property
    def processed_file_names(self):
        processed_file_names = os.listdir(self.processed_dir)
        if len(processed_file_names) < 1:
            return []
            # raise Exception("Processed Files are missing.", processed_file_names)
        return processed_file_names

    def get_num_types(self):
        if os.path.isfile(self.max_num_types_file_path):
            with open(self.max_num_types_file_path, "rb") as file:
                data = pickle.load(file)
            return data["num_types"]
        return self._compute_num_types()

    def compute_indexes(self):
        indexes = self.__get_indexes()
        self.graph_indexes = indexes["graph_indexes"]
        self.file_indexes = indexes["file_indexes"]

    def load_file(self, file_index):
        _, graph_count = self.file_indexes[file_index]
        filename = self.get_file_name(file_index, graph_count)
        with open(filename, "rb") as file:
            return pickle.load(file)

    def _setup_directories(self):
        self.__set_content_dir()
        _ = self.__create_directory("processed")
        path = self.__create_directory("dataset_info")
        self.dataset_info_path = path

    def __set_content_dir(self):
        content_dir = os.path.join(self.root, "ExtractGraphsTask")
        if os.path.exists(content_dir):
            self.content_dir = content_dir
            return
        raise Exception("Dataset content dir does not exist.")

    def __create_directory(self, dir):
        directory = os.path.join(self.root, dir)
        if not os.path.exists(directory):
            print(f"Creating {dir} directory for Anghabench Dataset: {directory}")
            os.mkdir(directory)
        else:
            print(f"Anghabench Dataset {dir} path exists: {os.path.exists(directory)}")
        return directory

    def __get_indexes(self):
        if os.path.isfile(self.graph_indexes_file_path):
            print("Loading index file")
            with open(self.graph_indexes_file_path, 'rb') as file:
                return pickle.load(file)
        print("Computing indexes")
        return self.__create_index_tables()

    def __create_index_tables(self):
        file_offset = 0
        pickle_files = os.listdir(self.content_dir)

        graph_indexes = np.empty(0, int)
        file_indexes = np.empty((len(pickle_files), 2), int)

        sorting_pattern = "(\d+)"
        re_sorting = re.compile(sorting_pattern)
        numerical_sort = lambda file_name: int(re_sorting.match(file_name).group())
        pickle_files.sort(key=numerical_sort)

        graph_count_pattern = "\d+\D+(\d+)"
        re_graph_count = re.compile(graph_count_pattern)
        get_count = lambda file_name: int(re_graph_count.match(file_name).group(1))

        for num_file, file in enumerate(tqdm(pickle_files, desc="Generating lookup table")):
            graph_count = get_count(file)
            file_indexes[num_file] = [file_offset, graph_count]
            file_offset += graph_count
            graph_indexes = np.concatenate((graph_indexes, np.repeat(num_file, graph_count)), axis=0)

        indexes = {
            "graph_indexes": graph_indexes,
            "file_indexes": file_indexes
        }

        self.__write_file(
            self.graph_indexes_file_path,
            indexes
        )
        return indexes

    def __process_data(self, data):
        return data

    def _compute_num_types(self):
        max_num_types = 0
        pickle_files = os.listdir(self.content_dir)
        for file in tqdm(pickle_files, desc="Computing max num_types of graphs..."):
            filename = f"{self.content_dir}/{file}"
            with open(filename, "rb") as f:
                collection = pickle.load(f)
                num_types = collection["num_types"]
                if num_types > max_num_types:
                    max_num_types = num_types
        self.__write_file(self.max_num_types_file_path)
        return max_num_types

    def __rename_pickle_files(self):
        pickle_files = os.listdir(self.content_dir)
        for file in tqdm(pickle_files, desc="Relabeling pickle files..."):
            filename = f"{self.content_dir}/{file}"
            if "graph_count" in filename:
                continue
            with open(filename, "rb") as f:
                collection = pickle.load(f)
                graph_count = len(collection["samples"])
                new_filename, file_extension = filename.rsplit(".", 1)
                new_filename = f"{new_filename}_graph_count_{graph_count}.{file_extension}"
                os.rename(filename, new_filename)

    @staticmethod
    def __write_file(file_path, data):
        with open(file_path, "wb") as file:
            pickle.dump(data, file)

    def get_file_name(self, graph_index, graph_count):
        return f"{self.content_dir}/{graph_index}_graph_count_{graph_count}.pickle"
