import os
import re
import tqdm
import torch
import pickle
import numpy as np
import multiprocessing
from compy.datasets import IndexCache
from torch_geometric.data import Data
from torch_geometric.data import Dataset


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

    def __init__(self, non_empty=False):
        self.datafolder_name = "ExtractGraphsTask"
        self.root = self.get_basepath(os.environ.get('ANGHA_PICKLE_DIR'))
        self._setup_directories()
        self.graph_indexes_file_path = f"{self.dataset_info_path}/graphs_info.pickle"
        self.max_num_types_file_path = f"{self.dataset_info_path}/max_num_types.pickle"
        self.non_empty_samples_file_path = f"{self.dataset_info_path}/non_empty_samples.pickle"
        self.graph_indexes = []
        self.file_indexes = []
        self.non_empty_samples = []
        self.compute_indexes()
        self.num_types = self.get_num_types()
        self.index_cache = IndexCache(self.load_file, max_cache_size=1000)
        self.total_num_samples = len(self.graph_indexes)
        if self.total_num_samples != len(self.processed_file_names):
            self.parallel(num_processes=8)
        if non_empty:
            EMPTY_COUNT = 345035
            self.non_empty_samples = self.load_non_empty_sample()
            if self.total_num_samples - len(self.non_empty_samples) < EMPTY_COUNT:
                print("Recomputing empty samples")
                self.post_process()
                total = [self.get_file_idx_from_name(sample) for sample in self.processed_file_names]
                file = f"{self.root}/dataset_info/empty_sample_indexes.pickle"
                with open(file, "rb") as f:
                    empty_samples = pickle.load(f)
                    for sample in tqdm.tqdm(empty_samples):
                        total.remove(sample)
                    with open(self.non_empty_samples_file_path, 'wb') as f:
                        pickle.dump(total, f)
            print("Loading non empty samples")
            self.total_num_samples = len(self.non_empty_samples)
        if not non_empty:
            self.get = self.get_from_unfiltered

        super().__init__(self.root)

    def rem(self, total, empty):
        for index, sample in enumerate(empty):
            del total[sample - 1]
            if index > 100:
                return total

    def get_basepath(self, path):
        if self.datafolder_name in path:
            path = os.path.split(path)[0]
        return path

    def get_from_unfiltered(self, index):
        return torch.load(f"{self.processed_dir}/graph_{index}.pt")

    def len(self):
        return self.total_num_samples

    def get(self, index):
        return torch.load(f"{self.processed_dir}/graph_{self.non_empty_samples[index]}.pt")

    @property
    def total_branches(self):
        return torch.load(f"{self.dataset_info_path}/total_branch_count.pt")

    def store_graph(self, graph, index):
        torch.save(graph, f"{self.processed_dir}/graph_{index}.pt")

    @staticmethod
    def __process_data(self, data):
        return data

    @staticmethod
    def process_data(data):
        return [
            {
                "nodes": data["x"]["code_rep"].get_node_list(),
                "edges": data["x"]["code_rep"].get_edge_list_tuple(),
                "probability": data["x"]["code_rep"].get_edge_list_with_data(),
            }
            for data in data
        ]

    def generate_tensors(self, batch_graphs, hidden_size_orig, total_graph_count):
        previous_source_node = -1
        # Graph
        for graph_index, batch_graph in enumerate(batch_graphs):
            # Nodes
            one_hot = np.zeros(
                (len(batch_graph["nodes"]), hidden_size_orig)
            )
            one_hot[np.arange(len(batch_graph["nodes"])), batch_graph["nodes"]] = 1

            # Edges
            edge_index, edge_features, probability_list, source_nodes = [], [], [], []
            probability = "probability"
            for index, edge in enumerate(batch_graph["edges"]):
                last_element = batch_graph[probability][index][-1]
                edge_type = batch_graph[probability][index][1]
                source_node = edge[0]
                # edge_type = edge[1]
                edge_index.append([source_node, edge[2]])
                edge_features.append(edge_type)

                if probability in last_element and edge_type == 5:
                    if source_node == previous_source_node:
                        previous_idx = len(probability_list) - 1
                        probability_list[previous_idx] = (probability_list[previous_idx], last_element[probability])
                    else:
                        source_nodes.append(source_node)
                        probability_list.append(last_element[probability])
                previous_source_node = source_node

            # Probability Nodes
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_features = torch.tensor(edge_features, dtype=torch.long)

            source_nodes = np.array(source_nodes)
            drop_idxs, drop_nodes, edge_probabilities = self.get_filtered_probability_tensors(probability_list,
                                                                                              source_nodes)

            along_dimension = 0
            source_nodes = np.delete(source_nodes, drop_idxs, along_dimension)
            source_nodes = torch.tensor(source_nodes)

            x = torch.tensor(one_hot, dtype=torch.float)

            graph = Data(
                x=x,
                edge_index=edge_index.t().contiguous(),
                edge_features=edge_features,
                offset=len(x),
                source_nodes=source_nodes,
                source_node_count=len(source_nodes),
                y=edge_probabilities
            )

            self.store_graph(graph, graph_index + total_graph_count)

    @staticmethod
    def get_filtered_probability_tensors(probability_list, source_nodes):
        drop_indices = []
        drop_nodes = []
        edge_probabilities = []
        for idx, source_node_prob in enumerate(zip(source_nodes, probability_list)):
            source_node, prob = source_node_prob
            if prob == 100:
                drop_indices.append(idx)
                drop_nodes.append(source_node)
                continue
            edge_probabilities.append(prob)
        return drop_indices, drop_nodes, torch.tensor(edge_probabilities, dtype=torch.float) / 100

    @staticmethod
    def get_graph_count_from_file(filename):
        graph_count_pattern = "(\d+).\w+$"
        re_graph_count = re.compile(graph_count_pattern)
        graph_count = int(re_graph_count.search(filename).group(1))
        return graph_count

    def process(self, file_split=(0, 0, [])):
        _, start_graph_count, files = file_split
        if len(files) <= 0:
            start_graph_count = 0
            files = self.raw_file_names

        base_path = self.content_dir
        max_num_types = self.get_num_types()
        for index, filename in enumerate(tqdm.tqdm(files, desc="Preprocessing raw files")):
            file_path = f"{base_path}/{filename}"
            with open(file_path, "rb") as f:
                wrapped_graphs = pickle.load(f)
                batch_graphs = wrapped_graphs["samples"]
                batch_graphs = self.process_data(batch_graphs)
            self.generate_tensors(batch_graphs, max_num_types, start_graph_count)
            current_graph_count = self.get_graph_count_from_file(file_path)
            start_graph_count += current_graph_count

    @staticmethod
    def _split(num_splits, files):
        split_range, remainder = divmod(len(files), num_splits)
        return list(files[i * split_range + min(i, remainder):(i + 1) * split_range + min(i + 1, remainder)]
                    for i in range(num_splits))

    @staticmethod
    def get_file_idx_from_name(file):
        get_file_idx = re.compile("\d+")
        return int(re.search(get_file_idx, file).group())

    def filter_empty_sample(self, files):
        index, split = files
        non_empty = []
        for idx in tqdm.tqdm(split, total=len(split), desc=f"Processing {index}", disable=False):
            sample = self.get(idx)
            target = sample.y
            if len(target) > 0:
                continue
            non_empty.append(idx)
        return non_empty

    def post_process(self, files=[], empty_count=345035):
        threads = 8
        if len(files) <= 0:
            files = [self.get_file_idx_from_name(sample) for sample in self.processed_file_names]

        chunked_files = self._split(threads, files)
        tasks_split_indexed = [(index, split) for index, split in enumerate(chunked_files)]
        with multiprocessing.Pool(processes=threads) as pool:
            results = list(tqdm.tqdm(pool.imap(self.filter_empty_sample, tasks_split_indexed), total=len(files), desc="Processing...", disable=True))
        pool.close()
        pool.join()

        # Filter out empty samples
        from itertools import chain
        empty_sample_indexes = list(chain.from_iterable(results))
        filename = 'empty_sample_indexes.pickle'
        file_path = f"{self.root}/dataset_info/{filename}"
        with open(file_path, 'wb') as file:
            pickle.dump(empty_sample_indexes, file)

    def load_non_empty_sample(self):
        file_path = self.non_empty_samples_file_path
        is_file = os.path.exists(file_path)
        if is_file:
            with open(file_path, 'rb') as file:
                non_empty = pickle.load(file)
            return non_empty
        return []

    def parallel(self, num_processes=8):
        files = self.raw_file_names
        split_files = self._split(num_processes, files)

        graph_count_start_indexes = [sum(list(map(self.get_graph_count_from_file, split))) for split in split_files]
        graph_count_start_indexes = np.asarray(graph_count_start_indexes)
        graph_count_start_indexes = np.roll(graph_count_start_indexes, 1)
        graph_count_start_indexes[0] = 0
        graph_count_start_indexes = np.cumsum(graph_count_start_indexes)

        split_files_indexed = [(index, graph_count_start_indexes[index], split) for index, split in enumerate(split_files)]

        with multiprocessing.Pool(processes=num_processes) as pool:
            list(tqdm.tqdm(pool.imap(self.process, split_files_indexed), total=len(self.raw_file_names)))

    def download(self):
        pass

    @property
    def raw_file_names(self):
        raw_file_names = os.listdir(self.content_dir)
        sorting_pattern = "(\d+)"
        re_sorting = re.compile(sorting_pattern)
        numerical_sort = lambda file_name: int(re_sorting.match(file_name).group())
        raw_file_names.sort(key=numerical_sort)

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
        content_dir = os.path.join(self.root, self.datafolder_name)
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

        for num_file, file in enumerate(tqdm.tqdm(pickle_files, desc="Generating lookup table")):
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

    @staticmethod
    def _compute_num_types(self):
        max_num_types = 0
        pickle_files = os.listdir(self.content_dir)
        for file in tqdm.tqdm(pickle_files, desc="Computing max num_types of graphs..."):
            filename = f"{self.content_dir}/{file}"
            with open(filename, "rb") as f:
                collection = pickle.load(f)
                num_types = collection["num_types"]
                if num_types > max_num_types:
                    max_num_types = num_types
        self.__write_file(self.max_num_types_file_path, {"num_types": max_num_types})
        return max_num_types

    def __rename_pickle_files(self):
        pickle_files = os.listdir(self.content_dir)
        for file in tqdm.tqdm(pickle_files, desc="Relabeling pickle files..."):
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
