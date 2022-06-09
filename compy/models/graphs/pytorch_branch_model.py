import csv
import os
import time
import torch
import datetime
import numpy as np
from torch.nn import functional as F

from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.nn import GatedGraphConv

from compy.models.model import Model


class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        annotation_size = config["hidden_size_orig"]
        hidden_size = config["gnn_h_size"]
        n_steps = config["num_timesteps"]
        n_etypes = config["num_edge_types"]
        branch_count = 1

        self.reduce = nn.Linear(annotation_size, hidden_size)
        self.gconv = GatedGraphConv(hidden_size, n_steps)
        self.lin = nn.Linear(hidden_size, branch_count)

    def forward(
            self, graph, dimension=0,
    ):
        x, edge_index, batch, index = graph.x, graph.edge_index, graph.batch, graph.source_nodes
        indices = [node_idx for sublist in index for node_idx in sublist]
        idx = torch.LongTensor(indices)
        idx = idx.to(batch.device)

        x = self.reduce(x)
        x = self.gconv(x, edge_index)

        x = torch.index_select(x, dimension, idx)
        x = self.lin(x)
        x = torch.sigmoid(x)

        return x


class GnnPytorchBranchProbabilityModel(Model):
    def __init__(self, config=None, num_types=None):
        if not config:
            config = {
                "num_timesteps": 3,
                "hidden_size_orig": num_types,
                "gnn_h_size": 64,
                "learning_rate": 0.001,
                "batch_size": 64, # Maybe increase this size
                "num_epochs": 100,
                "num_edge_types": 1,
            }
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net(config)
        self.model = self.model.to(self.device)
        self.log_file = datetime.datetime.now().strftime("%d-%m-%Y--%H:%M:%S")
        self.training_logs = f"{os.path.expanduser('~')}/training-logs/"
        self.thresholds = (nn.Threshold(0.2, 0), nn.Threshold(0.1, 0), nn.Threshold(0.05, 0))
        if not os.path.exists(self.training_logs):
            os.mkdir(self.training_logs)

    def process_data(self, data):
        return [
            {
                "nodes": data["x"]["code_rep"].get_node_list(),
                "edges": data["x"]["code_rep"].get_edge_list_tuple(),
                "probability": data["x"]["code_rep"].get_edge_list_with_data(),
            }
            for data in data
        ]

    def get_thresholds(self):
        return sorted(self.thresholds, key=lambda item: item.threshold)

    def __build_pg_graphs(self, batch_graphs):
        pg_graphs = []
        total_node_count = 0
        previous_source_node = -1

        # Graph
        for graph_index, batch_graph in enumerate(batch_graphs):
            # Nodes
            one_hot = np.zeros(
                (len(batch_graph["nodes"]), self.config["hidden_size_orig"])
            )
            one_hot[np.arange(len(batch_graph["nodes"])), batch_graph["nodes"]] = 1
            x = torch.tensor(one_hot, dtype=torch.float)

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

            node_count = len(batch_graphs[graph_index - 1]["nodes"]) if graph_index > 0 else 0
            total_node_count += node_count
            source_nodes = [source_node + total_node_count for source_node in source_nodes]

            edge_probabilities = self.__get_probability_tensor(probability_list)

            graph = Data(
                x=x,
                edge_index=edge_index.t().contiguous(),
                edge_features=edge_features,
                source_nodes=source_nodes,
                y=edge_probabilities
            )

            pg_graphs.append(graph)

        return pg_graphs

    def __get_probability_tensor(self, probabilities):
        edge_probabilities = []
        for index, prob in enumerate(probabilities):
            if prob == 100:
                edge_probabilities.append([prob, 0])
                continue
            if prob == 0:
                edge_probabilities.append([prob, 100])
                continue
            edge_probabilities.append([prob[0], prob[1]])
        return torch.tensor(edge_probabilities, dtype=torch.float) / 100

    def _calculate_accuracy_with_threshold(self, errors, total_values):
        # Replace correct values with zero, to count them afterwards
        small_threshold, medium_threshold, large_threshold = self.get_thresholds()
        return {
            "small": (total_values - torch.count_nonzero(small_threshold(errors))) / total_values,
            "medium": (total_values - torch.count_nonzero(medium_threshold(errors))) / total_values,
            "large": (total_values - torch.count_nonzero(large_threshold(errors))) / total_values
        }

    def _train_init(self, data_train=None, data_valid=None):
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        if not data_train or data_valid:
            return
        return self.__process_data(data_train), self.__process_data(data_valid)

    def _test_init(self):
        self.model.eval()

    def _train_with_batch(self, batch):
        sample_loss = 0
        errors = 0
        euclidean_distance = 0
        loss_fn = F.mse_loss
        batch_size = self.config["batch_size"]

        graph = self.__build_pg_graphs(batch)
        loader = GeometricDataLoader(graph, batch_size=batch_size)

        for data in loader:
            data = data.to(self.device)

            self.model.train()
            self.opt.zero_grad()

            pred = self.model(data)
            pred_left = pred[:, 0]
            truth = data.y[:, 0]
            loss = loss_fn(pred_left, truth)

            loss.backward()
            self.opt.step()

            #TODO: Maybe get rid of batch_loss and return loss directly
            # since this for loop only has one iteration
            sample_loss += loss
            errors = torch.abs(truth - pred_left)
            accuracy = self._calculate_accuracy_with_threshold(errors, len(truth))
            euclidean_distance += sum(((truth - pred)**2).reshape(-1)).sqrt()

        length_dataset = len(loader.dataset)

        train_accuracies = accuracy
        train_loss = sample_loss / length_dataset
        euclidean_distance /= length_dataset

        return train_loss, train_accuracies, euclidean_distance, loss_fn.__name__

    def _predict_with_batch(self, batch):
        euclidean = 0
        sample_loss = 0
        loss_fn = F.mse_loss
        batch_size = self.config["batch_size"]

        graphs = self.__build_pg_graphs(batch)
        loader = GeometricDataLoader(graphs, batch_size=batch_size)
        for data in loader:
            data = data.to(self.device)

            with torch.no_grad():
                pred = self.model(data)
                size = pred.shape[0]
                if size <= 1:
                    continue
                pred_left = pred[:, 0]
                truth = data.y[:, 0]
                loss = loss_fn(pred_left, truth)
                sample_loss += loss
                if not data.y.nelement() > 0:
                    print("uh oh")
                truth = data.y[:, 0] if data.y.nelement() > 0 else data.y

            errors = torch.abs(truth - pred_left)
            accuracy = self._calculate_accuracy_with_threshold(errors, len(truth))
            euclidean += sum(((truth - pred)**2).reshape(-1)).sqrt()

        length_dataset = len(loader.dataset)
        valid_accuracy = accuracy
        sample_loss = sample_loss / length_dataset
        valid_euclidean = euclidean / length_dataset

        return valid_accuracy, sample_loss, valid_euclidean

    def write_to_log(self, log, encoding='UTF-8'):
        file_path = f"{self.training_logs}/{self.log_file}-{log['type']}.csv"

        if not os.path.isfile(file_path):
            self.create_log(file_path, log)

        with open(file_path, 'a', newline='', encoding=encoding) as file:
            writer = csv.writer(file)
            writer.writerow(log["data"])

    def create_log(self, file_name, log, encoding="UTF-8"):
        with open(file_name, 'a', newline='', encoding=encoding) as file:
            writer = csv.writer(file)
            writer.writerow(log["header"])

    def add_metrics(self, loss, accuracy, euclidean, batch_loss, batch_accuracy, batch_euclidean):
        loss += batch_loss
        euclidean += batch_euclidean
        accuracy["small"] += batch_accuracy["small"]
        accuracy["medium"] += batch_accuracy["medium"]
        accuracy["large"] += batch_accuracy["large"]
        return loss, accuracy, euclidean

    def train(self, data_train, data_valid):
        train_summary = []
        batch_size = self.config["batch_size"]

        self._train_init()
        train_loader = TorchDataLoader(dataset=data_train, batch_size=batch_size, collate_fn=self.process_data)
        test_loader = TorchDataLoader(dataset=data_valid, batch_size=batch_size, collate_fn=self.process_data)
        total_train_iterations = len(train_loader)
        total_test_iterations = len(test_loader)

        print()
        for epoch in range(self.config["num_epochs"]):
            train_loss = 0
            train_euclidean = 0
            train_accuracy = {
                "small": 0,
                "medium": 0,
                "large": 0
            }

            print("Epoch", epoch)
            # Train
            start_time = time.time()
            for index, batch in enumerate(train_loader):
                train_batch_loss, train_batch_accuracy, euclidean_distance, loss_fn = self._train_with_batch(batch)
                train_loss, train_accuracy, train_euclidean = self.add_metrics(train_loss, train_accuracy, train_euclidean, train_batch_loss, train_batch_accuracy, euclidean_distance)

                print(f"Training Iteration {index + 1}/{total_train_iterations} batch accuracy: {train_batch_accuracy['medium']}, batch loss: {train_batch_loss}, euclidean distance: {euclidean_distance}")
            end_time = time.time()

            # Valid
            self._test_init()

            valid_loss = 0
            valid_euclidean = 0
            valid_accuracy = {
                "small": 0,
                "medium": 0,
                "large": 0
            }

            for index, batch in enumerate(test_loader):
                batch_accuracy, batch_loss, euclidean_distance = self._predict_with_batch(batch)
                valid_loss, valid_accuracy, valid_euclidean = self.add_metrics(valid_loss, valid_accuracy, valid_euclidean, batch_loss, batch_accuracy, euclidean_distance)
                print(f"Validation Iteration {index + 1}/{total_test_iterations} batch accuracy: {batch_accuracy['medium']}, batch loss: {batch_loss}, euclidean distance: {euclidean_distance}")

            # Logging
            # -> Loss
            train_loss = train_loss / total_train_iterations
            valid_loss = valid_loss / total_test_iterations

            # -> Accuracy
            train_accuracy["small"] = train_accuracy["small"] / total_train_iterations
            train_accuracy["medium"] = train_accuracy["medium"] / total_train_iterations
            train_accuracy["large"] = train_accuracy["large"] / total_train_iterations

            valid_accuracy["small"] = valid_accuracy["small"] / total_test_iterations
            valid_accuracy["medium"] = valid_accuracy["medium"] / total_test_iterations
            valid_accuracy["large"] = valid_accuracy["large"] / total_test_iterations

            # -> Euclidean Distance
            train_euclidean = train_euclidean / total_train_iterations
            valid_euclidean = valid_euclidean / total_test_iterations

            instances_per_sec = len(data_train) / (end_time - start_time)

            training_log = {
                "type": "train",
                "header": ["epoch", f"train_loss_{loss_fn}", "small_threshold_train_accuracy", "medium_threshold_train_accuracy", "large_threshold_train_accuracy", "euclidean_distance", "train instances/sec"],
                "data": [epoch, train_loss.item(), train_accuracy["small"].item(), train_accuracy["medium"].item(), train_accuracy["large"].item(), train_euclidean.item(), instances_per_sec],
            }

            validation_log = {
                "type": "valid",
                "header": ["epoch", f"valid_loss_{loss_fn}", "small_threshold_valid_accuracy", "edium_threshold_valid_accuracy", "large_threshold_valid_accuracy", "euclidean_distance"],
                "data": [epoch, valid_loss.item(), valid_accuracy["small"].item(), valid_accuracy["medium"].item(), valid_accuracy["large"].item(), valid_euclidean.item()]
            }

            self.write_to_log(training_log)
            self.write_to_log(validation_log)

            print(
                "epoch: %i, train_loss: %.8f, train_accuracy: %.4f, valid_accuracy:"
                " %.4f, train instances/sec: %.2f"
                % (epoch, train_loss, train_accuracy["medium"], valid_accuracy["medium"], instances_per_sec)
            )

            train_summary.append({"train_accuracy": train_accuracy})
            train_summary.append({"valid_accuracy": valid_accuracy})

        return train_summary


def get_edge_types(graph):
    types = []
    for edge in graph["probability"]:
        value = edge[-1]["attr"]
        if value not in types:
            types.append(value)
    return types
