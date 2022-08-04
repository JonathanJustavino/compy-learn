import csv
import os
import time
import torch
import datetime
import numpy as np
from collections import namedtuple
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
        branch_count = 2

        self.reduce = nn.Linear(annotation_size, hidden_size)
        self.gconv = GatedGraphConv(hidden_size, n_steps)
        self.lin = nn.Linear(hidden_size, branch_count)

    def forward(
            self, graph, dimension=0,
    ):
        #FIXME: This part might be ram intensive
        x, edge_index, batch, index = graph.x, graph.edge_index, graph.batch, graph.source_nodes
        indices = [node_idx for sublist in index for node_idx in sublist]
        idx = torch.LongTensor(indices)
        idx = idx.to(batch.device)

        x = self.reduce(x)
        x = self.gconv(x, edge_index)

        x = torch.index_select(x, dimension, idx)
        x = self.lin(x)
        x = torch.sigmoid(x)
        # x = torch.softmax(x)

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
                "num_epochs": 50,
                "num_edge_types": 1,
            }
        super().__init__(config)

        date = datetime.datetime.now().strftime("%d-%m-%Y--%H:%M:%S")
        thresholds = namedtuple('thresholds', ['small', 'medium', 'large', 'binary'])
        in_place_flag = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net(config)
        self.model = self.model.to(self.device)
        self.log_file = f"{date}-batch_size_{config['batch_size']}-hidden_size_{config['gnn_h_size']}_lr_{config['learning_rate']}"
        self.training_logs = f"{os.path.expanduser('~')}/training-logs/"
        self.thresholds = thresholds(
            nn.Threshold(0.05, 0, inplace=in_place_flag),
            nn.Threshold(0.1, 0, inplace=in_place_flag),
            nn.Threshold(0.2, 0, inplace=in_place_flag),
            nn.Threshold(0.5, 0, inplace=in_place_flag)
        )
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

    def __build_pg_graphs(self, batch_graphs):
        pg_graphs = []
        total_node_count = 0
        previous_source_node = -1

        # Graph
        for graph_index, batch_graph in enumerate(batch_graphs):

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
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
            edge_features = torch.tensor(edge_features, dtype=torch.long, device=self.device)

            node_count = len(batch_graphs[graph_index - 1]["nodes"]) if graph_index > 0 else 0
            total_node_count += node_count

            source_nodes = np.array(source_nodes)
            drop_idxs, drop_nodes, edge_probabilities = self.get_filtered_probability_tensors(probability_list, source_nodes)

            along_dimension = 0
            source_nodes = np.delete(source_nodes, drop_idxs, along_dimension)

            #1 determine 100 nodes  done
            #2 filter them out in source nodes AND one_hot
            #3 add offset to source nodes

            # Nodes
            one_hot = np.zeros(
                (len(batch_graph["nodes"]), self.config["hidden_size_orig"])
            )
            one_hot[np.arange(len(batch_graph["nodes"])), batch_graph["nodes"]] = 1

            # one_hot_without_single_branches = np.delete(one_hot, drop_nodes, along_dimension)

            x = torch.tensor(one_hot, dtype=torch.float, device=self.device)
            source_nodes = [source_node + total_node_count for source_node in source_nodes]

            graph = Data(
                x=x,
                edge_index=edge_index.t().contiguous(),
                edge_features=edge_features,
                source_nodes=source_nodes,
                y=edge_probabilities
            ).cuda()

            pg_graphs.append(graph)

        return pg_graphs

    def get_filtered_probability_tensors(self, probability_list, source_nodes):
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
        return drop_indices, drop_nodes, torch.tensor(edge_probabilities, dtype=torch.float, device=self.device) / 100

    def __get_probability_tensor(self, probabilities):
        def remove_single_branches(branch_value):
            return branch_value != 100

        filtered = filter(remove_single_branches, probabilities)
        edge_probabilities = list(filtered)

        # edge_probabilities = []
        # for index, prob in enumerate(probabilities):
        #     if prob == 100:
        #         edge_probabilities.append([prob, 0])
        #         continue
        #     if prob == 0:
        #         edge_probabilities.append([prob, 100])


        return torch.tensor(edge_probabilities, dtype=torch.float, device=self.device) / 100

    def _calculate_accuracy_with_threshold(self, errors, total_values):
        # Replace correct values with zero, to count them afterwards

        return [
            ((total_values - torch.count_nonzero(self.thresholds.small(errors))).item() / total_values),
            ((total_values - torch.count_nonzero(self.thresholds.medium(errors))).item() / total_values),
            ((total_values - torch.count_nonzero(self.thresholds.large(errors))).item() / total_values)
        ]

    # FIXME How can this be calculated
    def fixme_thresholded_branch_labels(self, truth, prediction):
        small_truth = self.thresholds.small(truth)
        medium_truth = self.thresholds.medium(truth)
        large_truth = self.thresholds.large(truth)
        binary_truth = self.thresholds.binary(truth)

        small_prediction = self.thresholds.small(prediction)
        medium_prediction = self.thresholds.medium(prediction)
        large_prediction = self.thresholds.large(prediction)
        binary_prediction = self.thresholds.binary(prediction)

        small_branch_labels = self._collect_branch_labels(small_truth, small_prediction)
        medium_branch_labels = self._collect_branch_labels(medium_truth, medium_prediction)
        large_branch_labels = self._collect_branch_labels(large_truth, large_prediction)
        binary_branch_labels = self._collect_branch_labels(binary_truth, binary_prediction)

    def _collect_branch_labels(self, truth_left, prediction, threshold=0.5):
        # 1. round everything > 0.5 to 1
        # 2. torch.where > 0. => true positives
        # 3. torch.where < 1. => false positives
        # 4. index_select true positive indexes in prediction
        # 5. index_select false positive indexes in prediction
        #FIXME figure out non-binary thresholds
        with torch.no_grad():
            one = torch.tensor(1.0, dtype=prediction.dtype).cuda()

            truth_left = torch.where(truth_left < threshold, truth_left, one)
            prediction = torch.where(prediction < threshold, prediction, one)

            left_branch_indexes = torch.where(truth_left == 1.)[0]
            right_branch_indexes = torch.where(truth_left != 1.)[0]

            should_predict_left = torch.index_select(prediction, 0, left_branch_indexes)
            should_predict_right = torch.index_select(prediction, 0, right_branch_indexes)

            true_positives = torch.where(should_predict_left == 1.)[0]
            false_negatives = torch.where(should_predict_left != 1.)[0]
            false_positives = torch.where(should_predict_right == 1.)[0]

            tp = len(true_positives)
            fp = len(false_positives)
            fn = len(false_negatives)
            tn = len(truth_left) - (tp + fp + fn)

            return tp, tn, fp, fn

    def _train_init(self, data_train=None, data_valid=None):
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        if not data_train or data_valid:
            return
        return self.__process_data(data_train), self.__process_data(data_valid)

    def _test_init(self):
        self.model.eval()

    def _train_with_batch(self, batch, loss_fn):
        sample_loss = 0
        euclidean_distance = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        train_accuracies = [0., 0., 0.]
        batch_size = self.config["batch_size"]

        graph = self.__build_pg_graphs(batch)
        loader = GeometricDataLoader(graph, batch_size=batch_size)

        for data in loader:
            # data = data.to(self.device)

            self.model.train()
            self.opt.zero_grad()

            pred = self.model(data)
            size = pred.shape[0]
            # FIXME checkout where the error might be coming from
            if size <= 1:
                continue
            pred_left = pred[:, 0]
            truth = data.y[:, 0]
            loss = loss_fn(pred_left, truth)
            # r2 = r2_score(pred_left, truth)

            loss.backward()
            self.opt.step()

            sample_loss += loss.item()
            errors = torch.abs(truth - pred_left)
            sample_count = len(truth)
            train_accuracies = self._calculate_accuracy_with_threshold(errors, sample_count)
            tp, tn, fp, fn = self._collect_branch_labels(truth, pred_left)
            exponent = 2
            euclidean_distance += torch.sqrt(torch.sum(torch.pow((truth - pred_left), exponent).view(-1))).item()

        length_dataset = len(loader.dataset)

        train_loss = sample_loss / length_dataset
        euclidean_distance /= length_dataset

        return train_loss, train_accuracies, euclidean_distance, (tp, tn, fp, fn)

    def _predict_with_batch(self, batch, loss_fn):
        sample_loss = 0
        euclidean_distance = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        valid_accuracies = [0., 0., 0.]
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
                sample_loss += loss.item()
                if not data.y.nelement() > 0:
                    print("uh oh")
                truth = data.y[:, 0] if data.y.nelement() > 0 else data.y

            errors = torch.abs(truth - pred_left)
            valid_accuracies = self._calculate_accuracy_with_threshold(errors, len(truth))
            tp, tn, fp, fn = self._collect_branch_labels(truth, pred_left)
            exponent = 2
            euclidean_distance += torch.sqrt(torch.sum(torch.pow((truth - pred_left), exponent).view(-1))).item()

        length_dataset = len(loader.dataset)
        sample_loss = sample_loss / length_dataset
        valid_euclidean = euclidean_distance / length_dataset

        return valid_accuracies, sample_loss, valid_euclidean, (tp, tn, fp, fn)

    def write_to_log(self, log, encoding='UTF-8'):
        file_path = f"{self.training_logs}/{self.log_file}-{log['type']}.csv"

        if not os.path.isfile(file_path):
            self._create_log(file_path, log)

        with open(file_path, 'a', newline='', encoding=encoding) as file:
            writer = csv.writer(file)
            writer.writerow(log["data"])

    def _create_log(self, file_name, log, encoding="UTF-8"):
        with open(file_name, 'a', newline='', encoding=encoding) as file:
            writer = csv.writer(file)
            writer.writerow(log["header"])

    def _store_epoch_data(self, epoch, loss_fn, loss, accuracy, euclidean_distance, branch_labels, log_type='train', instances_per_sec=0):
        header = [
            "epoch",
            f"loss_{loss_fn.__name__}",
            "small_threshold",
            "medium_threshold",
            "large_threshold",
            "euclidean_distance",
            "tp",
            "fp",
            "tn",
            "fn",
        ]

        small, medium, large = accuracy
        true_positive, false_positive, true_negative, false_negative = branch_labels

        data = [
            epoch,
            loss,
            small,
            medium,
            large,
            euclidean_distance,
            true_positive,
            false_positive,
            true_negative,
            false_negative
        ]

        if 'train' in log_type:
            header.append("train instances/sec")
            data.append(instances_per_sec)

        log = {
           "type": log_type,
           "header": header,
           "data": data
        }
        return log

    def add_metrics(self, loss, accuracy, euclidean, batch_loss, batch_accuracy, batch_euclidean):
        loss += batch_loss
        euclidean += batch_euclidean
        accuracy[0] += batch_accuracy[0]  # accuracy with small threshold
        accuracy[1] += batch_accuracy[1]  # accuracy with medium threshold
        accuracy[2] += batch_accuracy[2]  # accuracy with large threshold
        return loss, accuracy, euclidean

    def add_labels(self, branch_labels, new_labels):
        branch_labels[0] += new_labels[0]
        branch_labels[1] += new_labels[1]
        branch_labels[2] += new_labels[2]
        branch_labels[3] += new_labels[3]
        return branch_labels

    def train(self, data_train, data_valid):
        train_summary = []
        valid_summary = []
        batch_size = self.config["batch_size"]
        loss_fn = F.mse_loss

        self._train_init()
        train_loader = TorchDataLoader(dataset=data_train, batch_size=batch_size, collate_fn=self.process_data)
        test_loader = TorchDataLoader(dataset=data_valid, batch_size=batch_size, collate_fn=self.process_data)
        total_train_iterations = len(train_loader)
        total_test_iterations = len(test_loader)

        print()
        for epoch in range(self.config["num_epochs"]):
            train_loss = 0
            train_euclidean = 0
            train_accuracy = [0., 0., 0.]
            train_tp_fp_tn_fn_labels = [0, 0, 0, 0]

            print("Epoch", epoch)
            # Train
            start_time = time.time()
            for index, batch in enumerate(train_loader):
                train_batch_loss, train_batch_accuracy, euclidean_distance, batch_labels = self._train_with_batch(batch, loss_fn)
                train_loss, train_accuracy, train_euclidean = self.add_metrics(train_loss, train_accuracy, train_euclidean, train_batch_loss, train_batch_accuracy, euclidean_distance)
                train_tp_fp_tn_fn_labels = self.add_labels(train_tp_fp_tn_fn_labels, batch_labels)

                print(f"Training Iteration {index + 1}/{total_train_iterations} batch accuracy: {train_batch_accuracy[1]}, batch loss: {train_batch_loss}, euclidean distance: {euclidean_distance}")

            end_time = time.time()

            # Valid
            self._test_init()

            valid_loss = 0
            valid_euclidean = 0
            valid_accuracy = [0., 0., 0.]
            valid_tp_fp_tn_fn_labels = [0, 0, 0, 0]

            for index, batch in enumerate(test_loader):
                batch_accuracy, batch_loss, euclidean_distance, batch_labels = self._predict_with_batch(batch, loss_fn)
                valid_loss, valid_accuracy, valid_euclidean = self.add_metrics(valid_loss, valid_accuracy, valid_euclidean, batch_loss, batch_accuracy, euclidean_distance)
                valid_tp_fp_tn_fn_labels = self.add_labels(valid_tp_fp_tn_fn_labels, batch_labels)
                print(f"Validation Iteration {index + 1}/{total_test_iterations} batch accuracy: {batch_accuracy[1]}, batch loss: {batch_loss}, euclidean distance: {euclidean_distance}")

            # Logging
            # -> Loss
            train_loss = train_loss / total_train_iterations
            valid_loss = valid_loss / total_test_iterations

            # -> Accuracy
            train_accuracy[0] = train_accuracy[0] / total_train_iterations
            train_accuracy[1] = train_accuracy[1] / total_train_iterations
            train_accuracy[2] = train_accuracy[2] / total_train_iterations

            valid_accuracy[0] = valid_accuracy[0] / total_test_iterations
            valid_accuracy[1] = valid_accuracy[1] / total_test_iterations
            valid_accuracy[2] = valid_accuracy[2] / total_test_iterations

            # -> Euclidean Distance
            train_euclidean = train_euclidean / total_train_iterations
            valid_euclidean = valid_euclidean / total_test_iterations

            instances_per_sec = len(data_train) / (end_time - start_time)

            training_log = self._store_epoch_data(epoch, loss_fn, train_loss, train_accuracy, train_euclidean, train_tp_fp_tn_fn_labels, log_type='train', instances_per_sec=instances_per_sec)
            validation_log = self._store_epoch_data(epoch, loss_fn, valid_loss, train_accuracy, valid_euclidean, valid_tp_fp_tn_fn_labels, log_type='valid')

            self.write_to_log(training_log)
            self.write_to_log(validation_log)

            print(
                "epoch: %i, train_loss: %.8f, train_accuracy: %.4f, valid_accuracy:"
                " %.4f, train instances/sec: %.2f"
                % (epoch, train_loss, train_accuracy[1], valid_accuracy[1], instances_per_sec)
            )

            # train_summary.append(train_accuracy)
            # valid_summary.append(valid_accuracy)
            train_summary ="Finished"

        return train_summary


def r2_score(prediction, ground_truth):
    # Sources:
    # https://pytorch.org/ignite/generated/ignite.contrib.metrics.regression.R2Score.html
    # https://en.wikipedia.org/wiki/Coefficient_of_determination

    ground_truth_mean = torch.mean(ground_truth)
    squares_sum_total = torch.sum(torch.pow((ground_truth - ground_truth_mean), 2))
    squares_sum_residual = torch.sum(torch.pow((ground_truth - prediction), 2))
    return torch.sub(1, torch.div(squares_sum_residual, squares_sum_total))


def get_edge_types(graph):
    types = []
    for edge in graph["probability"]:
        value = edge[-1]["attr"]
        if value not in types:
            types.append(value)
    return types
