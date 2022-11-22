import os
import csv
import time
import datetime
import numpy as np
from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GatedGraphConv
from torch_geometric.loader import DataLoader as GeometricDataLoader

from compy.models.model import Model


class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        n_etypes = config["num_edge_types"]
        annotation_size = config["hidden_size_orig"]
        sequence_length = config["num_layers"]
        in_channel = config["gnn_h_size"]
        branch_count = 1

        self.reduce = nn.Linear(annotation_size, in_channel) #TODO: MLP -> Sequential(1,2,4,8 layer)
        self.gg_conv_1 = GatedGraphConv(in_channel, sequence_length)
        self.lin = nn.Linear(in_channel, branch_count) #TODO: MLP -> Sequential(1,2,4,8 layer)

    def forward(
            self, graph, dimension=0,
    ):
        x, edge_index, batch, index, offsets = graph.x, graph.edge_index, graph.batch, graph.source_nodes, graph.offset
        offsets = offsets.roll(1)
        offsets[0] = 0
        offsets = offsets.cumsum(dim=0)
        offset_indexes = torch.repeat_interleave(torch.arange(len(graph.offset)).cuda(), graph.source_node_count)
        offsets_per_index = torch.gather(offsets, dimension, offset_indexes)
        idx = torch.add(index, offsets_per_index)

        x = self.reduce(x)
        x = self.gg_conv_1(x, edge_index)
        x = torch.sigmoid(x)

        x = torch.index_select(x, dimension, idx)
        x = self.lin(x)
        x = torch.sigmoid(x)

        return x


class GnnPytorchBranchProbabilityModel(Model):
    def __init__(self, config=None, folder=None):
        default_path = f"{os.path.expanduser('~')}/training-logs/"
        self.date = datetime.datetime.now().strftime("%d-%m-%Y--%H:%M:%S")

        if folder:
            self.date = folder

        if not config:
            config = {
                "num_layers": 16,
                "hidden_size_orig": 80,
                "gnn_h_size": 80,
                "learning_rate": 0.001,
                "batch_size": 32,
                "num_epochs": 350,
                "num_edge_types": 5,
                "results_dir": default_path,
                "folder_name": self.date,
            }
        super().__init__(config)

        in_place_flag = True
        config["folder_name"] = self.date
        thresholds = namedtuple('thresholds', ['small', 'medium', 'large', 'binary'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net(config)
        self.model = self.model.to(self.device)
        self.log_file = f"{self.date}-layers-{config['num_layers']}-batch_size_{config['batch_size']}-hidden_size_{config['gnn_h_size']}_lr_{config['learning_rate']}"
        self.training_logs = default_path
        self.results_folder = os.path.join(config["results_dir"], self.date)
        self.state_dict_path = f"{self.date}_model_state_dict"
        self.optimizer_path = f"{self.date}_optimizer_state_dict"
        self.train_predictions = f"{self.results_folder}/train_predictions/"
        self.test_predictions = f"{self.results_folder}/test_predictions/"
        self.lr_scheduler = None
        self.cls_weights = None
        self.train_weights = config["train_weights"]
        self.test_weights = config["test_weights"]
        self.detect_missing_folders()
        self.thresholds = thresholds(
            nn.Threshold(0.05, 0, inplace=in_place_flag),
            nn.Threshold(0.1, 0, inplace=in_place_flag),
            nn.Threshold(0.2, 0, inplace=in_place_flag),
            nn.Threshold(0.5, 0, inplace=in_place_flag)
        )

    def detect_missing_folders(self):
        test_if_present = [
            self.results_folder,
            self.train_predictions,
            self.test_predictions
        ]

        for folder in test_if_present:
            if not os.path.exists(folder):
                os.mkdir(folder)
                print(f"Creating directory: {folder}")

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

            # Edges
            edge_index, edge_features, probability_list, source_nodes = [], [], [], []
            probability = "probability"
            for index, edge in enumerate(batch_graph["edges"]):
                last_element = batch_graph[probability][index][-1]
                edge_type = batch_graph[probability][index][1]
                source_node = edge[0]
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

        return torch.tensor(edge_probabilities, dtype=torch.float, device=self.device) / 100

    def _calculate_mispredicted_branches_with_threshold(self, errors):
        # Replace correct values with zero, to count them afterwards

        return [
            (torch.count_nonzero(self.thresholds.small(errors))).item(),
            (torch.count_nonzero(self.thresholds.medium(errors))).item(),
            (torch.count_nonzero(self.thresholds.large(errors))).item()
        ]

    @staticmethod
    def add_labels(branch_labels, new_labels):
        branch_labels[0] += new_labels[0]
        branch_labels[1] += new_labels[1]
        branch_labels[2] += new_labels[2]
        branch_labels[3] += new_labels[3]
        return branch_labels

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
        # if not data_train or data_valid:
        #     return
        # return self.__process_data(data_train), self.__process_data(data_valid)
        # x = F.softmax(x)
    def _test_init(self):
        self.model.eval()

    def _train_with_batch(self, batch, loss_fn, save_path, batch_nr=-1, epoch=0):
        train_loss = .0
        euclidean_distance = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        mispredicted_branches = [0., 0., 0.]
        batch_size = self.config["batch_size"]

        data = batch.to(self.device)
        self.model.train() #TODO: check if this is really necessary for every batch => move this line to init
        self.opt.zero_grad()

        pred = self.model(data)
        size = pred.shape[0]

        if size < 1:
            # FIXME checkout where the error might be coming from
            return train_loss, mispredicted_branches, euclidean_distance, (tp, tn, fp, fn)

        pred_left = pred[:, 0]
        truth = data.y[:, 0]
        loss = loss_fn(pred_left, truth, weights=self.train_weights)

        loss.backward()
        self.opt.step()

        train_loss += loss.item()
        errors = torch.abs(truth - pred_left)
        mispredicted_branches = self._calculate_mispredicted_branches_with_threshold(errors)
        tp, tn, fp, fn = self._collect_branch_labels(truth, pred_left)
        euclidean_distance += torch.sqrt(torch.sum(torch.square(truth - pred_left))).item()
        self.save_results(save_path, truth, pred_left, epoch, batch_nr)

        return train_loss, mispredicted_branches, euclidean_distance, (tp, tn, fp, fn)

    def _predict_with_batch(self, batch, loss_fn, save_path, batch_nr=-1, epoch=0):
        valid_loss = 0
        valid_euclidean = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        mispredicted_branches = [0., 0., 0.]

        data = batch.to(self.device)
        # TODO: check whether torch.no_grad is the same as model.eval()

        with torch.no_grad():
            pred = self.model(data)
            size = pred.shape[0]
            if size < 1:
                return mispredicted_branches, valid_loss, valid_euclidean, (tp, tn, fp, fn)
            pred_left = pred[:, 0]
            truth = data.y[:, 0]
            loss = loss_fn(pred_left, truth, weights=self.test_weights)
            valid_loss += loss.item()
            if not data.y.nelement() > 0:
                print("uh oh")
            truth = data.y[:, 0] if data.y.nelement() > 0 else data.y

        errors = torch.abs(truth - pred_left)
        mispredicted_branches = self._calculate_mispredicted_branches_with_threshold(errors)
        tp, tn, fp, fn = self._collect_branch_labels(truth, pred_left)
        valid_euclidean += torch.sqrt(torch.sum(torch.square(truth - pred_left))).item()
        self.save_results(save_path, truth, pred_left, epoch, batch_nr)

        return valid_loss, mispredicted_branches, valid_euclidean, (tp, tn, fp, fn)

    def write_to_log(self, log, encoding='UTF-8'):
        file_path = f"{self.results_folder}/{self.log_file}-{log['type']}.csv"

        if not os.path.isfile(file_path):
            self._create_log(file_path, log)

        with open(file_path, 'a', newline='', encoding=encoding) as file:
            writer = csv.writer(file)
            writer.writerow(log["data"])

    @staticmethod
    def _create_log(file_name, log, encoding="UTF-8"):
        with open(file_name, 'a', newline='', encoding=encoding) as file:
            writer = csv.writer(file)
            writer.writerow(log["header"])

    @staticmethod
    def _store_epoch_data(epoch, loss_fn, loss, accuracy, euclidean_distance, branch_labels, log_type='train', instances_per_sec=0):
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

    def save_model(self):
        torch.save(self.model.state_dict(), f"{self.results_folder}/{self.state_dict_path}")
        torch.save(self.opt.state_dict(), f"{self.results_folder}/{self.optimizer_path}")

    @staticmethod
    def save_results(result_path, ground_truth, prediction, epoch, batch_nr):
        torch.save(
            {"ground_truth": ground_truth, "prediciton": prediction},
            f"{result_path}/ep-{epoch}_batch-{batch_nr}.pt"
        )

    def initialize_training(self, epoch):
        model_path = f"{self.results_folder}/{self.state_dict_path}"
        optimizer_path = f"{self.results_folder}/{self.optimizer_path}"
        if os.path.isfile(model_path) and os.path.isfile(optimizer_path):
            print("Loading Model")
            self.model.load_state_dict(torch.load(model_path))
            self.opt.load_state_dict(torch.load(optimizer_path))
            self.model.eval()
        return self.set_epoch_range(epoch)

    def set_epoch_range(self, epoch):
        end = self.config["num_epochs"]
        end += epoch
        return range(epoch, end)

    @staticmethod
    def add_metrics(loss, accuracy, euclidean, batch_loss, batch_accuracy, batch_euclidean):
        loss += batch_loss
        euclidean += batch_euclidean
        accuracy[0] += batch_accuracy[0]  # accuracy with small threshold
        accuracy[1] += batch_accuracy[1]  # accuracy with medium threshold
        accuracy[2] += batch_accuracy[2]  # accuracy with large threshold
        return loss, accuracy, euclidean

    @staticmethod
    def compute_epoch_accuracy(train_accuracies, test_accuracies, train_loss, test_loss, train_euclidean, test_euclidean, train_dataset, test_dataset):
        total_train_branches = train_dataset.total_branches
        total_test_branches = test_dataset.total_branches
        # -> Loss
        train_loss /= total_train_branches
        test_loss /= total_test_branches

        # -> Accuracy
        for index in range(len(train_accuracies)):
            train_accuracies[index] = (total_train_branches - train_accuracies[index]) / total_train_branches
            test_accuracies[index] = (total_test_branches - test_accuracies[index]) / total_test_branches

        # -> Euclidean Distance
        train_euclidean /= total_train_branches
        test_euclidean /= total_test_branches

        return train_accuracies, test_accuracies, train_loss, test_loss, train_euclidean, test_euclidean

    def train(self, data_train, data_valid, epoch=0):
        train_summary = []
        valid_summary = []
        epoch_range = self.initialize_training(epoch)

        if len(data_train) < 2:
            self.log_file = f"{self.log_file}-single-sample_{data_train.indices[0]}"

        batch_size = self.config["batch_size"]
        loss_fn = weighted_mse
        self._train_init()
        train_loader = GeometricDataLoader(data_train, batch_size=batch_size, pin_memory=True)
        test_loader = GeometricDataLoader(data_valid, batch_size=batch_size, pin_memory=True)
        total_train_iterations = len(train_loader)
        total_test_iterations = len(test_loader)
        train_predition_path = self.train_predictions
        test_predition_path = self.test_predictions

        print(f"Total Epoch: {self.config['num_epochs']}, loss fn: {loss_fn.__name__}, batch size: {batch_size}")
        print()
        for epoch in epoch_range:
            train_loss = 0
            train_euclidean = 0
            train_accuracy = [0., 0., 0.]
            train_tp_fp_tn_fn_labels = [0, 0, 0, 0]

            print("Epoch", epoch)
            # Train
            start_time = time.time()
            for index, batch in enumerate(train_loader):
                batch_branches = len(batch.y)
                train_batch_loss, mispredicted_in_batch, euclidean_distance, train_batch_labels = self._train_with_batch(batch, loss_fn, train_predition_path, batch_nr=index, epoch=epoch)

                train_loss, train_accuracy, train_euclidean = self.add_metrics(train_loss, train_accuracy, train_euclidean, train_batch_loss, mispredicted_in_batch, euclidean_distance)
                train_tp_fp_tn_fn_labels = self.add_labels(train_tp_fp_tn_fn_labels, train_batch_labels)

                batch_accuracy = (batch_branches - mispredicted_in_batch[0]) / batch_branches
                batch_loss = train_batch_loss / batch_branches
                batch_euclidean = euclidean_distance / batch_branches

                print(f"Epoch: {epoch} - Training Iteration {index + 1}/{total_train_iterations} batch accuracy: {batch_accuracy}, batch loss: {batch_loss}, euclidean distance: {batch_euclidean}")

            end_time = time.time()

            # Valid
            self._test_init()

            valid_loss = 0
            valid_euclidean = 0
            valid_accuracy = [0., 0., 0.]
            valid_tp_fp_tn_fn_labels = [0, 0, 0, 0]

            for index, batch in enumerate(test_loader):
                batch_branches = len(batch.y)
                valid_batch_loss, mispredicted_in_batch, euclidean_distance, valid_batch_labels = self._predict_with_batch(batch, loss_fn, test_predition_path, batch_nr=index, epoch=epoch)

                valid_loss, valid_accuracy, valid_euclidean = self.add_metrics(valid_loss, valid_accuracy, valid_euclidean, valid_batch_loss, mispredicted_in_batch, euclidean_distance)
                valid_tp_fp_tn_fn_labels = self.add_labels(valid_tp_fp_tn_fn_labels, valid_batch_labels)

                batch_accuracy = (batch_branches - mispredicted_in_batch[0]) / batch_branches
                batch_loss = valid_loss / batch_branches
                batch_euclidean = euclidean_distance / batch_branches

                print(f"Epoch: {epoch} - Validation Iteration {index + 1}/{total_test_iterations} batch accuracy: {batch_accuracy}, batch loss: {batch_loss}, euclidean distance: {batch_euclidean}")

            train_accuracy, test_accuracy, train_loss, valid_loss, train_euclidean, valid_euclidean = self.compute_epoch_accuracy(train_accuracy, valid_accuracy, train_loss, valid_loss, train_euclidean, valid_euclidean, data_train, data_valid)
            instances_per_sec = len(data_train) / (end_time - start_time)

            training_log = self._store_epoch_data(epoch, loss_fn, train_loss, train_accuracy, train_euclidean, train_tp_fp_tn_fn_labels, log_type='train', instances_per_sec=instances_per_sec)
            validation_log = self._store_epoch_data(epoch, loss_fn, valid_loss, valid_accuracy, valid_euclidean, valid_tp_fp_tn_fn_labels, log_type='valid')

            self.write_to_log(training_log)
            self.write_to_log(validation_log)

            if epoch > 100 and epoch % 100 == 0 or epoch == epoch_range:
                print("Saving Model")
                self.save_model()

            print(
                f"epoch: {epoch}, train_loss: {train_loss}, train_accuracy: {train_accuracy[0]}, "
                f"valid_accuracy: {valid_accuracy}, train instances/sec: {instances_per_sec}"
            )

            train_summary.append(train_accuracy)
            valid_summary.append(valid_accuracy)

        return train_summary, valid_summary


def r2_score(prediction, ground_truth):
    # Sources:
    # https://pytorch.org/ignite/generated/ignite.contrib.metrics.regression.R2Score.html
    # https://en.wikipedia.org/wiki/Coefficient_of_determination

    ground_truth_mean = torch.mean(ground_truth)
    squares_sum_total = torch.sum(torch.pow((ground_truth - ground_truth_mean), 2))
    squares_sum_residual = torch.sum(torch.pow((ground_truth - prediction), 2))
    return torch.sub(1, torch.div(squares_sum_residual, squares_sum_total))


def weighted_mse(prediction, ground_truth, weights):
    indexes = torch.mul(ground_truth, 100).to(torch.int64)
    weight_mask = torch.gather(weights, dim=0, index=indexes)
    return torch.mean(weight_mask * torch.square(prediction - ground_truth))


def get_edge_types(graph):
    types = []
    for edge in graph["probability"]:
        value = edge[-1]["attr"]
        if value not in types:
            types.append(value)
    return types
