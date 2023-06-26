from flwr.common import MetricsAggregationFn, NDArrays, Parameters, Scalar
import numpy as np
import torch
import pickle
import logging
import torch.nn as nn
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import flwr as fl


class GCN(nn.Module):
    """
    Graph Convolutional Network based on https://arxiv.org/abs/1609.02907

    """

    def __init__(self,
                 feat_dim,
                 hidden_dim1,
                 hidden_dim2,
                 dropout,
                 is_sparse=False):
        """Dense version of GAT."""
        super(GCN, self).__init__()
        # self.dropout = dropout
        self.W1 = nn.Parameter(torch.FloatTensor(feat_dim, hidden_dim1))
        self.W2 = nn.Parameter(torch.FloatTensor(hidden_dim1, hidden_dim2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.W1.data)
        nn.init.xavier_uniform_(self.W2.data)

        self.is_sparse = is_sparse

    def forward(self, x, adj):
        # Layer 1
        support = torch.mm(x, self.W1)
        embeddings = (torch.sparse.mm(adj, support)
                      if self.is_sparse else torch.mm(adj, support))

        embeddings = self.dropout(embeddings)

        # Layer 2
        support = torch.mm(embeddings, self.W2)
        embeddings = (torch.sparse.mm(adj, support)
                      if self.is_sparse else torch.mm(adj, support))

        return embeddings


class Readout(nn.Module):
    """
    This module learns a single graph level representation for a molecule given GNN generated node embeddings
    """

    def __init__(self, attr_dim, embedding_dim, hidden_dim, output_dim,
                 num_cats):
        super(Readout, self).__init__()
        self.attr_dim = attr_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_cats = num_cats

        self.layer1 = nn.Linear(attr_dim + embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Linear(output_dim, num_cats)
        self.act = nn.ReLU()

    def forward(self, node_features, node_embeddings):
        combined_rep = torch.cat(
            (node_features, node_embeddings),
            dim=1)  # Concat initial node attributed with embeddings from sage
        hidden_rep = self.act(self.layer1(combined_rep))
        graph_rep = self.act(
            self.layer2(hidden_rep))  # Generate final graph level embedding

        logits = torch.mean(
            self.output(graph_rep),
            dim=0)  # Generated logits for multilabel classification

        return logits


class GcnMoleculeNet(nn.Module):
    """
    Network that consolidates GCN + Readout into a single nn.Module
    """

    def __init__(
        self,
        feat_dim,
        hidden_dim,
        node_embedding_dim,
        dropout,
        readout_hidden_dim,
        graph_embedding_dim,
        num_categories,
        sparse_adj=False,
    ):
        super(GcnMoleculeNet, self).__init__()
        self.gcn = GCN(feat_dim,
                       hidden_dim,
                       node_embedding_dim,
                       dropout,
                       is_sparse=sparse_adj)
        self.readout = Readout(
            feat_dim,
            node_embedding_dim,
            readout_hidden_dim,
            graph_embedding_dim,
            num_categories,
        )

    def forward(self, adj_matrix, feature_matrix):
        node_embeddings = self.gcn(feature_matrix, adj_matrix)
        logits = self.readout(feature_matrix, node_embeddings)
        return logits


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


args = {}
args["device"] = "cuda:0"
args["client_optimizer"] = "sgd"
args["learning_rate"] = 0.001
args["epochs"] = 5
args["frequency_of_the_test"] = 2
args["metric"] = "auc"

f = open("train_data_local_dict.pkl", "rb")
trainloader = pickle.load(f)


def train(model, train_data, test_data, device, args):
    logging.info("----------train--------")
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    if args["client_optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args["learning_rate"])
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args["learning_rate"])

    max_test_score = 0
    best_model_params = {}

    for epoch in range(args["epochs"]):
        for mol_idxs, (adj_matrix, feature_matrix, label,
                       mask) in enumerate(train_data):
            if torch.all(mask == 0).item():
                continue

            optimizer.zero_grad()

            adj_matrix = adj_matrix.to(device=device,
                                       dtype=torch.float32,
                                       non_blocking=True)
            feature_matrix = feature_matrix.to(device=device,
                                               dtype=torch.float32,
                                               non_blocking=True)

            label = label.to(device=device,
                             dtype=torch.float32,
                             non_blocking=True)
            mask = mask.to(device=device,
                           dtype=torch.float32,
                           non_blocking=True)

            logits = model(adj_matrix, feature_matrix)
            loss = criterion(logits, label) * mask
            loss = loss.sum() / mask.sum()

            loss.backward()
            optimizer.step()

            if ((mol_idxs + 1) % args["frequency_of_the_test"]
                    == 0) or (mol_idxs == len(train_data) - 1):
                if test_data is not None:
                    test_score, _ = test(model, test_data, device, args)
                    # eval on test dataset
                    # test_score, _ = self.test(self.test_data, device, args)
                    # print("Epoch = {}, Iter = {}/{}: Test Score = {}".format(
                    #     epoch, mol_idxs + 1, len(train_data), test_score))
                    if test_score > max_test_score:
                        max_test_score = test_score
                        best_model_params = {
                            k: v.cpu() for k, v in model.state_dict().items()
                        }
                    print("Current best = {}".format(max_test_score))

    return max_test_score, best_model_params


def test(model, test_data, device, args):
    logging.info("----------test--------")
    model.eval()
    model.to(device)

    with torch.no_grad():
        y_pred = []
        y_true = []
        masks = []
        for mol_idx, (adj_matrix, feature_matrix, label,
                      mask) in enumerate(test_data):
            adj_matrix = adj_matrix.to(device=device,
                                       dtype=torch.float32,
                                       non_blocking=True)
            feature_matrix = feature_matrix.to(device=device,
                                               dtype=torch.float32,
                                               non_blocking=True)

            logits = model(adj_matrix, feature_matrix)

            y_pred.append(logits.cpu().numpy())
            y_true.append(label.cpu().numpy())
            masks.append(mask.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    masks = np.array(masks)

    results = []
    for label in range(masks.shape[1]):
        valid_idxs = np.nonzero(masks[:, label])
        truth = y_true[valid_idxs, label].flatten()
        pred = y_pred[valid_idxs, label].flatten()

        if np.all(truth == 0.0) or np.all(truth == 1.0):
            results.append(float("nan"))
        else:
            if args["metric"] == "prc-auc":
                precision, recall, _ = precision_recall_curve(truth, pred)
                score = auc(recall, precision)
            else:
                score = roc_auc_score(truth, pred)

            results.append(score)

    score = np.nanmean(results)

    return score, model


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):

    def __init__(self,
                 cid,
                 net,
                 trainloader,
                 valloader=None,
                 args=None) -> None:
        super().__init__()
        self.cid = cid
        self.net = net
        self.trainloader = trainloader[self.cid]
        self.valloader = valloader
        self.args = args

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.net)

    def fit(self, parameters: NDArrays,
            config: Dict[str,
                         Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        set_parameters(self.net, parameters=parameters)
        train(self.net, self.trainloader, self.valloader, self.args['device'],
              self.args)

        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(
            self, parameters: NDArrays,
            config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        set_parameters(self.net, parameters)

        if self.valloader is None:
            return 0.0, 20, {"accuracy": 0.95}

        score, model = test(self.net, self.valloader, self.args['device'],
                            self.args)
        return score, len(self.valloader), {"accuracy": 0.95}


class FlowerServer(fl.server.strategy.FedAvg):

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[[int, NDArrays, Dict[str, Scalar]],
                                       Optional[Tuple[float,
                                                      Dict[str,
                                                           Scalar]]],]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str,
                                                             Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)


# if __name__ == "__main__":
#     net = GcnMoleculeNet(feat_dim=8,
#                          num_categories=2,
#                          hidden_dim=32,
#                          node_embedding_dim=32,
#                          dropout=0.3,
#                          readout_hidden_dim=64,
#                          graph_embedding_dim=64)
#     # strategy = fl.server.strategy.FedAdagrad()
#     params = get_parameters(net)
#     print("=======params=======", params)
