import logging
import torch
import pickle
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from collections import OrderedDict
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, Scalar


class GCNNodeCLF(torch.nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GCNNodeCLF, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout
        self.nclass = nclass

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()

        for l in range(nlayer - 1):
            self.graph_convs.append(GCNConv(nhid, nhid))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.post(x)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


def train(model, train_data, args):
    model.to(args["device"])
    model.train()

    if args['optimizer'] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args["learning_rate"],
                                    weight_decay=args["weight_decay"])
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args["learning_rate"],
                                     weight_decay=args["weight_decay"])

    max_test_score, max_val_score = 0, 0
    best_model_params = {}
    for epoch in range(args["epochs"]):
        # for idx_batch, batch in enumerate(train_data):
        #     batch.to(args["device"])
        #     optimizer.zero_grad()
        #     pred = model(batch)
        #     label = batch.y
        #     loss = model.loss(pred, label)
        #     loss.backward()
        #     optimizer.step()

        optimizer.zero_grad()
        pred = model(train_data)
        label = train_data.y
        loss = model.loss(pred, label)
        loss.backward()
        optimizer.step()
        print(loss)

        pred_y = torch.argmax(pred, dim=1)
        acc = (pred_y == label).sum() / len(pred_y)
        print("acc: ", acc)

    print("sum true", sum(pred_y))

    return max_test_score, best_model_params

    # criterion = nn.CrossEntropyLoss().to(args["device"])

    # if args['optimizer'] == "sgd":
    #     optimizer = torch.optim.SGD(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         lr=args["learning_rate"],
    #     )
    # else:
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
    #                                         model.parameters()),
    #                                  lr=args["learning_rate"],
    #                                  weight_decay=args["weight_decay"],
    #                                  amsgrad=True)

    # epoch_loss = []

    # for epoch in range(args["epochs"]):
    #     batch_loss = []

    #     # for batch_idx, (x, labels) in enumerate(train_data):
    #     for batch_idx, tmp_data_batch in enumerate(train_data):
    #         print("=======tmp_data_batch=====", tmp_data_batch.x.shape,
    #               tmp_data_batch.y.shape, tmp_data_batch.edge_index.shape)
    #         # feature = tmp_data_batch.x
    #         labels = tmp_data_batch.y.to(args["device"])
    #         # feature, labels = feature.to(args["device"]), labels.to(
    #         #     args["device"])
    #         model.zero_grad()
    #         log_probs = model(tmp_data_batch)
    #         labels = labels.long()
    #         loss = criterion(log_probs, labels)  # pylint: disable=E1102
    #         loss.backward()
    #         optimizer.step()

    #         # Uncommet this following line to avoid nan loss
    #         # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    #         batch_loss.append(loss.item())
    #     if len(batch_loss) == 0:
    #         epoch_loss.append(0.0)
    #     else:
    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #     logging.info("Client Epoch: {}\tLoss: {:.6f}".format(
    #         epoch,
    #         sum(epoch_loss) / len(epoch_loss)))


def test(model, test_data, args):
    model.to(args["device"])
    model.eval()

    metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

    criterion = nn.CrossEntropyLoss().to(args['device'])

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data):
            x = x.to(args["device"])
            target = target.to(args["device"])
            pred = model(x)
            target = target.long()
            loss = criterion(pred, target)  # pylint: disable=E1102

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            metrics["test_correct"] += correct.item()
            metrics["test_loss"] += loss.item() * target.size(0)
            metrics["test_total"] += target.size(0)

    return metrics


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


args = {}
args["device"] = "cpu"
args["optimizer"] = "sgd"
args["learning_rate"] = 0.05
args["epochs"] = 20
args["weight_decay"] = 0.01
args["metric"] = "auc"
args["nfeat"] = 602
args["nclass"] = 6
args["nlayer"] = 5
args["dropout"] = 0.3
args["nhid"] = 0.3
args["trainloader"] = None


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