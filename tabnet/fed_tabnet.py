from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from collections import OrderedDict
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, Scalar
import torch


def train_metrics(model, test_x, test_y):
    preds = model.predict(test_x.values)
    acc = (preds == test_y.values).astype(int).sum() / len(preds)
    # print("feature_importances_: ", clf.feature_importances_,
    #       clf.feature_importances_.shape)

    recall = (preds * test_y.values).sum() / (test_y.values
                                              == 1).astype(int).sum()

    prob = model.predict_proba(test_x.values)

    # print("prob: ", prob)

    return recall, acc


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class TabNetClient(fl.client.NumPyClient):

    def __init__(self, model, test_x, test_y, train_x, train_y) -> None:
        self.model = model
        self.test_x = test_x
        self.test_y = test_y
        self.train_x = train_x
        self.train_y = train_y
        self.model.fit(self.train_x.values, self.train_y.values, max_epochs=1)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model.network)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.network.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.network.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays,
            config: Dict[str,
                         Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters=parameters)
        self.model.fit(self.train_x.values, self.train_y.values, max_epochs=1)

        return self.get_parameters(config={}), len(self.test_y), {
            "loss=" + str(float(0.1)) + "_" + "accuracy": 0.1
        }

    def evaluate(
            self, parameters: NDArrays,
            config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:

        self.set_parameters(parameters=parameters)

        recall, acc = train_metrics(self.model, self.test_x, self.test_y)

        return recall, len(self.test_x), {"accuracy": acc}


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