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
from common import *


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
    return [val.numpy() for _, val in net.items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net['model'].keys(), parameters)

    for k, v in params_dict:
        net['model'][k] = torch.Tensor(v)


class BertClient(fl.client.NumPyClient):

    def __init__(self, model=None, CLIENT_ID=0) -> None:
        self.model = model
        self.CLIENT_ID = CLIENT_ID

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model['model'])

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters: NDArrays,
            config: Dict[str,
                         Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters=parameters)

        # for epoch in range(0, MAX_EPOCH, AVG_PERIOD):
        #     current_iter = epoch + AVG_PERIOD
        #     cmd_str = f"fairseq-train --fp16 {ROOT_DIR}/client_{self.CLIENT_ID}/data-bin --task masked_lm --criterion masked_lm" \
        #                 f"--arch roberta_base --sample-break-mode complete --tokens-per-sample {TOKENS_PER_SAMPLE} --optimizer adam" \
        #                 f"--adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 --lr-scheduler polynomial_decay --lr {PEAK_LR} --warmup-updates {WARMUP_UPDATES}" \
        #                 f"--total-num-update {TOTAL_UPDATES} --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --batch-size {MAX_SENTENCES} --update-freq {UPDATE_FREQ}" \
        #                 f"--max-update {TOTAL_UPDATES} --log-format simple --log-interval 1 --tensorboard-logdir {ROOT_DIR}/client_{self.CLIENT_ID}/logdir --save-interval 1 --max-epoch {current_iter}" \
        #                 f"--save-dir {ROOT_DIR}/client_{self.CLIENT_ID}/checkpoints --restore-file {ROOT_DIR}/client_{self.CLIENT_ID}/server/checkpoint_avg_{epoch}.pt"

        #     assert os.system(cmd_str) == 0

        cmd_str = f"fairseq-train --fp16 {ROOT_DIR}/client_{self.CLIENT_ID}/data-bin --task masked_lm --criterion masked_lm " \
                    f"--arch roberta_base --sample-break-mode complete --tokens-per-sample {TOKENS_PER_SAMPLE} --optimizer adam " \
                    f"--adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 --lr-scheduler polynomial_decay --lr {PEAK_LR} --warmup-updates {WARMUP_UPDATES} " \
                    f"--total-num-update {TOTAL_UPDATES} --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --batch-size {MAX_SENTENCES} --update-freq {UPDATE_FREQ} " \
                    f"--max-update {TOTAL_UPDATES} --log-format simple --log-interval 1 --tensorboard-logdir {ROOT_DIR}/client_{self.CLIENT_ID}/logdir --save-interval 1 --max-epoch {AVG_PERIOD} " \
                    f"--save-dir {ROOT_DIR}/client_{self.CLIENT_ID}/checkpoints --restore-file {ROOT_DIR}/client_{self.CLIENT_ID}/checkpoints/checkpoint_best.pt"
        assert os.system(cmd_str) == 0

        model = torch.load(
            f"{ROOT_DIR}/client_{self.CLIENT_ID}/checkpoints/checkpoint_best.pt",
            map_location=torch.device("cpu"))
        self.model = model

        return self.get_parameters(config={}), 100, {
            "loss=" + str(float(0.1)) + "_" + "accuracy": 0.1
        }


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