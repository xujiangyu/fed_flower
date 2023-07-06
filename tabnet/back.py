import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import copy


def split_data(feature, label, test_ratio=0.2):
    train_x, test_x, train_y, test_y = train_test_split(feature,
                                                        label,
                                                        test_size=test_ratio,
                                                        random_state=42)
    return train_x, test_x, train_y, test_y


def avg_model(
    model1,
    model2,
    key_list=[
        'tabnet.encoder.att_transformers.0.fc.weight',
        'tabnet.encoder.feat_transformers.2.specifics.glu_layers.1.fc.weight',
        'tabnet.encoder.feat_transformers.2.specifics.glu_layers.0.fc.weight'
    ]):
    network1 = model1.network.state_dict()
    network2 = model2.network.state_dict()

    i = 0

    keys = network1.keys()

    for tmp_key in keys:
        print("==========", network1[tmp_key], network2[tmp_key])
        network1[tmp_key] = (network1[tmp_key] + network2[tmp_key]) / 2.0
        network2[tmp_key] = (network1[tmp_key] + network2[tmp_key]) / 2.0
        i += 1
        if i > 20:
            break

    return network1, network2


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


def client_tabnet(new_model, client_x, client_y, params=None):

    # new_model = copy.deepcopy(model)
    new_model.fit(client_x.values, client_y.values, max_epochs=1)

    print("feature importance: ", new_model.feature_importances_)

    return new_model


def test_tabnet(data_path):
    print("start")
    data = pd.read_csv(data_path,
                       engine='python',
                       encoding="utf-8",
                       header=None)
    # extract label
    label = data.pop(0)
    print("label dict:", Counter(label))

    # extract pos-inds
    x = data.copy()

    # set float and int column-index
    # float_inds = np.arange(9, 30).tolist()
    float_inds = np.arange(11, 25).tolist() + [9]
    #float_inds = np.arange(9, 40).tolist()
    int_inds = np.arange(1, 9).tolist()

    index_0 = label == 0
    index_1 = label == 1

    for col in float_inds:
        #mean_0 = x[col].loc[index_0].mean()
        #mean_1 = x[col].loc[index_1].mean()
        mean_1 = x[col].mean()

        x[col].fillna(mean_1, inplace=True)

        #x[col].loc[index_0].fillna(mean_0, inplace=True)
        #x[col].loc[index_1].fillna(mean_1, inplace=True)

    # print("----------", x)
    train_x, test_x, train_y, test_y = split_data(x[int_inds + float_inds],
                                                  label,
                                                  test_ratio=0.2)

    # x0, x1, y0, y1 = split_data(train_x, train_y, test_ratio=0.5)
    x0 = train_x.iloc[:200,]
    x1 = train_x.iloc[200:,]

    y0 = train_y[:200]
    y1 = train_y[200:]

    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    clf = TabNetClassifier(cat_idxs=[0, 1, 2, 3, 4, 5, 6, 7],
                           cat_dims=[2, 2, 2, 2, 2, 2, 2, 2],
                           n_d=8,
                           n_a=8)  #TabNetRegressor()
    #clf = TabNetClassifier()  #TabNetRegressor()
    model1 = TabNetClassifier(cat_idxs=[0, 1, 2, 3, 4, 5, 6, 7],
                              cat_dims=[2, 2, 2, 2, 2, 2, 2, 2],
                              n_d=8,
                              n_a=8)
    model2 = TabNetClassifier(cat_idxs=[0, 1, 2, 3, 4, 5, 6, 7],
                              cat_dims=[2, 2, 2, 2, 2, 2, 2, 2],
                              n_d=8,
                              n_a=8)
    # model1.feature_importances_

    # model1 = TabNetClassifier(cat_idxs=[0, 1, 2, 3, 4, 5, 6, 7],
    #                           cat_dims=[2, 2, 2, 2, 2, 2, 2, 2],
    #                           n_d=8,
    #                           n_a=8)
    # model2 = TabNetClassifier(cat_idxs=[0, 1, 2, 3, 4, 5, 6, 7],
    #                           cat_dims=[2, 2, 2, 2, 2, 2, 2, 2],
    #                           n_d=8,
    #                           n_a=8)

    for i in range(1):
        print("***********", x0, x1)
        model1 = client_tabnet(model1, x0, y0, params=None)
        model2 = client_tabnet(model2, x1, y1, params=None)

        avg_network1, avg_network2 = avg_model(model1, model2)
        model1.network.load_state_dict(avg_network1)
        model2.network.load_state_dict(avg_network2)

        recall, acc = train_metrics(model1, test_x, test_y)
        print("11111", recall, acc)

        recall, acc = train_metrics(model2, test_x, test_y)
        print("22222", recall, acc)

    # import xgboost as xgb

    # clf = xgb.XGBClassifier()
    # clf.fit(train_x, train_y)
    # recall, acc = train_metrics(clf, test_x, test_y)
    # print("recall, acc: ", recall, acc)

    return clf


data_path = "/Users/xusong/Downloads/complex_disease.csv"
# data_path = "/home/yawei/tableNet/complex_disease.csv"
clf = test_tabnet(data_path)