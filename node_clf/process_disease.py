import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
# from torch.utils.data.dataloader import DataLoader
from torch_geometric.data import DataLoader
from collections import Counter


def split_data(feature, label, test_ratio=0.2):
    train_x, test_x, train_y, test_y = train_test_split(feature,
                                                        label,
                                                        test_size=test_ratio,
                                                        random_state=42)

    return train_x, test_x, train_y, test_y


def preprocess(data_path, k=3):
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
    float_inds = np.arange(9, 40).tolist()
    int_inds = np.arange(1, 9).tolist()

    for col in float_inds:
        mean_val = x[col].mean()
        x[col].fillna(mean_val, inplace=True)

    # print("----------", x)
    train_x, test_x, train_y, test_y = split_data(x, label, test_ratio=0.01)

    # construct neighbor_obj with 'train_x'
    float_train_x = train_x[float_inds]
    # print("train_x", type(train_y), train_y, train_x)
    neigh_obj = NearestNeighbors(n_neighbors=k)
    neigh_obj.fit(float_train_x)

    # search k-neighbors for 'float_train_x' and 'float_test_x'
    float_test_x = test_x[float_inds]

    train_neighbors = neigh_obj.kneighbors(float_train_x)
    test_neighbors = neigh_obj.kneighbors(float_test_x)

    # construct feature_mat and edge_index
    train_feature_array = []
    train_edge_index = []

    test_edge_index = []

    data_load = []

    # print("len ", len(train_neighbors), train_neighbors)

    for tmp_test_neighbor in test_neighbors[1]:
        for current_neighbor in tmp_test_neighbor:
            test_edge_index.append([])

    for ind, tmp_train_neighbor in enumerate(train_neighbors[1]):
        # print("===========", tmp_train_neighbor, float_train_x.iloc[0])
        tmp_feature_array = train_x.iloc[tmp_train_neighbor][int_inds].values

        # tmp_edge_index = []
        for current_neighbor in tmp_train_neighbor:

            # del owner
            if current_neighbor > ind:
                train_edge_index.append([ind, current_neighbor])
                train_edge_index.append([current_neighbor, ind])
            # if current_neighbor != ind:
            #     tmp_edge_index.append([ind, current_neighbor])
            #     tmp_edge_index.append([current_neighbor, ind])

    train_edge_index_tensor = torch.tensor(train_edge_index, dtype=torch.long)
    train_x_tensor = torch.tensor(train_x[int_inds].values, dtype=torch.float)
    train_y_tensor = torch.tensor(train_y.values, dtype=torch.int64)

    train_data_loader = Data(
        x=train_x_tensor,
        edge_index=train_edge_index_tensor.t().contiguous(),
        y=train_y_tensor)

    # tmp_edge_index_tensor = torch.tensor(tmp_edge_index, dtype=torch.long)

    # tmp_feature_tensor = torch.tensor(tmp_feature_array, dtype=torch.float)

    # tmp_label = torch.tensor(train_y.values[tmp_train_neighbor],
    #                          dtype=torch.int64)

    # tmp_data = Data(x=tmp_feature_tensor,
    #                 edge_index=tmp_edge_index_tensor.t().contiguous(),
    #                 y=tmp_label)

    # data_load.append(tmp_data)

    # print(tmp_feature_array)

    data_load_batch = DataLoader(train_data_loader, batch_size=1, shuffle=False)

    print("data_load_batch", data_load_batch)

    return train_data_loader

    # return train_x, test_x, train_y, test_y


def find_neighbors(k=3,):
    pass


data_path = "/Users/xusong/Downloads/complex_disease.csv"
train_data_loader = preprocess(data_path)


def train_cite(model, args):
    subgraphs, num_graphs, num_features, num_labels = pickle.load(
        open(
            "/Users/xusong/Documents/githubs/Federated_Classification_and_Segmentation-master/node_clf/egonetworks.pkl",
            "rb"))

    # subgraphs = DataLoader(subgraphs, batch_size=16, shuffle=False)

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
        current_loss = []
        for _, batch in enumerate(subgraphs):
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            current_loss.append(loss.detach().numpy())

        print("mean loss: ", np.mean(current_loss))
        # print(loss)

    total_correct = 0
    total_num = 0

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(subgraphs):
            target = batch.y
            target = target.to(args["device"])
            pred = model(batch)
            target = target.long()

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            total_correct += correct.item()
            total_num += target.size(0)

    print("acc: ", total_correct / total_num)

    # pred_y = torch.argmax(pred, dim=1)
    # acc = (pred_y == label).sum() / len(pred_y)
    # print("acc: ", acc, pred_y, label)


def get_client_data():
    f = open(
        "/Users/xusong/Documents/githubs/Federated_Classification_and_Segmentation-master/node_clf/partition-3-0.5-train.pkl",
        "rb")
    data = pickle.load(f)

    # print(data)


get_client_data()

# data = None
# # print(data)
# float_inds = np.arange(9, 30).tolist()

# # print(data[10])
# # print(data.iloc[0])

# float_data = data[float_inds]

# # mean_val = float_data.mean(axis=1)

# # float_data.fillna(mean_val, inplace=True)

# print(float_data)

# from sklearn.neighbors import NearestNeighbors

# k = 3
# neigh = NearestNeighbors(n_neighbors=k)
# neigh.fit(float_data.values)

# print(neigh.kneighbors(float_data.iloc[0].values.reshape(1, -1))[1])
