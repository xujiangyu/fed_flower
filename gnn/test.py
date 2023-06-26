import pickle

f = open(
    "/Users/xusong/Documents/githubs/Federated_Classification_and_Segmentation-master/gnn/train_data_local_dict.pkl",
    "rb")

train_data = pickle.load(f)

f.close()
