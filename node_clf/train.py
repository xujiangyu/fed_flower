from gcn import GCNNodeCLF, train, args
from process_disease import train_data_loader, train_cite

# args["device"] = "cuda:0"
args["device"] = "cpu"
args["optimizer"] = "sgd"
args["learning_rate"] = 0.05
args["epochs"] = 20
args["weight_decay"] = 0.001

model = GCNNodeCLF(nfeat=602, nclass=6, nlayer=5, dropout=0.3, nhid=32)
train_cite(model, args)