from gcn import *

net = GCNNodeCLF(nfeat=args["nfeat"],
                 nhid=args["nhid"],
                 nlayer=args["nlayer"],
                 dropout=args["dropout"],
                 nclass=args["nclass"])

client1 = FlowerClient(cid=0,
                       trainloader=args["trainloader"],
                       args=args,
                       net=net)

fl.client.start_numpy_client(client=client1, server_address="localhost:8091")
