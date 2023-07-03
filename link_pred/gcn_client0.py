from gcn import *

net = GCNLinkPred(in_channels=args["in_channels"], out_channels=args["out_channels"])

client1 = FlowerClient(cid=0, trainloader=args["trainloader"], args=args, net=net)

fl.client.start_numpy_client(client=client1, server_address="localhost:8091")
