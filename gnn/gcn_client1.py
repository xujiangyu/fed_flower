from gcn import FlowerClient, trainloader, args, fl, GcnMoleculeNet

net = GcnMoleculeNet(feat_dim=8,
                     num_categories=2,
                     hidden_dim=32,
                     node_embedding_dim=32,
                     dropout=0.3,
                     readout_hidden_dim=64,
                     graph_embedding_dim=64)

client1 = FlowerClient(cid=1, trainloader=trainloader, args=args, net=net)

fl.client.start_numpy_client(client=client1, server_address="localhost:8090")
