def load_dataset(name="Karate", root="/tmp/pyg"):
    name = name.lower()
    if name == "karate":
        dataset = KarateClub(root=os.path.join(root, "karate"))
        data = dataset[0]
        # Karate already has features (one-hot). Normalize for stability:
        transform = RandomLinkSplit(
            num_val=0.05, num_test=0.10,
            is_undirected=True, add_negative_train_samples=False
        )
        train_data, val_data, test_data = transform(data)
        num_features = data.num_features
        return (train_data, val_data, test_data, num_features)

    elif name == "cora":
        dataset = Planetoid(root=os.path.join(root, "cora"), name="Cora", transform=NormalizeFeatures())
        data = dataset[0]
        transform = RandomLinkSplit(
            num_val=0.05, num_test=0.10,
            is_undirected=True, add_negative_train_samples=False
        )
        train_data, val_data, test_data = transform(data)
        num_features = dataset.num_node_features
        return (train_data, val_data, test_data, num_features)

    else:
        raise ValueError("Supported datasets: 'Karate' or 'Cora'")
