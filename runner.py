def run_experiment(dataset_name="Karate",
                   hidden_channels=32,
                   out_channels=16,
                   dropout=0.0,
                   lr=0.01,
                   epochs_gae=300,
                   epochs_vgae=300,
                   kl_weight=1.0):
    print(f"\n=== Dataset: {dataset_name} ===")
    train_data, val_data, test_data, in_channels = load_dataset(dataset_name)

    # Move to device
    for d in (train_data, val_data, test_data):
        d.x = d.x.to(device)
        d.edge_index = d.edge_index.to(device)
        d.pos_edge_index = d.pos_edge_index.to(device)
        d.neg_edge_index = d.neg_edge_index.to(device)

    # ---------- GAE ----------
    gae = GAE(GCNEncoder(in_channels, hidden_channels, out_channels, dropout)).to(device)
    opt = torch.optim.Adam(gae.parameters(), lr=lr)

    best_val_auc, best_gae = -1, None
    for epoch in range(1, epochs_gae + 1):
        loss = train_epoch_ga(gae, train_data, opt)
        val_auc, val_ap, test_auc, test_ap = evaluate_ga(gae, train_data, val_data, test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_gae = (test_auc, test_ap)
        if epoch % 50 == 0 or epoch == 1:
            print(f"[GAE][{epoch:03d}] loss={loss:.4f} | val AUC={val_auc:.4f} AP={val_ap:.4f} | test AUC={test_auc:.4f} AP={test_ap:.4f}")

    gae_test_auc, gae_test_ap = best_gae
    print(f"-> GAE (best val) Test: AUC={gae_test_auc:.4f}, AP={gae_test_ap:.4f}")

    # ---------- VGAE ----------
    vgae = VGAE(VariationalGCNEncoder(in_channels, hidden_channels, out_channels, dropout)).to(device)
    opt_v = torch.optim.Adam(vgae.parameters(), lr=lr)

    best_val_auc, best_vgae = -1, None
    for epoch in range(1, epochs_vgae + 1):
        loss = train_epoch_ga(vgae, train_data, opt_v, kl_weight=kl_weight)
        val_auc, val_ap, test_auc, test_ap = evaluate_ga(vgae, train_data, val_data, test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_vgae = (test_auc, test_ap)
        if epoch % 50 == 0 or epoch == 1:
            print(f"[VGAE][{epoch:03d}] loss={loss:.4f} | val AUC={val_auc:.4f} AP={val_ap:.4f} | test AUC={test_auc:.4f} AP={test_ap:.4f}")

    vgae_test_auc, vgae_test_ap = best_vgae
    print(f"-> VGAE (best val) Test: AUC={vgae_test_auc:.4f}, AP={vgae_test_ap:.4f}")

    # Return models + data for optional visualization
    return {
        "gae": gae, "vgae": vgae,
        "train_data": train_data, "val_data": val_data, "test_data": test_data,
        "results": {
            "GAE": {"test_auc": gae_test_auc, "test_ap": gae_test_ap},
            "VGAE": {"test_auc": vgae_test_auc, "test_ap": vgae_test_ap},
        }
    }
