def train_epoch_ga(model, data, optimizer, kl_weight=0.0):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.pos_edge_index)
    if hasattr(model, 'kl_loss') and kl_weight > 0:
        loss = loss + kl_weight * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def evaluate_ga(model, train_data, val_data, test_data):
    model.eval()
    # Encode on the full (training) graph features/edges
    z = model.encode(train_data.x, train_data.edge_index)

    val_auc, val_ap = model.test(z, val_data.pos_edge_index, val_data.neg_edge_index)
    test_auc, test_ap = model.test(z, test_data.pos_edge_index, test_data.neg_edge_index)
    return (float(val_auc), float(val_ap), float(test_auc), float(test_ap))
