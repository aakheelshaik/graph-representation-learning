from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

@torch.no_grad()
def common_neighbors_baseline(train_data, test_data):
    # Build adjacency set from train positives
    edge_index = train_data.pos_edge_index.cpu().numpy()
    n = train_data.num_nodes
    neigh = defaultdict(set)
    for u, v in edge_index.T:
        neigh[int(u)].add(int(v))
        neigh[int(v)].add(int(u))

    def score(u, v):
        return len(neigh[int(u)].intersection(neigh[int(v)]))

    # Scores for test positives and negatives
    pos = test_data.pos_edge_index.cpu().numpy()
    neg = test_data.neg_edge_index.cpu().numpy()

    y_true = np.concatenate([np.ones(pos.shape[1]), np.zeros(neg.shape[1])])
    y_score = np.concatenate([
        np.array([score(u, v) for u, v in pos.T]),
        np.array([score(u, v) for u, v in neg.T])
    ]).astype(float)

    # Normalize scores (optional)
    if y_score.max() > 0:
        y_score = y_score / y_score.max()

    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)

print("\nHeuristic baseline on Cora (Common Neighbors):")
cn_auc, cn_ap = common_neighbors_baseline(cora_out["train_data"], cora_out["test_data"])
print(f"CN Baseline -> Test AUC={cn_auc:.4f}, AP={cn_ap:.4f}")
