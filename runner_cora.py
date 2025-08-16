cora_out = run_experiment(
    dataset_name="Cora",
    hidden_channels=64,
    out_channels=32,
    dropout=0.2,
    lr=0.01,
    epochs_gae=300,
    epochs_vgae=300,
    kl_weight=1.0
)

print("\nCora Results:", cora_out["results"])
