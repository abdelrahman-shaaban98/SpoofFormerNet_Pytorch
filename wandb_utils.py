import wandb


def log_model_wandb(run, model_path, artifact_name):
    model_artifact = wandb.Artifact(name=artifact_name, type="model")
    model_artifact.add_file(model_path)
    # wandb.log_artifact(model_artifact)
    run.log_artifact(model_artifact)


def log_train_metrics_wandb(run, epoch, train_metrics, val_metrics):
    row = dict(
        epoch        = epoch,
        train_loss   = round(train_metrics["loss"],     5),
        train_acc    = round(train_metrics["accuracy"], 5),
        val_loss     = round(val_metrics["loss"],       5),
        val_acc      = round(val_metrics["accuracy"],   5),
        val_apcer    = round(val_metrics["apcer"],      5),
        val_bpcer    = round(val_metrics["bpcer"],      5),
        val_acer     = round(val_metrics["acer"],       5),
        val_eer      = round(val_metrics["eer"],        5),
        val_auc      = round(val_metrics["auc"],        5),
    )

    run.log(row)