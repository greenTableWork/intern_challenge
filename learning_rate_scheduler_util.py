import torch.optim as optim


def build_scheduler_kwargs_from_args(args):
    """Translate CLI scheduler arguments into scheduler kwargs."""
    if args.scheduler == "plateau":
        return {
            "factor": args.scheduler_factor,
            "patience": args.scheduler_patience,
        }
    if args.scheduler == "cosine":
        return {
            "eta_min": args.scheduler_eta_min,
        }
    return {}


def create_lr_scheduler(
    optimizer,
    scheduler_name,
    num_epochs,
    scheduler_kwargs=None,
):
    """Build the requested learning-rate scheduler."""
    scheduler_kwargs = dict(scheduler_kwargs or {})

    if scheduler_name == "none":
        return None, False

    if scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=scheduler_kwargs.get("factor", 0.5),
            patience=scheduler_kwargs.get("patience", 50),
        )
        return scheduler, True

    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs),
            eta_min=scheduler_kwargs.get("eta_min", 1e-4),
        )
        return scheduler, False

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
