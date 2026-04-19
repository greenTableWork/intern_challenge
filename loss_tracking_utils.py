import csv
import os
from datetime import datetime


def save_loss_history_csv(loss_history, output_dir, run_metadata=None):
    """Save loss history values to a CSV file."""
    loss_history_dir = os.path.join(output_dir, "loss_history")
    os.makedirs(loss_history_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(
        loss_history_dir, f"loss_history_{timestamp}.csv"
    )
    metadata = dict(loss_history.get("run_metadata", {}))
    if run_metadata:
        metadata.update(run_metadata)

    metadata.setdefault("run_id", timestamp)
    metadata.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))

    total_loss = loss_history.get("total_loss", [])
    wirelength_loss = loss_history.get("wirelength_loss", [])
    overlap_loss = loss_history.get("overlap_loss", [])
    overlap_count = loss_history.get("overlap_count", [])
    total_overlap_area = loss_history.get("total_overlap_area", [])
    max_overlap_area = loss_history.get("max_overlap_area", [])

    row_count = max(
        len(total_loss),
        len(wirelength_loss),
        len(overlap_loss),
        len(overlap_count),
        len(total_overlap_area),
        len(max_overlap_area),
    )

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        metadata_fields = list(metadata.keys())
        metric_fields = [
            "epoch",
            "total_loss",
            "wirelength_loss",
            "overlap_loss",
            "overlap_count",
            "total_overlap_area",
            "max_overlap_area",
        ]
        writer.writerow(metadata_fields + metric_fields)

        for epoch in range(row_count):
            metric_row = [
                epoch,
                total_loss[epoch] if epoch < len(total_loss) else "",
                wirelength_loss[epoch] if epoch < len(wirelength_loss) else "",
                overlap_loss[epoch] if epoch < len(overlap_loss) else "",
                overlap_count[epoch] if epoch < len(overlap_count) else "",
                total_overlap_area[epoch]
                if epoch < len(total_overlap_area)
                else "",
                max_overlap_area[epoch] if epoch < len(max_overlap_area) else "",
            ]
            writer.writerow([metadata[field] for field in metadata_fields] + metric_row)

    return output_path
