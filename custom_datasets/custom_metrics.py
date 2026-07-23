import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from sklearn.metrics import average_precision_score


@METRICS.register_module()
class mAPMetric(BaseMetric):
    """Pixel-wise mean Average Precision for semantic segmentation.

    Computes per-class average precision (area under precision-recall curve)
    using scikit-learn's average_precision_score, then averages over all
    non-ignored classes.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 collect_device: str = 'cpu',
                 prefix: str | None = None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            pred = data_sample.pred_sem_seg.data
            gt = data_sample.gt_sem_seg.data
            self.results.append((pred.cpu(), gt.cpu()))

    def compute_metrics(self, results: list) -> dict:
        # Determine number of classes from max label value
        max_label = -1
        all_preds, all_gts = [], []
        for pred, gt in results:
            mask = gt != self.ignore_index
            all_preds.append(pred[mask].numpy())
            all_gts.append(gt[mask].numpy())
            max_label = max(max_label, gt.max().item())

        num_classes = int(max_label) + 1
        if num_classes < 1:
            return {"mAP": 0.0}

        y_true = np.concatenate(all_gts)
        y_pred = np.concatenate(all_preds)

        # Per-class one-vs-rest AP
        aps = []
        for c in range(num_classes):
            gt_binary = (y_true == c).astype(np.int64)
            if gt_binary.sum() == 0:
                continue
            pred_binary = (y_pred == c).astype(np.float64)
            ap = average_precision_score(gt_binary, pred_binary)
            aps.append(ap)

        if len(aps) == 0:
            return {"mAP": 0.0}

        metrics = {f"AP_class_{c}": float(ap) for c, ap in enumerate(aps)}
        metrics["mAP"] = float(np.mean(aps))
        return metrics
