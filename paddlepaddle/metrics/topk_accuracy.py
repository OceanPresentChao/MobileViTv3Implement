from paddle import Tensor
from typing import Optional
import paddle

from . import register_stats_fn


@register_stats_fn(name="top1")
@register_stats_fn(name="top5")
def top_k_accuracy(
    output: Tensor, target: Tensor, top_k: Optional[tuple] = (1,)
) -> list:
    maximum_k = max(top_k)
    batch_size = target.shape[0]

    _, pred = output.topk(maximum_k, 1, True, True)
    pred = pred.t()
    correct = pred.equal(target.reshape([1, -1]).expand_as(pred))

    results = []
    for k in top_k:
        correct_k = correct[:k].reshape([-1]).astype(paddle.float32).sum(0, keepdim=True)
        acc_k = correct_k * (100.0 / batch_size)
        results.append(acc_k)
    return results
