"""单机多卡(数据并行)训练的 torch.distributed 轻量封装。

设计原则:**不通过 torchrun 启动时,所有函数都是 no-op**,单卡代码路径与改动前
逐字节一致(eval/单进程训练完全不受影响)。

多卡语义(同步数据并行,等价于 batch 放大 world_size 倍):
  1) 启动:每张卡一个进程(torchrun --nproc_per_node=N),进程 i 用 cuda:i;
  2) setup_distributed():set_device + init_process_group(nccl);
  3) broadcast_module():初始化后把 rank0 的权重广播给所有 rank → 各副本初始权重一致;
  4) average_gradients():backward 之后、clip/step 之前对梯度做 all-reduce 取平均
     → 各副本梯度一致;
  5) all_reduce_mean(kl):自适应学习率前对 KL 取跨卡均值 → 各副本学习率一致。
  (1)(3)(4)(5) 共同保证:每一步后所有副本权重严格一致,故 rank0 保存的 ckpt 即共识模型。

每个 rank 各自跑独立的 num_envs 个环境、各自采样(seed 按 rank 偏移),从而提供更多样
的经验;梯度 all-reduce 把它们汇聚成一次等价于 N×num_envs 的更新。
"""
import os

import torch
import torch.distributed as dist


def _launched_by_torchrun():
    # torchrun 会注入这些环境变量;WORLD_SIZE>1 才算真正的多卡分布式启动。
    return (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


def setup_distributed():
    """从 torchrun 环境变量初始化默认进程组。

    返回 (local_rank, world_size)。未经 torchrun 启动(或 world_size==1)时,
    **不**初始化进程组并返回 (0, 1),下游一切行为与原始单卡代码完全一致。
    必须在创建 IsaacGym sim 之前调用(它会用 cuda:local_rank 建仿真)。
    """
    if not _launched_by_torchrun():
        return 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return local_rank, dist.get_world_size()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    """rank0 才做日志/保存/wandb,避免多进程互相覆盖。单进程恒为 True。"""
    return get_rank() == 0


def barrier():
    if is_dist():
        dist.barrier()


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


@torch.no_grad()
def broadcast_module(module, src=0):
    """把 module 的参数与 buffer 从 src rank 广播到所有 rank,使各副本初始权重一致。"""
    if not is_dist():
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


@torch.no_grad()
def average_gradients(module):
    """backward 后、step 前调用:单次融合 all-reduce 取平均,各副本迈出相同一步。

    遍历全部参数(grad 缺失用零占位),保证各 rank 的 all-reduce 张量一致,
    避免逐参数条件式 collective 数量不匹配导致的静默错乱/挂起。
    """
    if not is_dist():
        return
    params = list(module.parameters())
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.mul_(1.0 / get_world_size())
    for p, g in zip(params, torch._utils._unflatten_dense_tensors(flat, grads)):
        if p.grad is not None:
            p.grad.copy_(g)


def all_reduce_mean(tensor):
    """返回标量 tensor 的跨卡均值(不修改入参)。用于让自适应学习率在各卡一致。

    入参是已在 cuda:local_rank 上的标量;单进程时原样返回。
    """
    if not is_dist():
        return tensor
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced.mul_(1.0 / get_world_size())
    return reduced
