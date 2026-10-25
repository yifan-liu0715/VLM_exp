"""
验证多节点分布式通信是否正确建立。

用法（每台机器上都要跑）：
  机器 0:
    torchrun --nnodes=2 --nproc_per_node=8 \
      --master_addr=$MASTER_IP --master_port=29500 --node_rank=0 \
      verify_multinode.py

  机器 1:
    torchrun --nnodes=2 --nproc_per_node=8 \
      --master_addr=$MASTER_IP --master_port=29500 --node_rank=1 \
      verify_multinode.py
"""
import os
import time
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    print(f"[rank {rank}] local_rank={local_rank}, world_size={world_size}, device={device}")

    # 1) all-reduce 验证：sum of all ranks = N*(N-1)/2
    tensor = torch.tensor([float(rank)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size - 1) / 2
    assert tensor.item() == expected, (
        f"all_reduce FAILED! rank={rank}, got {tensor.item()}, expected {expected}"
    )

    # 2) barrier
    dist.barrier()

    if rank == 0:
        print(f"\n{'='*50}")
        print(f"  all_reduce check PASSED (world_size={world_size})")
        print(f"{'='*50}")

    # 3) 带宽测试
    dist.barrier()
    big_tensor = torch.randn(256 * 1024 * 1024, device=device)  # 1GB fp32

    # warmup
    for _ in range(3):
        dist.all_reduce(big_tensor)
    torch.cuda.synchronize()

    # timed
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    n_iters = 5
    for _ in range(n_iters):
        dist.all_reduce(big_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    data_size_gb = big_tensor.numel() * 4 / 1e9
    avg_time = elapsed / n_iters
    bandwidth = data_size_gb / avg_time

    if rank == 0:
        print(f"\n  === All-reduce bandwidth test ===")
        print(f"  Tensor size: {data_size_gb:.2f} GB (fp32)")
        print(f"  Avg all-reduce time: {avg_time*1000:.1f} ms")
        print(f"  Effective bandwidth: {bandwidth:.2f} GB/s")

    dist.barrier()

    if rank == 0:
        print(f"\n{'='*50}")
        print(f"  ALL CHECKS PASSED!")
        print(f"  {world_size} GPUs across all nodes communicating correctly.")
        print(f"{'='*50}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
