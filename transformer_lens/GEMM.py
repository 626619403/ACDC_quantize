import argparse

import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import triton.profiler as proton
from contextlib import contextmanager

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "tiles_per_update" in args:
        ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}, tiles_per_update={args['tiles_per_update']:02}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                  M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: tl.constexpr,  #
                  BLOCK_SIZE_N: tl.constexpr,  #
                  BLOCK_SIZE_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 8
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3,
            "num_warps": 8
        }
    }
    # Check constraints.
    # 检查约束条件
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    # 1 维启动内核，每个线程块获取自己的程序。
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(a_ptr, b_ptr, c_ptr,  #
                             M, N, K,  #
                             stride_am, stride_ak,  #
                             stride_bk, stride_bn,  #
                             stride_cm, stride_cn,  #
                             BLOCK_SIZE_M: tl.constexpr,  #
                             BLOCK_SIZE_N: tl.constexpr,  #
                             BLOCK_SIZE_K: tl.constexpr,  #
                             GROUP_SIZE_M: tl.constexpr,  #
                             NUM_SMS: tl.constexpr,  #
                             ):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_SIZE_M
            start_n = pid_n * BLOCK_SIZE_N
            offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        if ki == k_tiles - 1:
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            if (c_ptr.dtype.element_ty == tl.float8e4nv):
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.float16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_persistent(a, b):
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 8
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3,
            "num_warps": 8
        }
    }
    # Check constraints.
    # 检查限制条件。
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    # 分配输出空间。
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    # 1 维启动内核，每个线程块获取自己的程序。
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_persistent[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(a_desc_ptr, b_desc_ptr, c_desc_ptr,  #
                                 M, N, K,  #
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 FP8_OUTPUT: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr):  #
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype)
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_tma_persistent(a, b):
    # Autotuner does not work with TMA. Use manual config.
    # 自动调优器与TMA不兼容。请使用手动配置。
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 8
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3,
            "num_warps": 8
        }
    }

    # Check constraints.
    # 检查约束条件。
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    desc_a = triton.tools.experimental_descriptor.create_2d_tma_descriptor(a.data_ptr(), M, K,
                                                                           configs[dtype]["BLOCK_SIZE_M"],
                                                                           configs[dtype]["BLOCK_SIZE_K"],
                                                                           a.element_size())
    desc_b = triton.tools.experimental_descriptor.create_2d_tma_descriptor(b.data_ptr(), N, K,
                                                                           configs[dtype]["BLOCK_SIZE_N"],
                                                                           configs[dtype]["BLOCK_SIZE_K"],
                                                                           b.element_size())
    desc_c = triton.tools.experimental_descriptor.create_2d_tma_descriptor(c.data_ptr(), M, N,
                                                                           configs[dtype]["BLOCK_SIZE_M"],
                                                                           configs[dtype]["BLOCK_SIZE_N"],
                                                                           c.element_size())
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_tma_persistent[grid](
        desc_a, desc_b, desc_c,  #
        M, N, K,  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_device_tma_persistent(workspace_ptr,  #
                                        tiles_per_update: tl.constexpr,  #
                                        a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr,  #
                                        BLOCK_SIZE_N: tl.constexpr,  #
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        GROUP_SIZE_M: tl.constexpr,  #
                                        NUM_SMS: tl.constexpr):  #
    # Matmul using TMA and device-side descriptor creation
    # 使用 TMA 和设备端描述符创建的矩阵乘法。
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    TMA_SIZE: tl.constexpr = 128
    workspace_base = workspace_ptr + start_pid * 3 * TMA_SIZE
    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE
    c_desc_ptr = workspace_base + 2 * TMA_SIZE

    tl.extra.cuda.experimental_device_tensormap_create2d(desc_ptr=a_desc_ptr, global_address=a_ptr,
                                                         load_size=[BLOCK_SIZE_M, BLOCK_SIZE_K], global_size=[M, K],
                                                         element_ty=a_ptr.dtype.element_ty)
    tl.extra.cuda.experimental_device_tensormap_create2d(desc_ptr=b_desc_ptr, global_address=b_ptr,
                                                         load_size=[BLOCK_SIZE_N, BLOCK_SIZE_K], global_size=[N, K],
                                                         element_ty=b_ptr.dtype.element_ty)
    tl.extra.cuda.experimental_device_tensormap_create2d(desc_ptr=c_desc_ptr, global_address=c_ptr,
                                                         load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N], global_size=[M, N],
                                                         element_ty=c_ptr.dtype.element_ty)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1
    ni = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            ni += 1

            # Simulate a grouped gemm
            # 模拟一个分组的GEMM (General Matrix Multiply) 操作。
            if ni == tiles_per_update:
                tl.extra.cuda.experimental_device_tensormap_create2d(desc_ptr=a_desc_ptr, global_address=a_ptr,
                                                                     load_size=[BLOCK_SIZE_M,
                                                                                BLOCK_SIZE_K], global_size=[M, K],
                                                                     element_ty=a_ptr.dtype.element_ty)
                tl.extra.cuda.experimental_device_tensormap_create2d(desc_ptr=b_desc_ptr, global_address=b_ptr,
                                                                     load_size=[BLOCK_SIZE_N,
                                                                                BLOCK_SIZE_K], global_size=[N, K],
                                                                     element_ty=b_ptr.dtype.element_ty)
                tl.extra.cuda.experimental_device_tensormap_create2d(desc_ptr=c_desc_ptr, global_address=c_ptr,
                                                                     load_size=[BLOCK_SIZE_M,
                                                                                BLOCK_SIZE_N], global_size=[M, N],
                                                                     element_ty=c_ptr.dtype.element_ty)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)
                ni = 0

            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype)
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_device_tma_persistent(a, b, tiles_per_update):
    # Autotuner does not work with TMA. Use manual config.
    # 自动调优器与 TMA 不兼容。请使用手动配置。
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 8
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3,
            "num_warps": 8
        }
    }

    # Check constraints.
    # 检查约束条件。
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    tma_size = 128
    workspace = torch.empty(NUM_SMS * 3 * tma_size, dtype=torch.uint8, device="cuda")

    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_device_tma_persistent[grid](
        workspace,  #
        tiles_per_update,  #
        a, b, c,  #
        M, N, K,  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )
    return c


def cublas_matmul(a, b):
    # Check constraints.
    # 检查约束条件。
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"cublas [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        cublas.matmul(a, b, c)
    return c


def torch_matmul(a, b):
    M, K = a.shape
    N, K = b.shape
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"torch [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        c = torch.matmul(a, b.T)
    return c


@contextmanager
def proton_context():
    proton.activate(0)
    try:
        yield
    finally:
        proton.deactivate(0)


def bench_fn(reps, warmup_reps, fn, *args):
    for _ in range(warmup_reps):
        fn(*args)
    with proton_context():
        for _ in range(reps):
            fn(*args)


def bench(K, dtype, tiles_per_update, reps=1000, warmup_reps=10000):
    M = 8192
    N = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)

    b = b.T.contiguous()

    if cublas is not None:
        bench_fn(reps, warmup_reps, cublas_matmul, a, b)
    if dtype == torch.float16:
        bench_fn(reps, warmup_reps, torch_matmul, a, b)
    bench_fn(reps, warmup_reps, matmul, a, b.T)
    bench_fn(reps, warmup_reps, matmul_persistent, a, b.T)
    if supports_tma():
        bench_fn(reps, warmup_reps, matmul_tma_persistent, a, b)
        bench_fn(reps, warmup_reps, matmul_device_tma_persistent, a, b, tiles_per_update)


def validate(M, N, K, dtype, tiles_per_update):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()

    torch_result = torch_matmul(a, b) if dtype == torch.float16 else None
    cublas_result = cublas_matmul(a, b) if cublas is not None else None
    naive_result = matmul(a, b.T)
    persistent_result = matmul_persistent(a, b.T)
    tma_persistent_result = matmul_tma_persistent(a, b) if supports_tma() else None
    device_tma_persistent_result = matmul_device_tma_persistent(a, b, tiles_per_update) if supports_tma() else None

    if torch_result is not None:
        naive_vs_torch = "✅" if torch.allclose(naive_result.to(torch.float16), torch_result.to(torch.float16),
                                               atol=1.0) else "❌"
    if cublas_result is not None:
        naive_vs_cublas = "✅" if torch.allclose(naive_result.to(torch.float16), cublas_result.to(torch.float16),
                                                atol=1.0) else "❌"
    naive_vs_persistent = "✅" if torch.allclose(naive_result.to(torch.float16), persistent_result.to(torch.float16),
                                                atol=1.0) else "❌"
    if tma_persistent_result is not None:
        naive_vs_tma_persistent = "✅" if torch.allclose(cublas_result.to(torch.float16),
                                                        tma_persistent_result.to(torch.float16), atol=1.0) else "❌"
    if device_tma_persistent_result is not None:
        naive_vs_device_tma_persistent = "✅" if torch.allclose(cublas_result.to(
            torch.float16), device_tma_persistent_result.to(torch.float16), atol=1.0) else "❌"
    print(f"M={M}, N={N}, K={K} verification naive vs: ", end="")
    if torch_result is not None:
        print(f"torch: {naive_vs_torch} ", end="")
    if cublas_result is not None:
        print(f"cublas: {naive_vs_cublas} ", end="")
    print(f"persistent: {naive_vs_persistent} ", end="")
    if tma_persistent_result is not None:
        print(f"TMA persistent: {naive_vs_tma_persistent} ", end="")
    if device_tma_persistent_result is not None:
        print(f"Device TMA persistent: {naive_vs_device_tma_persistent} ", end="")
    print()


def show_profile(precision, profile_name):
    import triton.profiler.viewer as proton_viewer
    metrics = ["time/ms"]
    if precision == 'fp8':
        metrics = ["tflop8/s"] + metrics
    elif precision == 'fp16':
        metrics = ["tflop16/s"] + metrics
    file_name = f"{profile_name}.hatchet"
    proton_viewer.parse(metrics, file_name, depth=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument(
        "--tiles_per_update",
        type=int,
        default=1,
        help=
        "Number of output tiles calculated for each update of the tma descriptor in matmul_device_tma_persistent_kernel",
    )
    parser.add_argument("--prec", type=str, choices=["fp8", "fp16"], default="fp16")
    args = parser.parse_args()

    if args.prec == 'fp8' and (not hasattr(torch, "float8_e4m3fn") or not is_cuda()):
        print("This example requires CUDA with fp8 support.")
        exit(1)

    dtype = torch.float8_e4m3fn if args.prec == 'fp8' else torch.float16

    if args.K and args.K_range is None:
        args.K_range = [args.K, args.K]
        args.K_step = 1  # doesn't matter as long as it's not 0

    torch.manual_seed(0)

    validate(32, 32, 32, dtype, args.tiles_per_update)
    validate(8192, 8192, 512, dtype, args.tiles_per_update)

    proton.start("matmul", hook="triton")
    for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
        bench(K, dtype, args.tiles_per_update)
    proton.finalize()
    show_profile(args.prec, "matmul")
