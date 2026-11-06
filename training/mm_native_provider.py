"""Native Qwen3.5 provider with identity indexing when context parallelism is off."""
import os
os.environ["FLA_FLASH_QLA"]="0"
import torch
import slime_plugins.models.qwen3_5_vl as native
import slime_plugins.models.qwen3_5_vl_utils as native_utils

_native_cp_indices = native_utils.get_packed_cp_local_indices


def get_packed_cp_local_indices(cu_seqlens, cp_size, cp_rank, device):
    if cp_size == 1:
        if cp_rank != 0:
            raise ValueError('A single context-parallel rank must have rank zero')
        # Native get_batch does not pad each sample when CP=1. Odd sample
        # lengths are valid; the full packed stream includes its own pad segment.
        return torch.arange(int(cu_seqlens[-1]), dtype=torch.long, device=device)
    return _native_cp_indices(cu_seqlens, cp_size, cp_rank, device)


native.get_packed_cp_local_indices = get_packed_cp_local_indices
native_utils.get_packed_cp_local_indices = get_packed_cp_local_indices
get_qwen3_5_vl_model_provider = native.get_qwen3_5_vl_model_provider
