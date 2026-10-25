#!/usr/bin/env python3
"""
Inspect safetensors checkpoint keys and shapes.
Usage:
    python inspect_safetensors.py <path_to_safetensors> [--prefix <filter_prefix>] [--output <log_path>]
"""

import sys
import argparse
import struct
import os
from datetime import datetime


def parse_safetensors_header(path):
    """Parse safetensors file header without external dependencies."""
    with open(path, "rb") as f:
        # Read header length (first 8 bytes, little-endian uint64)
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("Invalid safetensors file: too short")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        
        # Read header JSON
        header_bytes = f.read(header_len)
        if len(header_bytes) < header_len:
            raise ValueError("Invalid safetensors file: header incomplete")
        
        import json
        header = json.loads(header_bytes.decode("utf-8"))
        
        # Filter out __metadata__
        tensors = {k: v for k, v in header.items() if not k.startswith("__")}
        return tensors


def inspect_checkpoint(path, prefix_filter=None, max_keys=200, output_path=None):
    """Inspect a safetensors checkpoint header and save to log."""
    
    # Build log content
    lines = []
    lines.append(f"Loading header: {path}")
    tensors = parse_safetensors_header(path)
    
    all_keys = list(tensors.keys())
    
    # Better param count
    def count_params(shape):
        p = 1
        for s in shape:
            p *= int(s)
        return p
    
    total_params = sum(count_params(v["shape"]) for v in tensors.values())
    
    # Estimate size (assume float32 unless dtype says otherwise)
    dtype_sizes = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I32": 4, "I64": 8, "U8": 1}
    total_size = sum(
        count_params(v["shape"]) * dtype_sizes.get(v.get("dtype", "F32"), 4)
        for v in tensors.values()
    )
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("Checkpoint Summary")
    lines.append("=" * 70)
    lines.append(f"Total keys:     {len(all_keys)}")
    lines.append(f"Total params:   {total_params / 1e9:.3f}B")
    lines.append(f"Total size:     {total_size / 1024**2:.1f} MB")
    lines.append("=" * 70)
    lines.append("")
    
    # Filter by prefix if specified
    if prefix_filter:
        keys = [k for k in all_keys if prefix_filter in k]
        lines.append(f"Filtered by '{prefix_filter}': {len(keys)} keys")
        lines.append("")
    else:
        keys = all_keys
    
    # Group keys by top-level module
    groups = {}
    for k in keys:
        top = k.split('.')[0]
        groups.setdefault(top, []).append(k)
    
    lines.append(f"Top-level modules ({len(groups)}):")
    for mod, mod_keys in sorted(groups.items()):
        mod_params = sum(count_params(tensors[k]["shape"]) for k in mod_keys)
        lines.append(f"  {mod:30s} {len(mod_keys):4d} keys  {mod_params/1e6:8.1f}M params")
    lines.append("")
    
    # Show detailed keys
    lines.append(f"Detailed keys (showing first {min(max_keys, len(keys))}):")
    lines.append(f"{'Key':<80s} {'Shape':<30s} {'Dtype':<8s} {'Params':<12s}")
    lines.append("-" * 140)
    
    for i, k in enumerate(keys[:max_keys]):
        v = tensors[k]
        shape_str = str(v["shape"])
        dtype = v.get("dtype", "F32")
        params = count_params(v["shape"])
        lines.append(f"{k:<80s} {shape_str:<30s} {dtype:<8s} {params/1e6:>10.2f}M")
    
    if len(keys) > max_keys:
        lines.append(f"")
        lines.append(f"... and {len(keys) - max_keys} more keys (use --max-keys to show more)")
    
    # Special check for kv_compressor
    kv_keys = [k for k in all_keys if "kv_compressor" in k]
    if kv_keys:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"kv_compressor keys ({len(kv_keys)}):")
        lines.append("=" * 70)
        for k in kv_keys:
            v = tensors[k]
            params = count_params(v["shape"])
            lines.append(f"  {k}: {v['shape']}  dtype={v.get('dtype', 'F32')}  ({params/1e3:.1f}K)")
    
    # Special check for lora
    lora_keys = [k for k in all_keys if "lora" in k.lower()]
    if lora_keys:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"LoRA keys ({len(lora_keys)}):")
        lines.append("=" * 70)
        for k in lora_keys[:20]:
            v = tensors[k]
            lines.append(f"  {k}: {v['shape']}")
        if len(lora_keys) > 20:
            lines.append(f"  ... and {len(lora_keys) - 20} more")
    
    # Join all lines
    log_content = "\n".join(lines)
    
    # Output
    if output_path:
        # Save to file
        with open(output_path, 'w') as f:
            f.write(log_content)
        print(f"Log saved to: {output_path}")
    else:
        # Print to stdout
        print(log_content)
    
    return tensors


def main():
    parser = argparse.ArgumentParser(description="Inspect safetensors checkpoint")
    parser.add_argument("--path", type=str, default="/primus_xpfs_workspace_T08/zhangyuxuan/projects2/LiveAnimate_stage1_0411/output_fsdp_lora_compress_attn_tf_multinode/step-500.safetensors", help="Path to safetensors file")
    parser.add_argument("--prefix", type=str, default=None, help="Filter keys by prefix")
    parser.add_argument("--max-keys", type=int, default=1000, help="Max keys to display")
    parser.add_argument("--output", type=str, default=None, help="Output log file path (default: print to stdout)")
    args = parser.parse_args()
    
    # Auto-generate output path if not specified
    output_path = args.output
    if output_path is None:
        # Default: same directory as input, with .log extension
        base = os.path.splitext(args.path)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base}_inspect_{timestamp}.log"
        print(f"Auto-generated log path: {output_path}")
    
    inspect_checkpoint(
        args.path,
        prefix_filter=args.prefix,
        max_keys=args.max_keys,
        output_path=output_path
    )


if __name__ == "__main__":
    main()
