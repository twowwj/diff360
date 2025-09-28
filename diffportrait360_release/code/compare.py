#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 比较两份 PyTorch state_dict 的三件事：
# 1) key 集合是否一致
# 2) 交集中是否有形状不一致
# 3) 数值是否一致（逐元素完全相等或在 allclose 容差内）
#
# 使用方法：
#   直接修改下方【配置区】里的 A_PATH 与 B_PATH（必要），其余开关按需改动，然后：
#       python compare_state_dicts_fixed.py
#
# 备注：
# - KEY_A/KEY_B：当 checkpoint 外层包了一层，如 ckpt["model"] 或 ckpt["state_dict"]，可以写成 "model" / "state_dict" / "foo.bar"。
# - STRIP_PREFIX_*：用来去掉例如 "module." 的前缀。
# - IGNORE_PATTERN：正则过滤一些不想比较的键（例如 BN 的 running_mean/var）。
# - EXACT=False 时使用 torch.allclose（ATOL/RTOL 可调）；若 EXACT=True 则使用逐元素完全相等检查。

# ===========================
# ========【配置区】==========
# ===========================
A_PATH = r"/home/weijie.wang/project/DiffPortrait360/diffportrait360/model_state-340000.th"   # ←← 改成你的 A 模型
B_PATH = r"/home/weijie.wang/project/DiffPortrait360/diffportrait360_release/code/consistency_output/model_state-100.th"   # ←← 改成你的 B 模型

KEY_A = None                 # 例如 "model" 或 "state_dict" 或 "model.ema"；不需要就设为 None
KEY_B = None                 # 例如 "model" 或 "state_dict"；不需要就设为 None
STRIP_PREFIX_A = ""          # 例如 "module."
STRIP_PREFIX_B = ""          # 例如 "module."
IGNORE_PATTERN = None        # 例如 r"running_(mean|var)|num_batches_tracked"

ATOL = 1e-6                  # allclose 的绝对误差容差（EXACT=False 时生效）
RTOL = 1e-5                  # allclose 的相对误差容差（EXACT=False 时生效）
EXACT = False                # True=逐元素完全相等；False=使用 allclose

MAX_PRINT = 60               # 列表最多打印多少行
SAVE_JSON_PATH = None        # 保存 JSON 报告的路径；例如 "compare_report.json"；不保存则设为 None
# ===========================
# ========【配置区】==========
# ===========================

import json
import re
import sys
from collections import OrderedDict
from typing import Dict, Tuple

def _lazy_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception as e:
        print("[ERROR] PyTorch is required to run this script:", e, file=sys.stderr)
        sys.exit(1)

def _get_nested(d, path: str):
    """通过点路径访问嵌套字典；不存在就返回 None。"""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def _looks_like_state_dict(obj) -> bool:
    """简单判断是否像 state_dict：字典 + 字符串键 + Tensor 值（允许空字典）。"""
    torch = _lazy_import_torch()
    if not isinstance(obj, dict):
        return False
    if len(obj) == 0:
        return True
    for k, v in obj.items():
        if not isinstance(k, str):
            return False
        if not torch.is_tensor(v):
            return False
    return True

def _to_state_dict(obj, preferred_key: str = None) -> Dict[str, "torch.Tensor"]:
    """
    尝试几种常见布局：obj 本身、或 obj['state_dict']、obj['model']，或用户提供的 --key 路径。
    """
    if preferred_key:
        cand = _get_nested(obj, preferred_key)
        if cand is not None and _looks_like_state_dict(cand):
            return cand  # type: ignore

    if _looks_like_state_dict(obj):
        return obj  # type: ignore

    # 常见包裹字段
    for k in ["state_dict", "model", "ema", "model_ema", "net", "params", "weights"]:
        if isinstance(obj, dict) and k in obj and _looks_like_state_dict(obj[k]):
            return obj[k]  # type: ignore

    # 有些框架存成 "module"
    if isinstance(obj, dict) and "module" in obj and _looks_like_state_dict(obj["module"]):
        return obj["module"]  # type: ignore

    # 兜底：浅层搜一个像 state_dict 的 dict
    if isinstance(obj, dict):
        for v in obj.values():
            if _looks_like_state_dict(v):
                return v  # type: ignore

    raise ValueError("Could not find a state_dict. Try setting KEY_A/KEY_B to specify the dict path.")

def _strip_prefix(sd: Dict[str, "torch.Tensor"], prefix: str) -> Dict[str, "torch.Tensor"]:
    if not prefix:
        return sd
    plen = len(prefix)
    out = OrderedDict()
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
        else:
            out[k] = v
    return out

def _filter_by_regex(sd: Dict[str, "torch.Tensor"], pattern: str) -> Dict[str, "torch.Tensor"]:
    """保留“不匹配 pattern”的键（即忽略匹配 pattern 的键）。"""
    if not pattern:
        return sd
    regex = re.compile(pattern)
    return OrderedDict((k, v) for k, v in sd.items() if not regex.search(k))

def compare_state_dicts(sd1: Dict[str, "torch.Tensor"],
                        sd2: Dict[str, "torch.Tensor"],
                        atol: float = 1e-6,
                        rtol: float = 1e-5,
                        check_exact: bool = False) -> Tuple[dict, dict]:
    torch = _lazy_import_torch()

    keys1, keys2 = set(sd1.keys()), set(sd2.keys())
    inter = keys1 & keys2
    only1 = sorted(keys1 - keys2)
    only2 = sorted(keys2 - keys1)

    same_keys = (keys1 == keys2)

    shape_mismatch = []
    same_shape = []
    for k in sorted(inter):
        if sd1[k].shape != sd2[k].shape:
            shape_mismatch.append({"key": k, "shape_a": list(sd1[k].shape), "shape_b": list(sd2[k].shape)})
        else:
            same_shape.append(k)

    # 值比较（只在形状相同的键上）
    equal_exact = []
    equal_close = []
    diff_keys = []
    for k in same_shape:
        a, b = sd1[k], sd2[k]
        if check_exact:
            same = torch.equal(a, b)
        else:
            same = torch.allclose(a, b, atol=atol, rtol=rtol)
        if same:
            if torch.equal(a, b):
                equal_exact.append(k)
            else:
                equal_close.append(k)
        else:
            diff_keys.append(k)

    summary = {
        "count_a": len(sd1),
        "count_b": len(sd2),
        "same_keys": same_keys,                 # 情况1：key 集合是否完全一致
        "num_intersection": len(inter),
        "num_only_in_a": len(only1),
        "num_only_in_b": len(only2),
        "num_shape_mismatch": len(shape_mismatch),  # 情况2：交集中形状不一致
        "num_same_shape": len(same_shape),
        "num_equal_exact": len(equal_exact),        # 情况3：完全相等（逐元素）
        "num_equal_close": len(equal_close),        # 情况3：在容差内相等
        "num_value_diff": len(diff_keys),           # 情况3：数值不同
    }
    details = {
        "only_in_a": only1,
        "only_in_b": only2,
        "shape_mismatch": shape_mismatch,
        "value_diff_sample": diff_keys[:50],        # 仅展示前 50 个，防刷屏
        "equal_exact_sample": equal_exact[:50],
        "equal_close_sample": equal_close[:50],
    }
    return summary, details

def _print_list(title, lst, maxp):
    print(f"\n--- {title} (showing up to {maxp}) ---")
    for i, item in enumerate(lst[:maxp]):
        print(item)
    if len(lst) > maxp:
        print(f"... ({len(lst) - maxp} more)")

def main():
    torch = _lazy_import_torch()

    # 读文件到 CPU，避免占用 GPU 显存
    ckpt_a = torch.load(A_PATH, map_location="cpu")
    ckpt_b = torch.load(B_PATH, map_location="cpu")

    sd_a = _to_state_dict(ckpt_a, preferred_key=KEY_A)
    sd_b = _to_state_dict(ckpt_b, preferred_key=KEY_B)

    if STRIP_PREFIX_A:
        sd_a = _strip_prefix(sd_a, STRIP_PREFIX_A)
    if STRIP_PREFIX_B:
        sd_b = _strip_prefix(sd_b, STRIP_PREFIX_B)

    if IGNORE_PATTERN:
        sd_a = _filter_by_regex(sd_a, IGNORE_PATTERN)
        sd_b = _filter_by_regex(sd_b, IGNORE_PATTERN)

    summary, details = compare_state_dicts(sd_a, sd_b, atol=ATOL, rtol=RTOL, check_exact=EXACT)

    # 输出摘要
    print("=== Summary ===")
    for k, v in summary.items():
        print(f"{k:20s}: {v}")

    # 列表输出
    _print_list("Only in A", details["only_in_a"], MAX_PRINT)
    _print_list("Only in B", details["only_in_b"], MAX_PRINT)
    _print_list("Shape mismatch", details["shape_mismatch"], MAX_PRINT)
    _print_list("Value different (sample)", details["value_diff_sample"], MAX_PRINT)
    _print_list("Equal exact (sample)", details["equal_exact_sample"], MAX_PRINT)
    _print_list("Equal close (sample)", details["equal_close_sample"], MAX_PRINT)

    if SAVE_JSON_PATH:
        report = {"summary": summary, "details": details}
        with open(SAVE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON report to: {SAVE_JSON_PATH}")

if __name__ == "__main__":
    main()
