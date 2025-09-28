import torch, sys, os, re

path = sys.argv[1] if len(sys.argv) > 1 else "/home/weijie.wang/project/DiffPortrait360/diffportrait360/model_state-340000.th"
ckpt = torch.load(path, map_location="cpu")
sd = ckpt.get("state_dict", ckpt)

# 常见三种前缀：exact / 带model. / 叫vae.
prefixes = ["first_stage_model.", "model.first_stage_model.", "vae."]

hits = {p: [k for k in sd.keys() if k.startswith(p)] for p in prefixes}
print(">>>> VAE key 命中统计：")
for p, ks in hits.items():
    print(f"  prefix '{p}': {len(ks)} keys")
    if ks:
        print("   e.g.", ks[:5])

# 汇总一下到底有没有任何 VAE key
has_any_vae = any(len(ks) > 0 for ks in hits.values())
print("\ncontain ANY VAE keys?", has_any_vae)

# 如果命中在 'vae.'，展示一个从 vae. 到 first_stage_model. 的映射示例
if hits["vae."]:
    mapped = {k.replace("vae.", "first_stage_model."): sd[k] for k in hits["vae."][:5]}
    print("\n示例映射(vae. -> first_stage_model.):")
    for k in mapped.keys():
        print("  ", k)
