import timm
import torch

def get_model_param_count(model_name):
    try:
        model = timm.create_model(model_name, pretrained=False)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model_name, total, trainable
    except Exception as e:
        return model_name, -1, -1  # 忽略无法构建的模型

# 获取所有模型名
model_names = timm.list_models(pretrained=True)

# 收集参数量信息（建议只跑 top 模型，避免太久）
results = []
for name in model_names:
    print(f"Checking {name}...")
    info = get_model_param_count(name)
    if info[1] > 0:
        results.append(info)

# 排序输出（按 total 参数量降序）
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

# 写入文件
with open("timm_models_sorted.txt", "w") as f:
    f.write(f"All {len(results_sorted)} models by parameter count:\n")
    for name, total, trainable in results_sorted:
        f.write(f"{name:50s} | Total: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M\n")

print("已写入 timm_models_sorted.txt") 