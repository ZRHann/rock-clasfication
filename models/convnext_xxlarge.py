import timm
import torch.nn as nn

def create_model(num_classes=9):
    model = timm.create_model(
        'convnext_xxlarge.clip_laion2b_soup_ft_in12k',
        pretrained=True,
        num_classes=num_classes
    )
    
    # 获取原始head结构
    original_head = model.head
    
    # 替换head中的最后一层，保持其他层不变
    model.head = nn.Sequential(
        original_head.global_pool,  # 保持原有的全局池化
        original_head.norm,         # 保持原有的LayerNorm
        original_head.flatten,      # 保持原有的Flatten
        nn.Dropout(p=0.2),         # 添加dropout
        nn.Linear(3072, 1536),     # 新增的中间层
        nn.GELU(),
        nn.Dropout(p=0.4),
        nn.Linear(1536, num_classes)
    )

    # Freeze feature extraction layers
    for name, param in model.named_parameters():
        if "head" not in name:  # freeze all except head
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    return model

def get_model_name():
    return "convnext_xxlarge"

if __name__ == "__main__":
    model = create_model()