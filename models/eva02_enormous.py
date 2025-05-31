import timm
import torch.nn as nn

def create_model(num_classes=9):
    model = timm.create_model(
        'eva02_enormous_patch14_clip_224.laion2b_plus',
        pretrained=True,
        num_classes=num_classes
    )
    
    # 获取原始head结构
    original_head = model.head
    
    # 替换head中的最后一层，保持其他层不变
    model.head = nn.Sequential(
        nn.LayerNorm(model.head.in_features),  # 添加LayerNorm
        nn.Dropout(p=0.2),                     # 添加dropout
        nn.Linear(model.head.in_features, 2048),  # 新增的中间层
        nn.GELU(),
        nn.Dropout(p=0.4),
        nn.Linear(2048, num_classes)
    )

    # Freeze feature extraction layers
    for name, param in model.named_parameters():
        if "head" not in name:  # freeze all except head
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    return model

def get_model_name():
    return "eva02_enormous"

if __name__ == "__main__":
    model = create_model() 