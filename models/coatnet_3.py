import torch
import torch.nn as nn
import timm

def create_model(num_classes=9):
    # 加载预训练的CoAtNet-3
    model = timm.create_model('coatnet_3_rw_224.sw_in12k', pretrained=True)
    
    # 获取最后一层的输入特征维度
    num_features = 1536  # CoAtNet-3的特征维度
    
    # 替换分类头，处理 [B, C, H, W] -> [B, num_classes]
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),    # [B, C, H, W] -> [B, C, 1, 1]
        nn.Flatten(1),              # [B, C, 1, 1] -> [B, C]
        nn.LayerNorm(num_features), # [B, C]
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 1024),
        nn.GELU(),
        nn.Dropout(p=0.4),
        nn.Linear(1024, num_classes)
    )

    # 冻结特征提取层
    for name, param in model.named_parameters():
        if "head" not in name:  # 冻结除head外的所有层
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    return model

def get_model_name():
    return "coatnet_3" 