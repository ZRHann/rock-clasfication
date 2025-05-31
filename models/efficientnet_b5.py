import timm
import torch.nn as nn

def create_model(num_classes=9):
    # 创建模型
    model = timm.create_model(
        'efficientnet_b5.sw_in12k_ft_in1k',
        pretrained=True,
        num_classes=num_classes
    )
    
    # 获取特征维度
    in_features = 2048  # EfficientNet-B5的特征维度
    
    # 替换分类器，使用更简单的结构但保持较大参数量
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),          
        nn.Linear(in_features, 1024),
        nn.GELU(),
        nn.LayerNorm(1024),          
        nn.Dropout(p=0.4),          
        nn.Linear(1024, num_classes)
    )

    # 冻结所有特征提取层
    for name, param in model.named_parameters():
        if 'classifier' not in name:  # 冻结除classifier之外的所有层
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    return model

def get_model_name():
    return "efficientnet_b5" 