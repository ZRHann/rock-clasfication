import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T_cpu
import torchvision.transforms.v2 as T_gpu
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from PIL import Image
import csv
from collections import defaultdict
import importlib

torch.set_float32_matmul_precision('high')

# 配置是否从checkpoint恢复训练
resume_training = True
model_name = "eva02_enormous"  # 在这里选择要使用的模型

# 超参数
batch_size = 16
num_epochs = 100000000
learning_rate = 1e-3
num_classes = 9  # 类别数（9种岩石）

# 设备配置
device = torch.device('cuda')

# --- 数据预处理定义 ---
# CPU端：只做ToTensor和简单Resize/CenterCrop（无随机）
cpu_to_tensor = T_cpu.Compose([
    T_cpu.ToTensor(),          # 转Tensor
])

# GPU端增强：随机裁剪、翻转、归一化
gpu_train_transforms = T_gpu.Compose([
    # 随机裁剪：将图片裁剪到224x224
    # scale: 裁剪面积比例范围(0.6~1.0)
    # ratio: 裁剪宽高比范围(0.75~1.33)
    T_gpu.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33)),

    # 水平翻转：对称物体可以启用
    T_gpu.RandomHorizontalFlip(),

    # 垂直翻转：如果物体上下翻转仍有意义则可启用
    T_gpu.RandomVerticalFlip(),

    # 随机旋转
    # T_gpu.RandomRotation(30),

    # 转换数据类型到float32并归一化到[0,1]
    T_gpu.ToDtype(torch.float32, scale=True),

    # 标准化：使用ImageNet预训练模型的统计值
    # mean和std是ImageNet数据集的均值和标准差
    T_gpu.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# GPU端验证/测试：固定中心裁剪+归一化
gpu_valid_transforms = T_gpu.Compose([
    T_gpu.Resize((224, 224)),  # Resize 短边为256
    T_gpu.ToDtype(torch.float32, scale=True),
    T_gpu.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# --- 一次性加载所有图片到GPU ---
# 工具函数：加载目录下所有图像
def load_images_to_gpu(root_dir, cpu_transform, device):
    data = []
    labels = []
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(root_dir)))}
    
    # 先统计所有图片路径
    img_label_list = []
    for class_name, idx in class_to_idx.items():
        class_folder = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        for fname in os.listdir(class_folder):
            path = os.path.join(class_folder, fname)
            if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png') or fname.lower().endswith('.jpeg')):
                continue
            img_label_list.append((path, idx))
    
    # tqdm进度条遍历
    for path, idx in tqdm(img_label_list, desc=f"Loading {root_dir}"):
        img = Image.open(path).convert('RGB')
        tensor = cpu_transform(img)
        data.append(tensor)
        labels.append(idx)
    
    data_tensor = torch.stack(data).to(device, non_blocking=True)
    labels_tensor = torch.tensor(labels, device=device)
    return data_tensor, labels_tensor



# 加载训练、验证、测试数据
train_data, train_labels = load_images_to_gpu('data/train', cpu_to_tensor, device)
valid_data, valid_labels = load_images_to_gpu('data/valid', cpu_to_tensor, device)
test_data,  test_labels  = load_images_to_gpu('data/test',  cpu_to_tensor, device)

# --- 模型定义 ---
# 动态导入模型
model_module = importlib.import_module(f"models.{model_name}")
model = model_module.create_model(num_classes=num_classes)

# 打印可训练参数数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} / Total: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

# 搬到GPU并编译
model = model.to(device)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss(label_smoothing=0.2)  # 标签平滑
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.9, patience=1000, min_lr=1e-6
)

# 恢复训练状态
start_epoch = 0
best_accuracy = 0.0
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)
checkpoint_path = f'checkpoints/{model_module.get_model_name()}_best.pth'

if resume_training and os.path.exists(checkpoint_path):
    chk = torch.load(checkpoint_path)
    model.load_state_dict(chk['model_state_dict'])
    optimizer.load_state_dict(chk['optimizer_state_dict'])
    start_epoch = chk['epoch'] + 1
    best_accuracy = chk['best_accuracy']
    print(f"=> Resumed from epoch {start_epoch}, best acc = {best_accuracy:.2f}%")

# --- 训练与验证函数 ---
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    indices = torch.randperm(train_data.size(0), device=device)
    for i in range(0, train_data.size(0), batch_size):
        batch_idx = indices[i:i+batch_size]
        inputs = train_data[batch_idx]
        labels = train_labels[batch_idx]

        # GPU 上进行增强
        inputs = gpu_train_transforms(inputs)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    avg_loss = running_loss / (total / batch_size)
    print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, lr: {optimizer.param_groups[0]['lr']}")
    return acc, avg_loss


def validate(epoch):
    model.eval()
    correct_valid = 0
    total_valid = 0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        # 验证集评估
        for i in range(0, valid_data.size(0), batch_size):
            inputs = valid_data[i:i+batch_size]
            labels = valid_labels[i:i+batch_size]
            inputs = gpu_valid_transforms(inputs)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct_valid += (preds == labels).sum().item()
            total_valid += labels.size(0)
            
        # 测试集评估
        for i in range(0, test_data.size(0), batch_size):
            inputs = test_data[i:i+batch_size]
            labels = test_labels[i:i+batch_size]
            inputs = gpu_valid_transforms(inputs)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
            
    valid_acc = 100 * correct_valid / total_valid
    test_acc = 100 * correct_test / total_test
    print(f"Epoch {epoch}/{num_epochs} - Validation Acc: {valid_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    return valid_acc

csv_path = f'logs/{model_module.get_model_name()}_train_log.csv'

# 如果是重新训练，先新建文件并写表头
if not resume_training or not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train_Accuracy', 'Validation_Accuracy'])

# --- 主训练循环 ---
for epoch in range(start_epoch, num_epochs):
    train_acc, train_loss = train_one_epoch(epoch)
    valid_acc = validate(epoch)
    # 保存最优模型
    if valid_acc >= best_accuracy:
        best_accuracy = valid_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
        }, checkpoint_path)
        print(f"=> Saved checkpoint at epoch {epoch}, best acc = {best_accuracy:.2f}%")
    
    if epoch % 10 == 0:
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, valid_acc])

# --- 测试评估 ---
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for i in range(0, test_data.size(0), batch_size):
        inputs = test_data[i:i+batch_size]
        labels = test_labels[i:i+batch_size]
        inputs = gpu_valid_transforms(inputs)
        outputs = model(inputs)
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
# 拼接并计算准确率
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
test_acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
print(f"Test Accuracy: {test_acc * 100:.2f}%")
