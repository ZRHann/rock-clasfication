import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T_cpu
import torchvision.transforms.v2 as T_gpu
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from PIL import Image
import csv
from collections import defaultdict
torch.set_float32_matmul_precision('high')
# 配置是否从checkpoint恢复训练
resume_training = True

# 超参数
batch_size = 64
num_epochs = 100000000
learning_rate = 1e-3

# 设备配置
device = torch.device('cuda')

# --- 数据预处理定义 ---
# CPU端：只做ToTensor和简单Resize/CenterCrop（无随机）
cpu_to_tensor = T_cpu.Compose([
    T_cpu.ToTensor(),          # 转Tensor
])

# GPU端增强：随机裁剪、翻转、归一化
gpu_train_transforms = T_gpu.Compose([
    T_gpu.Resize(256),
    T_gpu.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    T_gpu.RandomHorizontalFlip(),
    T_gpu.RandomVerticalFlip(p=0.3),
    T_gpu.RandomRotation(360),
    T_gpu.RandomAffine(degrees=0, shear=(-15, 15, -15, 15)),
    T_gpu.RandomPerspective(distortion_scale=0.3, p=0.5),
    T_gpu.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
    T_gpu.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.05),
    T_gpu.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0)),
    T_gpu.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    T_gpu.ToDtype(torch.float32, scale=True),
])
# GPU端验证/测试：固定中心裁剪+归一化
gpu_valid_transforms = T_gpu.Compose([
    T_gpu.Resize((256, 256)),  # Resize 短边为256
    T_gpu.ToDtype(torch.float32, scale=True),
])

# --- 一次性加载所有图片到GPU ---
# 工具函数：加载目录下所有图像
def load_train_images_to_gpu(train_subdirs, cpu_transform, device):
    data = []
    labels = []
    img_label_list = []
    for sub_path, sub_id in train_subdirs:
        for fname in os.listdir(sub_path):
            path = os.path.join(sub_path, fname)
            img_label_list.append((path, sub_id))
    for path, idx in tqdm(img_label_list, desc="Loading train"):
        img = Image.open(path).convert('RGB')
        tensor = cpu_transform(img)
        data.append(tensor)
        labels.append(idx)

    data_tensor = torch.stack(data).to(device, non_blocking=True)
    labels_tensor = torch.tensor(labels, device=device)
    return data_tensor, labels_tensor

def load_flat_images_to_gpu(root_dir, cpu_transform, parent_name_to_id, device):
    data = []
    labels = []
    img_label_list = []

    for parent_name, parent_id in sorted(parent_name_to_id.items()):
        class_path = os.path.join(root_dir, parent_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(class_path, fname)
            img_label_list.append((path, parent_id))

    for path, idx in tqdm(img_label_list, desc=f"Loading {root_dir}"):
        img = Image.open(path).convert('RGB')
        tensor = cpu_transform(img)
        data.append(tensor)
        labels.append(idx)

    data_tensor = torch.stack(data).to(device, non_blocking=True)
    labels_tensor = torch.tensor(labels, device=device)
    return data_tensor, labels_tensor

def build_subclass_hierarchy(train_root):
    """
    根据两层目录结构建立子类 → 父类 的映射表，并返回训练子目录路径与子类标签。

    Parameters:
        train_root (str): 训练数据的根目录路径

    Returns:
        train_subdirs (List[Tuple[str, int]]): 每个子类文件夹路径与其子类标签索引
        subclass_to_parent (Dict[int, int]): 子类标签索引 → 父类标签索引
        class_to_idx (Dict[str, int]): 子类文件夹名 → 子类标签索引
        parent_name_to_id (Dict[str, int]): 父类文件夹名 → 父类标签索引
    """
    subclass_to_parent = {}
    parent_name_to_id = {}
    parent_id_counter = 0
    subclass_id_counter = 0
    class_to_idx = {}

    train_subdirs = []

    for parent in sorted(os.listdir(train_root)):
        parent_path = os.path.join(train_root, parent)
        if not os.path.isdir(parent_path):
            continue
        if parent not in parent_name_to_id:
            parent_name_to_id[parent] = parent_id_counter
            parent_id_counter += 1
        for sub in sorted(os.listdir(parent_path)):
            sub_path = os.path.join(parent_path, sub)
            if not os.path.isdir(sub_path):
                continue
            subclass_to_parent[subclass_id_counter] = parent_name_to_id[parent]
            class_to_idx[sub] = subclass_id_counter
            train_subdirs.append((sub_path, subclass_id_counter))
            subclass_id_counter += 1

    return train_subdirs, subclass_to_parent, class_to_idx, parent_name_to_id

train_subdirs, subclass_to_parent, class_to_idx, parent_name_to_id = build_subclass_hierarchy('clustered_data/train')
num_classes = len(class_to_idx)
num_parents = len(parent_name_to_id)  # 大类数量
parent_class_to_idx = parent_name_to_id
print(f"Found {num_classes} subclasses and {num_parents} parent classes.")
print(f"Subclasses: {class_to_idx}")
print(f"parent_class_to_idx: {parent_class_to_idx}")
# 加载训练、验证、测试数据
train_data, train_labels = load_train_images_to_gpu(train_subdirs, cpu_to_tensor, device)
valid_data, valid_labels = load_flat_images_to_gpu('data/valid', cpu_to_tensor, parent_class_to_idx, device)
test_data,  test_labels  = load_flat_images_to_gpu('data/test',  cpu_to_tensor, parent_class_to_idx, device)

# --- 模型定义 ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# 替换最后一层为9分类
# model.fc = nn.Linear(model.fc.in_features, num_classes)
model.fc = nn.Sequential(
    # nn.Dropout(p=0.1),  # Dropout 概率 50%
    nn.Linear(model.fc.in_features, num_classes)
)
# 解冻部分层
for name, param in model.named_parameters():
    if name.startswith("fc"):
        param.requires_grad = True
    else:
        param.requires_grad = True
# 搬到GPU并编译
model = model.to(device)
model = torch.compile(model)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.9, patience=10000, min_lr=1e-8
)

# 恢复训练状态
start_epoch = 0
latest_accuracy = 0.0
checkpoint_path = 'best_checkpoint.pth'
if resume_training and os.path.exists(checkpoint_path):
    chk = torch.load(checkpoint_path)
    model.load_state_dict(chk['model_state_dict'])
    optimizer.load_state_dict(chk['optimizer_state_dict'])
    start_epoch = chk['epoch'] + 1
    latest_accuracy = chk['latest_accuracy']
    print(f"=> Resumed from epoch {start_epoch}, best acc = {latest_accuracy:.2f}%")

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
    correct = 0
    total = 0

    # 每类的总数与正确数统计字典
    class_total = defaultdict(int)
    class_correct = defaultdict(int)

    with torch.no_grad():
        for i in range(0, valid_data.size(0), batch_size):
            inputs = valid_data[i:i+batch_size]
            labels = valid_labels[i:i+batch_size]
            inputs = gpu_valid_transforms(inputs)
            outputs = model(inputs)

            preds_subclass = outputs.argmax(dim=1)
            preds_parent = torch.tensor([subclass_to_parent[i.item()] for i in preds_subclass], device=device)
            labels_parent = labels  # valid_labels 已经是 parent class 的标签

            for pred, label in zip(preds_parent, labels_parent):
                class_total[label.item()] += 1
                if pred.item() == label.item():
                    class_correct[label.item()] += 1

            correct += (preds_parent == labels_parent).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch}/{num_epochs} - Validation Acc: {acc:.2f}%")

    print("Per-class accuracy:")
    for class_id in sorted(class_total.keys()):
        acc_cls = 100 * class_correct[class_id] / class_total[class_id]
        print(f"  Class {class_id} ({acc_cls:.2f}% - {class_correct[class_id]}/{class_total[class_id]})")

    return acc

csv_path = 'train_log.csv'

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
    if valid_acc > latest_accuracy:
        latest_accuracy = valid_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'latest_accuracy': latest_accuracy,
        }, checkpoint_path)
        print(f"=> Saved checkpoint at epoch {epoch}, best acc = {latest_accuracy:.2f}%")
    
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
