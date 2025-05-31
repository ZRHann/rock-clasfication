import os
import torch
import torch.nn as nn
import torchvision.transforms as T_cpu
import torchvision.transforms.v2 as T_gpu
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import importlib.util
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# 配置要使用的模型和权重
model_configs = [
    {'name': 'convnext_large', 'weight': 1},
    {'name': 'coatnet_3', 'weight': 1},
    {'name': 'efficientnet_b5', 'weight': 1.1}, 
    {'name': 'convnext_xxlarge', 'weight': 10}
]


def load_images_to_gpu(root_dir, cpu_transform, device):
    """从train.py复制的数据加载函数"""
    data = []
    labels = []
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(root_dir)))}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
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
            img_label_list.append((path, idx, fname))  # 添加文件名用于错误分析
    
    # tqdm进度条遍历
    for path, idx, fname in tqdm(img_label_list, desc=f"Loading {root_dir}"):
        img = Image.open(path).convert('RGB')
        tensor = cpu_transform(img)
        data.append(tensor)
        labels.append(idx)
    
    data_tensor = torch.stack(data).to(device, non_blocking=True)
    labels_tensor = torch.tensor(labels, device=device)
    return data_tensor, labels_tensor, img_label_list, idx_to_class

def load_model_from_file(model_file, num_classes=9):
    """从.py文件动态加载模型"""
    module_name = os.path.splitext(os.path.basename(model_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    model = module.create_model(num_classes=num_classes)
    model_name = module.get_model_name()
    return model, model_name

def load_checkpoint(model, checkpoint_path):
    """加载模型检查点"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
    return model

def plot_confusion_matrix(cm, classes, save_path):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_error_analysis(predictions, labels, img_paths, idx_to_class, save_path):
    """保存错误分析结果"""
    with open(save_path, 'w') as f:
        f.write("Error Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        for pred, true_label, (img_path, _, fname) in zip(predictions, labels, img_paths):
            if pred != true_label:
                f.write(f"Image: {fname}\n")
                f.write(f"True class: {idx_to_class[true_label]}\n")
                f.write(f"Predicted class: {idx_to_class[pred]}\n")
                f.write("-" * 30 + "\n")

class EnsembleModel:
    def __init__(self, model_configs, device):
        """
        Args:
            model_configs: 模型配置列表，每个配置是一个字典，包含:
                - name: 模型名称 (必需)
                - weight: 模型权重 (可选，默认为1.0)
            device: 运行设备
        """
        self.device = device
        self.model_names = []
        self.models = []
        weights = []
        
        # 加载每个模型
        for config in model_configs:
            name = config['name']
            self.model_names.append(name)
            
            # 从models文件夹加载模型
            model_file = f'models/{name}.py'
            try:
                model, _ = load_model_from_file(model_file)
                # 加载检查点
                checkpoint_path = f'checkpoints/{name}_best.pth'
                model = load_checkpoint(model, checkpoint_path)
                model = model.to(device)
                self.models.append(model)
                weights.append(config.get('weight', 1.0))  # 如果没有设置权重，默认为1.0
                print(f"Successfully loaded model: {name}")
            except Exception as e:
                print(f"Failed to load model {name}: {str(e)}")
                raise e
            
        # 确保权重和为1
        weights = torch.tensor(weights, device=device)
        self.weights = weights / weights.sum()
        
    def eval(self):
        for model in self.models:
            model.eval()
            
    def predict(self, x):
        predictions = []
        # 获取每个模型的预测
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(torch.softmax(pred, dim=1))
        
        # 加权集成
        predictions = torch.stack(predictions)  # [num_models, batch_size, num_classes]
        weighted_preds = predictions * self.weights.view(-1, 1, 1)  # 广播权重到每个预测
        ensemble_pred = weighted_preds.sum(dim=0)  # [batch_size, num_classes]
        return ensemble_pred

def evaluate_model(model, data, labels, transforms, device, batch_size=32):
    """评估单个模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, data.size(0), batch_size):
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # 应用GPU端变换
            inputs = transforms(batch_data)
            outputs = model(inputs)
            preds = torch.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算准确率
    pred_labels = np.argmax(all_preds, axis=1)
    accuracy = accuracy_score(all_labels, pred_labels)
    
    return accuracy, pred_labels, all_labels

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建结果目录
    os.makedirs('eval_results', exist_ok=True)
    
    # 数据预处理
    cpu_transform = T_cpu.Compose([
        T_cpu.ToTensor(),
    ])
    
    gpu_transform = T_gpu.Compose([
        T_gpu.Resize((224, 224)),
        T_gpu.ToDtype(torch.float32, scale=True),
        T_gpu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据集
    test_data, test_labels, test_img_paths, idx_to_class = load_images_to_gpu(
        'data/test', cpu_transform, device
    )
    class_names = list(idx_to_class.values())
    
    
    
    # 创建和评估集成模型
    print("\nEvaluating ensemble model:")
    ensemble = EnsembleModel(model_configs, device)
    ensemble.eval()
    
    # 评估每个单独的模型
    print("\nEvaluating individual models:")
    individual_results = {}
    for model, model_name in zip(ensemble.models, ensemble.model_names):
        print(f"\nEvaluating {model_name}...")
        accuracy, pred_labels, true_labels = evaluate_model(
            model, test_data, test_labels, gpu_transform, device
        )
        individual_results[model_name] = accuracy
        
        # 生成并保存混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)
        plot_confusion_matrix(
            cm, class_names,
            f'eval_results/{model_name}_confusion_matrix.png'
        )
        
        # 保存分类报告
        report = classification_report(true_labels, pred_labels, 
                                    target_names=class_names)
        with open(f'eval_results/{model_name}_classification_report.txt', 'w') as f:
            f.write(report)
        
        # 保存错误分析
        save_error_analysis(
            pred_labels, true_labels, test_img_paths,
            idx_to_class, f'eval_results/{model_name}_error_analysis.txt'
        )
        
        print(f"{model_name}: {accuracy:.4f}")
    
    print("\nEnsemble Model Weights:")
    for name, weight in zip(ensemble.model_names, ensemble.weights.cpu().numpy()):
        print(f"{name}: {weight:.4f}")
    print()
    
    all_ensemble_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, test_data.size(0), 32):
            batch_data = test_data[i:i+32]
            batch_labels = test_labels[i:i+32]
            
            # 应用GPU端变换
            inputs = gpu_transform(batch_data)
            ensemble_outputs = ensemble.predict(inputs)
            all_ensemble_preds.extend(ensemble_outputs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_ensemble_preds = np.array(all_ensemble_preds)
    all_labels = np.array(all_labels)
    ensemble_pred_labels = np.argmax(all_ensemble_preds, axis=1)
    ensemble_accuracy = accuracy_score(all_labels, ensemble_pred_labels)
    
    # 生成集成模型的评估结果
    cm = confusion_matrix(all_labels, ensemble_pred_labels)
    plot_confusion_matrix(
        cm, class_names,
        'eval_results/ensemble_confusion_matrix.png'
    )
    
    report = classification_report(all_labels, ensemble_pred_labels,
                                target_names=class_names)
    with open('eval_results/ensemble_classification_report.txt', 'w') as f:
        f.write(report)
    
    save_error_analysis(
        ensemble_pred_labels, all_labels, test_img_paths,
        idx_to_class, 'eval_results/ensemble_error_analysis.txt'
    )
    
    # 打印最终结果
    print("\nFinal Results:")
    print("=" * 50)
    for model_name, acc in individual_results.items():
        print(f"{model_name}: {acc:.4f}")
    print("-" * 50)
    print(f"Ensemble Model: {ensemble_accuracy:.4f}")
    print("=" * 50)
    
    print("\nEvaluation results have been saved to 'eval_results' directory")

if __name__ == "__main__":
    main() 