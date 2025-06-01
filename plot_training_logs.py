import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path

def plot_training_logs():
    """从logs目录读取训练日志并绘制图表"""
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 创建结果保存目录
    os.makedirs('training_plots', exist_ok=True)
    
    # 获取所有CSV日志文件
    log_files = glob.glob('logs/*_train_log.csv')
    
    if not log_files:
        print("No training log files found! Please ensure there are *_train_log.csv files in the logs directory")
        return
    
    print(f"Found {len(log_files)} training log files:")
    for file in log_files:
        print(f"  - {file}")
    
    # 读取所有日志数据
    all_data = {}
    for log_file in log_files:
        model_name = Path(log_file).stem.replace('_train_log', '')
        try:
            df = pd.read_csv(log_file)
            # 检查列名是否正确
            expected_columns = ['Epoch', 'Train Loss', 'Train_Accuracy', 'Validation_Accuracy']
            if not all(col in df.columns for col in expected_columns):
                print(f"Warning: Column names in {log_file} do not match expected format")
                print(f"Expected columns: {expected_columns}")
                print(f"Actual columns: {list(df.columns)}")
                continue
            
            all_data[model_name] = df
            print(f"Successfully loaded {model_name}: {len(df)} epochs of data")
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    if not all_data:
        print("No log files were successfully loaded!")
        return
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training Progress Comparison', fontsize=16, fontweight='bold')
    
    # 1. 训练损失对比
    ax1 = axes[0]
    for model_name, df in all_data.items():
        ax1.plot(df['Epoch'], df['Train Loss'], label=model_name, linewidth=2, marker='o', markersize=3)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 训练vs验证准确率对比（所有模型）
    ax2 = axes[1]
    for model_name, df in all_data.items():
        ax2.plot(df['Epoch'], df['Train_Accuracy'], label=f'{model_name} (Train)', 
                linewidth=2, marker='o', markersize=2, linestyle='-')
        ax2.plot(df['Epoch'], df['Validation_Accuracy'], label=f'{model_name} (Val)', 
                linewidth=2, marker='^', markersize=2, linestyle='--')
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_plots/training_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved training comparison plot: training_plots/training_comparison.png")
    
    # 为每个模型单独创建详细图表
    for model_name, df in all_data.items():
        create_individual_plot(model_name, df)
    
    # 创建最终结果汇总表
    create_summary_table(all_data)
    
    plt.show()

def create_individual_plot(model_name, df):
    """为单个模型创建详细的训练图表"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{model_name} - Training Details', fontsize=16, fontweight='bold')
    
    # 1. 训练损失
    axes[0].plot(df['Epoch'], df['Train Loss'], color='red', linewidth=2, marker='o', markersize=4)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 训练vs验证准确率
    axes[1].plot(df['Epoch'], df['Train_Accuracy'], label='Training', 
                   color='blue', linewidth=2, marker='o', markersize=3)
    axes[1].plot(df['Epoch'], df['Validation_Accuracy'], label='Validation', 
                   color='green', linewidth=2, marker='^', markersize=3)
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_plots/{model_name}_detailed.png', dpi=300, bbox_inches='tight')
    print(f"Saved detailed plot for model: training_plots/{model_name}_detailed.png")
    plt.close()

def create_summary_table(all_data):
    """创建训练结果汇总表"""
    summary_data = []
    
    for model_name, df in all_data.items():
        if len(df) > 0:
            final_epoch = df['Epoch'].iloc[-1]
            final_train_acc = df['Train_Accuracy'].iloc[-1]
            final_val_acc = df['Validation_Accuracy'].iloc[-1]
            best_val_acc = df['Validation_Accuracy'].max()
            best_val_epoch = df.loc[df['Validation_Accuracy'].idxmax(), 'Epoch']
            final_loss = df['Train Loss'].iloc[-1]
            min_loss = df['Train Loss'].min()
            
            summary_data.append({
                'Model': model_name,
                'Final Epoch': int(final_epoch),
                'Final Train Acc (%)': f"{final_train_acc:.2f}",
                'Final Val Acc (%)': f"{final_val_acc:.2f}",
                'Best Val Acc (%)': f"{best_val_acc:.2f}",
                'Best Val Epoch': int(best_val_epoch),
                'Final Loss': f"{final_loss:.4f}",
                'Min Loss': f"{min_loss:.4f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存为CSV
    summary_df.to_csv('training_plots/training_summary.csv', index=False)
    
    # 创建表格图像
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置表格样式
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Training Results Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('training_plots/training_summary.png', dpi=300, bbox_inches='tight')
    print("Saved training summary table: training_plots/training_summary.png")
    print("Saved training summary table: training_plots/training_summary.csv")
    plt.close()
    
    # 打印汇总信息
    print("\n========== Training Results Summary ==========")
    print(summary_df.to_string(index=False))
    print("=" * 50)

if __name__ == "__main__":
    plot_training_logs() 