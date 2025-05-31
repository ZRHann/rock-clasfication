import os
import re
import shutil

# 路径设置
input_root = "data/train"  # 每个子目录是一个大类
output_root = "clustered_data/train"

# 清理函数：去除数字
def clean_name(name):
    return re.sub(r'\d+', '', name)

# 计算最长公共前缀
def longest_common_prefix(s1, s2):
    i = 0
    while i < min(len(s1), len(s2)) and s1[i] == s2[i]:
        i += 1
    return s1[:i]

# 对某一目录中的文件按公共前缀聚类
def cluster_by_prefix(filenames, min_prefix_len=10):
    clusters = []
    assigned = set()

    for i, f1 in enumerate(filenames):
        if f1 in assigned:
            continue
        base_name1 = clean_name(f1)
        cluster = [f1]
        for j in range(i + 1, len(filenames)):
            f2 = filenames[j]
            if f2 in assigned:
                continue
            base_name2 = clean_name(f2)
            prefix = longest_common_prefix(base_name1, base_name2)
            if len(prefix) >= min_prefix_len:
                cluster.append(f2)
                assigned.add(f2)
        assigned.add(f1)
        clusters.append(cluster)

    return clusters

# 处理每个子类文件夹
def process_subfolder(subfolder_path, output_subroot):
    files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    clusters = cluster_by_prefix(files)
    subfolder_name = os.path.basename(subfolder_path)
    for idx, cluster in enumerate(clusters):
        cluster_out_dir = os.path.join(output_subroot, f"{subfolder_name}_subclass_{idx}")
        os.makedirs(cluster_out_dir, exist_ok=True)
        for fname in cluster:
            src = os.path.join(subfolder_path, fname)
            dst = os.path.join(cluster_out_dir, fname)
            shutil.copy2(src, dst)

# 主执行逻辑
if __name__ == "__main__":
    os.makedirs(output_root, exist_ok=True)
    for class_name in os.listdir(input_root):
        class_path = os.path.join(input_root, class_name)
        if not os.path.isdir(class_path):
            continue
        print(f"📂 正在细分大类：{class_name}")
        output_class_dir = os.path.join(output_root, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        process_subfolder(class_path, output_class_dir)

    print("✅ 所有子类细分完成，输出到:", output_root)