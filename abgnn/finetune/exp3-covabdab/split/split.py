import pandas as pd
import subprocess
import os
from collections import defaultdict

def extract_cdrh3(csv_file):
    df = pd.read_csv(csv_file)
    # 去除重复序列
    unique_sequences = list(dict.fromkeys(df['CDRH3'].tolist()))
    return unique_sequences

def write_fasta(sequences, file_path):
    with open(file_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">sequence_{i}\n{seq}\n")

def cluster_sequences(sequences, identity_threshold=0.8):
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    
    # 写入FASTA文件
    write_fasta(sequences, 'tmp/input.fasta')
    
    # 运行MMseqs2聚类
    subprocess.run(['mmseqs', 'createdb', 'tmp/input.fasta', 'tmp/seqDB'], capture_output=True)
    subprocess.run([
        'mmseqs', 'cluster',
        'tmp/seqDB',
        'tmp/cluDB',
        'tmp',
        '--min-seq-id', str(identity_threshold)
    ], capture_output=True)
    subprocess.run([
        'mmseqs', 'createtsv',
        'tmp/seqDB',
        'tmp/seqDB',
        'tmp/cluDB',
        'tmp/clusters.tsv'
    ], capture_output=True)
    
    # 解析聚类结果
    clusters = defaultdict(list)
    sequence_dict = {f"sequence_{i}": seq for i, seq in enumerate(sequences)}
    
    with open('tmp/clusters.tsv', 'r') as f:
        for line in f:
            rep, member = line.strip().split('\t')
            if member in sequence_dict:
                seq = sequence_dict[member]
                clusters[rep].append(seq)
    
    # 清理临时文件
    subprocess.run(['rm', '-rf', 'tmp'], capture_output=True)
    
    return list(clusters.values())

def split_data(clusters, train_ratio=0.7, val_ratio=0.15):
    """
    修改后的分割函数，确保验证集和测试集大小相近
    train_ratio = 0.7  # 70% 训练集
    val_ratio = 0.15   # 15% 验证集
    test_ratio = 0.15  # 15% 测试集
    """
    train_seqs = []
    val_seqs = []
    test_seqs = []
    
    # 按簇的大小排序，从大到小
    sorted_clusters = sorted(clusters, key=len, reverse=True)
    
    for cluster in sorted_clusters:
        size = len(cluster)
        
        if size == 1:
            # 单个序列放入训练集
            train_seqs.extend(cluster)
        elif size == 2:
            # 两个序列，一个训练集，一个验证集
            train_seqs.append(cluster[0])
            val_seqs.append(cluster[1])
        elif size == 3:
            # 三个序列分别放入三个集合
            train_seqs.append(cluster[0])
            val_seqs.append(cluster[1])
            test_seqs.append(cluster[2])
        else:
            # 计算每个集合应得的序列数
            train_size = int(size * train_ratio)
            val_size = int(size * val_ratio)
            test_size = size - train_size - val_size
            
            # 确保每个集合至少有一个样本
            if train_size < 1:
                train_size = 1
            if val_size < 1:
                val_size = 1
            if test_size < 1:
                test_size = 1
                
            # 调整分配以确保总数正确
            total = train_size + val_size + test_size
            if total > size:
                # 如果总数超过了簇的大小，需要调整
                excess = total - size
                while excess > 0:
                    if val_size > 1:
                        val_size -= 1
                        excess -= 1
                    elif test_size > 1:
                        test_size -= 1
                        excess -= 1
                    elif train_size > 1:
                        train_size -= 1
                        excess -= 1
            
            # 分配序列
            train_seqs.extend(cluster[:train_size])
            val_seqs.extend(cluster[train_size:train_size + val_size])
            test_seqs.extend(cluster[train_size + val_size:])
    
    # 去除重复
    train_seqs = list(dict.fromkeys(train_seqs))
    val_seqs = list(dict.fromkeys([x for x in val_seqs if x not in train_seqs]))
    test_seqs = list(dict.fromkeys([x for x in test_seqs if x not in train_seqs and x not in val_seqs]))
    
    # 平衡验证集和测试集
    while len(val_seqs) > len(test_seqs) + 5:  # 允许5个序列的差异
        # 将一些验证集的序列移到测试集
        test_seqs.append(val_seqs.pop())
    
    return train_seqs, val_seqs, test_seqs

def main():
    csv_file = 'covabdab.csv'  # 替换为实际的CSV文件名
    sequences = extract_cdrh3(csv_file)
    print(f"Initial unique sequences: {len(sequences)}")
    
    # 聚类
    clusters = cluster_sequences(sequences)
    print(f"Number of clusters: {len(clusters)}")
    
    # 分割数据集
    train_set, val_set, test_set = split_data(clusters)
    
    # 写入结果
    with open('dataset.txt', 'w') as f:
        for seq in train_set:
            f.write(f"{seq} train\n")
        for seq in val_set:
            f.write(f"{seq} val\n")
        for seq in test_set:
            f.write(f"{seq} test\n")
    
    # 输出统计信息
    total_split = len(train_set) + len(val_set) + len(test_set)
    print(f"Original unique sequences: {len(sequences)}")
    print(f"Training set size: {len(train_set)} ({len(train_set)/total_split*100:.1f}%)")
    print(f"Validation set size: {len(val_set)} ({len(val_set)/total_split*100:.1f}%)")
    print(f"Test set size: {len(test_set)} ({len(test_set)/total_split*100:.1f}%)")
    print(f"Total sequences in split: {total_split}")
    
    if total_split != len(sequences):
        print(f"Warning: Number of sequences changed during processing!")
        print(f"Missing sequences: {len(sequences) - total_split}")

if __name__ == "__main__":
    main()
