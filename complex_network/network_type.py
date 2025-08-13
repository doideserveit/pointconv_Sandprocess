import os
import pandas as pd
import networkx as nx
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="比较实际网络与随机网络的平均最短路径和聚类系数")
    parser.add_argument('--stage', type=str, required=True, help='阶段名称')
    parser.add_argument('--repeat', type=int, default=5, help='随机网络重复次数，取平均')
    args = parser.parse_args()

    # 1. 读取实际网络的平均最短路径和聚类系数
    data_dir = '/share/home/202321008879/data/contact_network'
    contact_file = os.path.join(data_dir, f'{args.stage}_contacts.txt')  # 假设 contacts.txt 在每个阶段目录下
    avg_path_file = os.path.join(data_dir, args.stage, 'avg_shortest_path.csv')  # 假设文件名为 avg_shortest_path.csv
    clustering_file = os.path.join(data_dir, args.stage, 'clustering.csv')  # 假设文件名为 clustering.csv
    df_path = pd.read_csv(avg_path_file)
    avg_path = df_path['avg_shortest_path'].mean()
    print(f"实际网络平均最短路径长度: {avg_path:.4f}")

    df_clust = pd.read_csv(clustering_file)
    avg_clust = df_clust['clustering'].mean()
    print(f"实际网络平均聚类系数: {avg_clust:.4f}")

    # 2. 读取网络，获取节点数和边数
    edges = []
    with open(contact_file, 'r') as f:
        for line in f:
            a, b = map(int, line.strip().split(','))
            edges.append((a, b))
    nodes = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
    n = len(nodes)
    m = len(edges)
    print(f"实际网络节点数: {n}，边数: {m}")

    # 3. 生成同规模随机网络并计算平均最短路径和聚类系数
    path_list = []
    clust_list = []
    p = 2 * m / (n * (n - 1)) if n > 1 else 0
    for i in range(args.repeat):
        G_rand = nx.erdos_renyi_graph(n, p)
        # 平均最短路径
        if nx.is_connected(G_rand):
            avg_path_rand = nx.average_shortest_path_length(G_rand)
        else:
            largest_cc = max(nx.connected_components(G_rand), key=len)
            subgraph = G_rand.subgraph(largest_cc)
            avg_path_rand = nx.average_shortest_path_length(subgraph)
        path_list.append(avg_path_rand)
        # 平均聚类系数
        clust_list.append(nx.average_clustering(G_rand))
    avg_rand_path = np.mean(path_list)
    avg_rand_clust = np.mean(clust_list)
    print(f"{args.repeat}次随机网络平均最短路径长度: {avg_rand_path:.4f}")
    print(f"{args.repeat}次随机网络平均聚类系数: {avg_rand_clust:.4f}")

    # 4. 对比输出
    print("\n对比结果：")
    print("若你的网络平均最短路径与随机网络接近且聚类系数远高于随机网络，则为小世界网络。")
    print("若平均最短路径远大于随机网络，则更像规则网络。")
    print("若两者都接近，则更像随机网络。")

if __name__ == '__main__':
    main()