import argparse
import os
import networkx as nx
import pandas as pd
from typing import Dict, Callable, Any, List


class NetworkAnalyzer:
    """简化版复杂网络分析工具"""
    
    def __init__(self, contact_file: str):
        self.contact_file = contact_file
        self.graph = None
        
    def load_graph(self) -> nx.Graph:
        """加载网络数据并构建图"""
        if self.graph is None:
            print(f"加载网络数据: {self.contact_file}")
            edges = self.read_contacts(self.contact_file)
            self.graph = nx.Graph()
            self.graph.add_edges_from(edges)
            print(f"网络加载完成: {self.graph.number_of_nodes()} 个节点, {self.graph.number_of_edges()} 条边")
        return self.graph
    
    @staticmethod
    def read_contacts(contact_file: str) -> List[tuple]:
        """读取接触网络数据文件"""
        edges = []
        with open(contact_file, 'r') as f:
            for line in f:
                a, b = map(int, line.strip().split(','))
                edges.append((a, b))
        return edges
    
    def compute_degree(self) -> Dict[int, int]:
        """计算节点度"""
        return dict(self.graph.degree())
    
    def compute_clustering(self) -> Dict[int, float]:
        """计算聚类系数"""
        return nx.clustering(self.graph)
    
    def compute_avg_shortest_path(self) -> Dict[int, float]:
        """计算平均最短路径长度"""
        lengths = {}
        for i, component in enumerate(nx.connected_components(self.graph)):
            subgraph = self.graph.subgraph(component)
            if len(subgraph) > 1:
                avg_len = nx.average_shortest_path_length(subgraph)
                for node in subgraph.nodes:
                    lengths[node] = avg_len
            else:
                for node in subgraph.nodes:
                    lengths[node] = 0.0
        return lengths
    
    # 这里可以添加新的指标计算函数
    def compute_betweenness_centrality(self) -> Dict[int, float]:
        """计算介数中心性"""
        # 介数中心性是复杂网络分析中的一个重要指标，用于衡量节点在网络中作为 "桥梁" 的重要性。它反映了节点在多大程度上位于其他节点对之间的最短路径上。
        return nx.betweenness_centrality(self.graph)
    
    def compute_closeness_centrality(self) -> Dict[int, float]:
        """计算接近中心性"""
        # 衡量节点到其他所有节点的平均距离，反映信息传播效率
        return nx.closeness_centrality(self.graph)
    
    
    def save_metric(self, metric_name: str, result: Dict[int, Any], output_dir: str) -> None:
        """保存指标计算结果到CSV文件"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存结果
        output_file = os.path.join(output_dir, f"{metric_name}.csv")
        df = pd.DataFrame(list(result.items()), columns=['node', metric_name])
        df.to_csv(output_file, index=False)
        print(f"指标 '{metric_name}' 已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description="简化版复杂网络分析工具")
    parser.add_argument('--type', type=str, required=True,
                        help='应变阶段体系，例如 origin, load1, load5, load10, load15')
    parser.add_argument('--contact_file', type=str, default=None, help='输入的contacts.txt路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--metrics', type=str, nargs='+', default=None, 
                        help='要计算的网络参数，可选: degree, clustering, avg_shortest_path, betweenness, closeness, eigenvector')
    args = parser.parse_args()
    
    # 自动补全目录
    if args.contact_file is None:
        args.contact_file = f'/share/home/202321008879/data/contact_network/{args.type}_contacts.txt'
    if args.output_dir is None:
        args.output_dir = f'/share/home/202321008879/data/contact_network/{args.type}'

     # 创建分析器实例
    analyzer = NetworkAnalyzer(args.contact_file)
    
    # 加载网络
    analyzer.load_graph()
    
    # 可用的指标计算函数映射
    metric_functions = {
        'degree': analyzer.compute_degree,
        'clustering': analyzer.compute_clustering,
        'avg_shortest_path': analyzer.compute_avg_shortest_path,
        'betweenness': analyzer.compute_betweenness_centrality,
        'closeness': analyzer.compute_closeness_centrality,
        # 可以在这里添加新的指标映射
    }
    
    # 计算并保存指定的指标
    if args.metrics is None:
        print("未指定指标，默认计算所有可用指标")
        args.metrics = list(metric_functions.keys())
    for metric in args.metrics:
        if metric in metric_functions:
            print(f"\n计算指标: {metric}")
            result = metric_functions[metric]()
            analyzer.save_metric(metric, result, args.output_dir)
            print(f"指标 '{metric}' 计算完成")
        else:
            print(f"警告: 未知的指标 '{metric}'，已忽略")
            print(f"可用指标: {list(metric_functions.keys())}")

if __name__ == '__main__':
    main()
