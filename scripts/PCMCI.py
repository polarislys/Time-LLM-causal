# 安装依赖
# pip install tigramite matplotlib networkx

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import networkx as nx
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import plotting as tp

class CausalGraphBuilder:
    def __init__(self, data, var_names):
        """
        data: numpy array, shape (T, N) - T个时间步，N个变量
        var_names: list of str - 变量名称
        """
        self.data = data
        self.var_names = var_names
        self.dataframe = pp.DataFrame(data, var_names=var_names)
        
    def build_causal_graph(self, tau_max=5, pc_alpha=0.05):
        """
        使用 PCMCI 构建因果图
        tau_max: 最大时间滞后
        pc_alpha: 显著性水平
        """
        # 使用偏相关作为独立性检验
        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=self.dataframe, cond_ind_test=parcorr, verbosity=1)
        
        # 运行 PCMCI
        results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
        
        self.causal_graph = results['graph']
        self.p_matrix = results['p_matrix']
        self.val_matrix = results['val_matrix']
        
        return results
    
    def get_causal_relationships(self):
        """提取因果关系"""
        relationships = []
        n_vars = len(self.var_names)
        
        for j in range(n_vars):
            for i in range(n_vars):
                for tau in range(self.causal_graph.shape[2]):
                    if self.causal_graph[i, j, tau] != "":
                        link_type = self.causal_graph[i, j, tau]
                        strength = self.val_matrix[i, j, tau]
                        p_value = self.p_matrix[i, j, tau]
                        
                        relationships.append({
                            'cause': self.var_names[i],
                            'effect': self.var_names[j],
                            'lag': tau,
                            'type': link_type,
                            'strength': strength,
                            'p_value': p_value
                        })
        
        return relationships
    
    def visualize_causal_graph(self, output_dir='./causal_results'):
        """可视化因果图"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        relationships = self.get_causal_relationships()
        
        # 1. 使用 tigramite 自带的绘图功能
        print("生成 tigramite 原生因果图...")
        fig, ax = plt.subplots(figsize=(12, 8))
        tp.plot_graph(
            val_matrix=self.val_matrix,
            graph=self.causal_graph,
            var_names=self.var_names,
            link_colorbar_label='cross-correlation',
            node_colorbar_label='auto-correlation',
            fig_ax=(fig, ax),
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/causal_graph_tigramite.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存到: {output_dir}/causal_graph_tigramite.png")
        
        # 2. 网络图可视化（使用 networkx）
        print("生成网络图...")
        self._plot_network_graph(relationships, output_dir)
        
        # 3. 热力图可视化
        print("生成热力图...")
        self._plot_heatmap(output_dir)
        
        # 4. 时间滞后分布图
        print("生成时间滞后分布图...")
        self._plot_lag_distribution(relationships, output_dir)
        
        # 5. 因果强度排名
        print("生成因果强度排名图...")
        self._plot_strength_ranking(relationships, output_dir)
        
        print(f"\n所有可视化结果已保存到: {output_dir}/")
    
    def _plot_network_graph(self, relationships, output_dir):
        """绘制网络图"""
        G = nx.DiGraph()
        
        # 添加节点
        for var in self.var_names:
            G.add_node(var)
        
        # 添加边（只显示 lag=0 和 lag=1 的关系，避免过于复杂）
        edge_labels = {}
        for rel in relationships:
            if rel['lag'] <= 1:  # 只显示即时和1步滞后
                edge = (rel['cause'], rel['effect'])
                weight = abs(rel['strength'])
                
                if edge not in G.edges():
                    G.add_edge(rel['cause'], rel['effect'], weight=weight, lag=rel['lag'])
                    edge_labels[edge] = f"lag={rel['lag']}\n{rel['strength']:.2f}"
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 子图1: lag=0 的关系
        G0 = nx.DiGraph()
        for var in self.var_names:
            G0.add_node(var)
        for rel in relationships:
            if rel['lag'] == 0:
                G0.add_edge(rel['cause'], rel['effect'], weight=abs(rel['strength']))
        
        pos0 = nx.spring_layout(G0, k=2, iterations=50)
        edges0 = G0.edges()
        weights0 = [G0[u][v]['weight'] * 5 for u, v in edges0]
        
        nx.draw_networkx_nodes(G0, pos0, node_color='lightblue', 
                              node_size=2000, alpha=0.9, ax=ax1)
        nx.draw_networkx_labels(G0, pos0, font_size=10, font_weight='bold', ax=ax1)
        nx.draw_networkx_edges(G0, pos0, width=weights0, alpha=0.6, 
                              edge_color='gray', arrows=True, 
                              arrowsize=20, arrowstyle='->', ax=ax1)
        ax1.set_title('Instantaneous Causal Relationships (lag=0)', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 子图2: lag=1 的关系
        G1 = nx.DiGraph()
        for var in self.var_names:
            G1.add_node(var)
        for rel in relationships:
            if rel['lag'] == 1:
                G1.add_edge(rel['cause'], rel['effect'], weight=abs(rel['strength']))
        
        pos1 = nx.spring_layout(G1, k=2, iterations=50)
        edges1 = G1.edges()
        weights1 = [G1[u][v]['weight'] * 5 for u, v in edges1]
        
        nx.draw_networkx_nodes(G1, pos1, node_color='lightcoral', 
                              node_size=2000, alpha=0.9, ax=ax2)
        nx.draw_networkx_labels(G1, pos1, font_size=10, font_weight='bold', ax=ax2)
        nx.draw_networkx_edges(G1, pos1, width=weights1, alpha=0.6, 
                              edge_color='gray', arrows=True, 
                              arrowsize=20, arrowstyle='->', ax=ax2)
        ax2.set_title('Lagged Causal Relationships (lag=1)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/causal_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存到: {output_dir}/causal_network.png")
    
    def _plot_heatmap(self, output_dir):
        """绘制因果关系热力图（按时间滞后）"""
        n_vars = len(self.var_names)
        n_lags = self.val_matrix.shape[2]
        
        fig, axes = plt.subplots(1, min(n_lags, 4), figsize=(5*min(n_lags, 4), 5))
        if n_lags == 1:
            axes = [axes]
        
        for lag in range(min(n_lags, 4)):
            ax = axes[lag] if n_lags > 1 else axes[0]
            
            # 创建热力图矩阵
            heatmap_data = self.val_matrix[:, :, lag].copy()
            
            # 只显示显著的因果关系
            mask = self.causal_graph[:, :, lag] == ""
            heatmap_data[mask] = 0
            
            im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', 
                          vmin=-1, vmax=1)
            
            ax.set_xticks(np.arange(n_vars))
            ax.set_yticks(np.arange(n_vars))
            ax.set_xticklabels(self.var_names, rotation=45, ha='right')
            ax.set_yticklabels(self.var_names)
            ax.set_xlabel('Effect', fontweight='bold')
            ax.set_ylabel('Cause', fontweight='bold')
            ax.set_title(f'Lag = {lag}', fontweight='bold')
            
            # 添加数值标注
            for i in range(n_vars):
                for j in range(n_vars):
                    if not mask[i, j]:
                        text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=ax, label='Correlation Strength')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/causal_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存到: {output_dir}/causal_heatmap.png")
    
    def _plot_lag_distribution(self, relationships, output_dir):
        """绘制时间滞后分布"""
        if not relationships:
            return
        
        lags = [rel['lag'] for rel in relationships]
        strengths = [abs(rel['strength']) for rel in relationships]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1: 滞后分布直方图
        ax1.hist(lags, bins=range(max(lags)+2), alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Time Lag', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Number of Causal Links', fontweight='bold', fontsize=12)
        ax1.set_title('Distribution of Time Lags', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 滞后 vs 强度散点图
        scatter = ax2.scatter(lags, strengths, alpha=0.6, s=100, c=strengths, 
                            cmap='viridis', edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Time Lag', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Causal Strength (|correlation|)', fontweight='bold', fontsize=12)
        ax2.set_title('Causal Strength vs Time Lag', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Strength')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/lag_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存到: {output_dir}/lag_distribution.png")
    
    def _plot_strength_ranking(self, relationships, output_dir):
        """绘制因果关系强度排名"""
        if not relationships:
            return
        
        # 按强度排序
        sorted_rels = sorted(relationships, key=lambda x: abs(x['strength']), reverse=True)
        top_n = min(20, len(sorted_rels))
        
        labels = [f"{rel['cause']}→{rel['effect']}(lag={rel['lag']})" 
                 for rel in sorted_rels[:top_n]]
        strengths = [rel['strength'] for rel in sorted_rels[:top_n]]
        colors = ['red' if s < 0 else 'green' for s in strengths]
        
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
        y_pos = np.arange(top_n)
        
        ax.barh(y_pos, [abs(s) for s in strengths], color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Causal Strength (|correlation|)', fontweight='bold', fontsize=12)
        ax.set_title(f'Top {top_n} Strongest Causal Relationships', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Positive'),
                          Patch(facecolor='red', alpha=0.7, label='Negative')]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/strength_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存到: {output_dir}/strength_ranking.png")


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("PCMCI 因果发现与可视化")
    print("="*60)
    
    print("\n[1/5] 加载数据...")
    df = pd.read_csv('./dataset/dataset/ETT-small/ETTh1.csv')
    data = df.iloc[:2000, 1:].values  # 使用前2000个样本
    var_names = df.columns[1:].tolist()
    
    print(f"  变量名称: {var_names}")
    print(f"  数据形状: {data.shape}")
    
    print("\n[2/5] 构建因果图（这可能需要几分钟）...")
    causal_builder = CausalGraphBuilder(data, var_names)
    results = causal_builder.build_causal_graph(tau_max=4, pc_alpha=0.05)
    
    print("\n[3/5] 提取因果关系...")
    relationships = causal_builder.get_causal_relationships()
    
    print(f"\n发现 {len(relationships)} 个显著因果关系")
    print("\n前15个最强因果关系:")
    sorted_rels = sorted(relationships, key=lambda x: abs(x['strength']), reverse=True)
    for i, rel in enumerate(sorted_rels[:15]):
        print(f"  {i+1}. {rel['cause']:>6} → {rel['effect']:>6} "
              f"(lag={rel['lag']}, strength={rel['strength']:>6.3f}, p={rel['p_value']:.4f})")
    
    print("\n[4/5] 生成可视化图表...")
    causal_builder.visualize_causal_graph(output_dir='./causal_results')
    
    print("\n[5/5] 保存因果关系数据...")
    # 保存为 CSV
    rel_df = pd.DataFrame(relationships)
    rel_df.to_csv('./causal_results/causal_relationships.csv', index=False)
    print(f"  保存到: ./causal_results/causal_relationships.csv")
    
    # 保存因果图矩阵
    np.save('./causal_results/causal_graph.npy', results['graph'])
    np.save('./causal_results/val_matrix.npy', results['val_matrix'])
    np.save('./causal_results/p_matrix.npy', results['p_matrix'])
    print(f"  保存到: ./causal_results/causal_graph.npy")
    
    print("\n" + "="*60)
    print("✓ 完成！所有结果已保存到 ./causal_results/ 目录")
    print("="*60)
    print("\n生成的可视化文件:")
    print("  1. causal_graph_tigramite.png  - Tigramite 原生因果图")
    print("  2. causal_network.png          - 网络图（lag=0 和 lag=1）")
    print("  3. causal_heatmap.png          - 因果关系热力图")
    print("  4. lag_distribution.png        - 时间滞后分布")
    print("  5. strength_ranking.png        - 因果强度排名")
    print("  6. causal_relationships.csv    - 因果关系数据表")