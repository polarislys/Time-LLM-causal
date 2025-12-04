"""
因果模块 - 整合 PCMCI 因果发现和图转文本生成器到 Time-LLM
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import pandas as pd
import os

# 条件导入 tigramite
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    HAS_TIGRAMITE = True
except ImportError:
    HAS_TIGRAMITE = False
    print("Warning: tigramite not installed. Causal discovery will be disabled.")


class CausalDiscovery:
    """PCMCI 因果发现模块"""
    
    def __init__(self, var_names: List[str], tau_max: int = 5, pc_alpha: float = 0.05):
        self.var_names = var_names
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.causal_graph = None
        self.val_matrix = None
        self.relationships = None
        
    def fit(self, data: np.ndarray) -> List[Dict]:
        """
        运行 PCMCI 因果发现
        Args:
            data: shape (T, N) - T时间步，N变量
        Returns:
            因果关系列表
        """
        if not HAS_TIGRAMITE:
            return []
        
        dataframe = pp.DataFrame(data, var_names=self.var_names)
        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
        
        results = pcmci.run_pcmci(tau_max=self.tau_max, pc_alpha=self.pc_alpha)
        
        self.causal_graph = results['graph']
        self.val_matrix = results['val_matrix']
        self.p_matrix = results['p_matrix']
        
        # 提取因果关系
        self.relationships = self._extract_relationships()
        return self.relationships
    
    def _extract_relationships(self) -> List[Dict]:
        """提取因果关系"""
        relationships = []
        n_vars = len(self.var_names)
        
        for j in range(n_vars):
            for i in range(n_vars):
                for tau in range(self.causal_graph.shape[2]):
                    if self.causal_graph[i, j, tau] != "":
                        relationships.append({
                            'cause': self.var_names[i],
                            'effect': self.var_names[j],
                            'lag': tau,
                            'type': self.causal_graph[i, j, tau],
                            'strength': self.val_matrix[i, j, tau],
                            'p_value': self.p_matrix[i, j, tau]
                        })
        return relationships
    
    def load_from_cache(self, cache_dir: str) -> bool:
        """从缓存加载因果关系"""
        csv_path = os.path.join(cache_dir, 'causal_relationships.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            self.relationships = df.to_dict('records')
            return True
        return False


class CausalTextGenerator:
    """因果图转文本生成器"""
    
    def __init__(self, var_names: List[str], var_descriptions: Optional[Dict] = None):
        self.var_names = var_names
        self.var_desc = var_descriptions or {}
        
    def generate_causal_prompt(self, relationships: List[Dict], 
                               target_var: Optional[str] = None,
                               top_k: int = 10,
                               format_type: str = 'natural') -> str:
        """
        生成因果知识的文本提示
        Args:
            relationships: 因果关系列表
            target_var: 目标变量（如果指定，只生成影响该变量的因果关系）
            top_k: 返回前k个最强因果关系
            format_type: 'natural', 'structured', 'compact'
        """
        if not relationships:
            return ""
        
        # 过滤和排序
        if target_var:
            rels = [r for r in relationships if r['effect'] == target_var]
        else:
            rels = relationships
        
        sorted_rels = sorted(rels, key=lambda x: abs(x['strength']), reverse=True)[:top_k]
        
        if format_type == 'natural':
            return self._format_natural(sorted_rels)
        elif format_type == 'structured':
            return self._format_structured(sorted_rels)
        else:  # compact
            return self._format_compact(sorted_rels)
    
    def _format_natural(self, rels: List[Dict]) -> str:
        """自然语言格式"""
        if not rels:
            return ""
        
        lines = ["Causal relationships discovered: "]
        for r in rels:
            direction = "positively" if r['strength'] > 0 else "negatively"
            lag_desc = f"with {r['lag']}-step delay" if r['lag'] > 0 else "immediately"
            lines.append(f"{r['cause']} {direction} affects {r['effect']} {lag_desc}")
        
        return "; ".join(lines)
    
    def _format_structured(self, rels: List[Dict]) -> str:
        """结构化格式"""
        if not rels:
            return ""
        
        lines = ["<causal_knowledge>"]
        for r in rels:
            direction = "+" if r['strength'] > 0 else "-"
            lines.append(f"  {r['cause']} -{direction}(lag={r['lag']})-> {r['effect']}")
        lines.append("</causal_knowledge>")
        return "\n".join(lines)
    
    def _format_compact(self, rels: List[Dict]) -> str:
        """紧凑格式，适合嵌入prompt"""
        if not rels:
            return ""
        
        items = []
        for r in rels:
            sign = "+" if r['strength'] > 0 else "-"
            items.append(f"{r['cause']}{sign}>{r['effect']}(L{r['lag']})")
        
        return "Causal: " + ", ".join(items)


class CausalEncoder(nn.Module):
    """因果知识编码器 - 将因果关系编码为可学习的embedding"""
    
    def __init__(self, n_vars: int, d_model: int, max_lag: int = 5):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.max_lag = max_lag
        
        # 变量embedding
        self.var_embedding = nn.Embedding(n_vars, d_model)
        
        # 因果强度编码 (正/负/无)
        self.strength_embedding = nn.Embedding(3, d_model)  # 0: none, 1: positive, 2: negative
        
        # 时间滞后编码
        self.lag_embedding = nn.Embedding(max_lag + 1, d_model)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, causal_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            causal_matrix: shape (batch, n_vars, n_vars, max_lag+1)
                        值为因果强度 (-1到1之间)
        Returns:
            causal_embedding: shape (batch, n_vars, d_model)
        """
        batch_size = causal_matrix.shape[0]
        # 使用模型参数所在的设备，而不是输入的设备
        device = self.var_embedding.weight.device
        
        # 确保输入在正确的设备上
        causal_matrix = causal_matrix.to(device)
        
        # 获取每个变量的因果特征
        var_features = []
        
        for var_idx in range(self.n_vars):
            # 获取影响该变量的所有因果关系
            causes = causal_matrix[:, :, var_idx, :]  # (batch, n_vars, max_lag+1)
            
            # 找到最强的因果关系
            max_strength, _ = causes.abs().max(dim=-1)  # (batch, n_vars)
            strongest_cause_idx = max_strength.argmax(dim=-1)  # (batch,)
            
            # 编码
            var_idx_tensor = torch.tensor([var_idx], device=device, dtype=torch.long).expand(batch_size)
            var_emb = self.var_embedding(var_idx_tensor)
            
            # 强度类别 (简化: 取平均强度的符号)
            mean_strength = causes.mean(dim=(1, 2))  # (batch,)
            strength_cat = torch.where(mean_strength > 0.1, 
                                    torch.ones(batch_size, device=device, dtype=torch.long),
                                    torch.where(mean_strength < -0.1,
                                                torch.full((batch_size,), 2, device=device, dtype=torch.long),
                                                torch.zeros(batch_size, device=device, dtype=torch.long)))
            strength_emb = self.strength_embedding(strength_cat)
            
            # 最重要的lag
            max_lag_idx = causes.abs().sum(dim=1).argmax(dim=-1)  # (batch,)
            max_lag_idx = max_lag_idx.clamp(0, self.max_lag)
            lag_emb = self.lag_embedding(max_lag_idx)
            
            # 融合
            combined = torch.cat([var_emb, strength_emb, lag_emb], dim=-1)
            var_feature = self.fusion(combined)
            var_features.append(var_feature)
        
        causal_embedding = torch.stack(var_features, dim=1)  # (batch, n_vars, d_model)
        return causal_embedding


class CausalModule(nn.Module):
    """
    完整的因果模块 - 整合因果发现、文本生成和编码
    用于与 Time-LLM 集成
    """
    
    def __init__(self, configs):
        super().__init__()
        
        self.n_vars = configs.enc_in
        self.d_model = configs.d_model
        self.d_llm = configs.llm_dim
        self.use_causal_discovery = getattr(configs, 'use_causal_discovery', True)
        self.causal_cache_dir = getattr(configs, 'causal_cache_dir', './causal_results')
        self.tau_max = getattr(configs, 'causal_tau_max', 4)
        self.pc_alpha = getattr(configs, 'causal_pc_alpha', 0.05)
        self.causal_top_k = getattr(configs, 'causal_top_k', 10)
        
        # 因果编码器
        self.causal_encoder = CausalEncoder(self.n_vars, self.d_model, self.tau_max)
        
        # 投影到LLM维度
        self.causal_projector = nn.Sequential(
            nn.Linear(self.d_model, self.d_llm),
            nn.GELU(),
            nn.Linear(self.d_llm, self.d_llm)
        )
        
        # 因果发现器和文本生成器（非参数化）
        self.causal_discovery = None
        self.text_generator = None
        self.causal_relationships = None
        self.causal_matrix = None
        
    def initialize_causal_discovery(self, var_names: List[str], 
                                   var_descriptions: Optional[Dict] = None):
        """初始化因果发现组件"""
        self.var_names = var_names
        self.causal_discovery = CausalDiscovery(var_names, self.tau_max, self.pc_alpha)
        self.text_generator = CausalTextGenerator(var_names, var_descriptions)
        
        # 尝试从缓存加载
        if self.causal_discovery.load_from_cache(self.causal_cache_dir):
            self.causal_relationships = self.causal_discovery.relationships
            print(f"Loaded {len(self.causal_relationships)} causal relationships from cache")
    
    def discover_causality(self, data: np.ndarray):
        """运行因果发现（通常在训练前运行一次）"""
        if self.causal_discovery is None:
            raise ValueError("Call initialize_causal_discovery first")
        
        self.causal_relationships = self.causal_discovery.fit(data)
        self._build_causal_matrix()
        print(f"Discovered {len(self.causal_relationships)} causal relationships")
        
    def _build_causal_matrix(self):
        """构建因果矩阵用于编码"""
        if not self.causal_relationships:
            self.causal_matrix = torch.zeros(1, self.n_vars, self.n_vars, self.tau_max + 1)
            return
        
        matrix = np.zeros((self.n_vars, self.n_vars, self.tau_max + 1))
        name_to_idx = {name: i for i, name in enumerate(self.var_names)}
        
        for rel in self.causal_relationships:
            i = name_to_idx.get(rel['cause'], -1)
            j = name_to_idx.get(rel['effect'], -1)
            tau = rel['lag']
            if i >= 0 and j >= 0 and tau <= self.tau_max:
                matrix[i, j, tau] = rel['strength']
        
        self.causal_matrix = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    
    def get_causal_prompt(self, target_var: Optional[str] = None,
                         format_type: str = 'compact') -> str:
        """获取因果知识的文本提示"""
        if self.text_generator is None or not self.causal_relationships:
            return ""
        
        return self.text_generator.generate_causal_prompt(
            self.causal_relationships,
            target_var=target_var,
            top_k=self.causal_top_k,
            format_type=format_type
        )
    
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        生成因果知识的embedding
        Args:
            batch_size: 批次大小
        Returns:
            causal_tokens: shape (batch, n_vars, d_llm) - 因果软token
        """
        if self.causal_matrix is None:
            self._build_causal_matrix()
        
        # 扩展到batch大小
        causal_input = self.causal_matrix.expand(batch_size, -1, -1, -1)
        
        # 编码
        causal_emb = self.causal_encoder(causal_input)  # (batch, n_vars, d_model)
        
        # 投影到LLM维度
        causal_tokens = self.causal_projector(causal_emb)  # (batch, n_vars, d_llm)
        
        return causal_tokens
    
    def get_soft_tokens(self, batch_size: int = 1) -> Tuple[torch.Tensor, str]:
        """
        获取因果软token和文本提示
        Returns:
            soft_tokens: 因果embedding
            text_prompt: 因果文本描述
        """
        soft_tokens = self.forward(batch_size)
        text_prompt = self.get_causal_prompt(format_type='compact')
        return soft_tokens, text_prompt