"""自动时序日志生成：为时序数据自动生成描述性日志（无需手动标注）"""

import numpy as np
import pandas as pd
from typing import List, Optional
import os


class TimeSeriesLogGenerator:
    """自动为时序数据生成描述性日志"""
    
    def __init__(self, var_names: List[str], var_descriptions: Optional[dict] = None):
        self.var_names = var_names
        self.var_desc = var_descriptions or {
            'HUFL': 'High UseFul Load', 'HULL': 'High UseLess Load',
            'MUFL': 'Middle UseFul Load', 'MULL': 'Middle UseLess Load',
            'LUFL': 'Low UseFul Load', 'LULL': 'Low UseLess Load',
            'OT': 'Oil Temperature',
        }
    
    def generate_window_log(self, data: np.ndarray, timestamp: str = None) -> str:
        """
        为时间窗口生成完整日志
        Args:
            data: shape (seq_len, n_vars)
            timestamp: 窗口起始时间戳
        """
        header = f"<window_log length='{len(data)}'"
        if timestamp:
            header += f" start='{timestamp}'"
        header += ">"
        
        lines = [header]
        lines.append(self._generate_stats(data))
        lines.append(self._generate_trends(data))
        
        anomaly_log = self._generate_anomalies(data)
        if anomaly_log:
            lines.append(anomaly_log)
        
        corr_log = self._generate_correlations(data)
        if corr_log:
            lines.append(corr_log)
        
        lines.append("</window_log>")
        return "\n".join(lines)
    
    def _generate_stats(self, data: np.ndarray) -> str:
        """统计信息"""
        lines = ["  <statistics>"]
        for i, var in enumerate(self.var_names):
            s = data[:, i]
            cv = np.std(s) / (abs(np.mean(s)) + 1e-8)
            volatility = "high" if cv > 0.3 else "medium" if cv > 0.1 else "low"
            lines.append(f"    {var}: mean={np.mean(s):.2f}, std={np.std(s):.2f}, "
                        f"range=[{np.min(s):.2f}, {np.max(s):.2f}], volatility={volatility}")
        lines.append("  </statistics>")
        return "\n".join(lines)
    
    def _generate_trends(self, data: np.ndarray) -> str:
        """趋势分析"""
        lines = ["  <trends>"]
        for i, var in enumerate(self.var_names):
            s = data[:, i]
            pct = (s[-1] - s[0]) / (abs(s[0]) + 1e-8) * 100
            
            if abs(pct) < 1:
                trend, desc = "→", "stable"
            elif pct > 0:
                trend = "↑"
                desc = "strongly rising" if pct > 10 else "rising"
            else:
                trend = "↓"
                desc = "strongly falling" if pct < -10 else "falling"
            
            lines.append(f"    {var}: {trend} {desc} ({pct:+.1f}%)")
        lines.append("  </trends>")
        return "\n".join(lines)
    
    def _generate_anomalies(self, data: np.ndarray) -> Optional[str]:
        """异常检测"""
        anomalies = []
        for i, var in enumerate(self.var_names):
            s = data[:, i]
            z = np.abs((s - np.mean(s)) / (np.std(s) + 1e-8))
            n_anomaly = np.sum(z > 2)
            
            if n_anomaly > 0:
                pct = n_anomaly / len(s) * 100
                max_z = np.max(z)
                anomalies.append(f"    {var}: {n_anomaly} anomalies ({pct:.1f}%, max_z={max_z:.1f}σ)")
        
        if anomalies:
            return "  <anomalies>\n" + "\n".join(anomalies) + "\n  </anomalies>"
        return None
    
    def _generate_correlations(self, data: np.ndarray) -> Optional[str]:
        """变量间相关性"""
        corr = np.corrcoef(data.T)
        strong = []
        n = len(self.var_names)
        
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr[i, j]) > 0.7:
                    direction = "+" if corr[i, j] > 0 else "-"
                    strong.append(f"    {self.var_names[i]} ↔ {self.var_names[j]} ({direction}{abs(corr[i,j]):.2f})")
        
        if strong:
            return "  <correlations>\n" + "\n".join(strong) + "\n  </correlations>"
        return None
    
    def generate_compact_log(self, data: np.ndarray) -> str:
        """生成紧凑格式日志（适合嵌入提示）"""
        parts = []
        for i, var in enumerate(self.var_names):
            s = data[:, i]
            trend = "↑" if s[-1] > s[0] * 1.01 else "↓" if s[-1] < s[0] * 0.99 else "→"
            pct = (s[-1] - s[0]) / (abs(s[0]) + 1e-8) * 100
            parts.append(f"{var}:{trend}{pct:+.0f}%,μ={np.mean(s):.1f}")
        return " | ".join(parts)
    
    def generate_batch_logs(self, data: np.ndarray, seq_len: int, 
                           timestamps: List[str] = None) -> List[str]:
        """批量生成日志"""
        logs = []
        for i in range(0, len(data) - seq_len + 1, seq_len):
            window = data[i:i + seq_len]
            ts = timestamps[i] if timestamps else None
            logs.append(self.generate_window_log(window, ts))
        return logs
    
    def save_logs(self, data: np.ndarray, seq_len: int, 
                  output_path: str = './causal_results/window_logs.txt'):
        """保存日志到文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logs = self.generate_batch_logs(data, seq_len)
        
        with open(output_path, 'w') as f:
            f.write("\n\n".join(logs))
        print(f"✓ 保存 {len(logs)} 个窗口日志到 {output_path}")


# 测试
if __name__ == "__main__":
    print("加载数据...")
    df = pd.read_csv('./dataset/dataset/ETT-small/ETTh1.csv')
    data = df.iloc[:2000, 1:].values
    var_names = df.columns[1:].tolist()
    
    gen = TimeSeriesLogGenerator(var_names)
    
    # 单窗口日志
    window = data[100:612]  # 512步
    print("\n=== 完整窗口日志 ===")
    print(gen.generate_window_log(window, "2016-07-05 00:00"))
    
    print("\n=== 紧凑日志 ===")
    print(gen.generate_compact_log(window))
    
    # 保存
    gen.save_logs(data, seq_len=512)