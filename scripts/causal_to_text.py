"""因果知识文本化：将 PCMCI 因果图转换为 LLM 可理解的提示"""

import numpy as np
from typing import List, Dict, Optional
import json
import os


class CausalGraphToText:
    """将因果图转换为 LLM 可理解的文本"""
    
    def __init__(self, relationships: List[Dict], var_names: List[str],
                 var_descriptions: Optional[Dict] = None):
        self.relationships = relationships
        self.var_names = var_names
        self.var_desc = var_descriptions or {
            'HUFL': 'High UseFul Load', 'HULL': 'High UseLess Load',
            'MUFL': 'Middle UseFul Load', 'MULL': 'Middle UseLess Load',
            'LUFL': 'Low UseFul Load', 'LULL': 'Low UseLess Load',
            'OT': 'Oil Temperature',
        }
    
    def generate_natural_description(self, language: str = 'en') -> str:
        """生成自然语言因果描述"""
        if not self.relationships:
            return "No significant causal relationships found."
        
        sorted_rels = sorted(self.relationships, key=lambda x: abs(x['strength']), reverse=True)
        
        if language == 'zh':
            return self._chinese_description(sorted_rels)
        return self._english_description(sorted_rels)
    
    def _english_description(self, sorted_rels: List[Dict]) -> str:
        lines = [f"[Causal Discovery] Found {len(sorted_rels)} relationships:", ""]
        
        by_effect = {}
        for rel in sorted_rels:
            by_effect.setdefault(rel['effect'], []).append(rel)
        
        for effect, causes in by_effect.items():
            desc = self.var_desc.get(effect, effect)
            lines.append(f"• {effect} ({desc}) is influenced by:")
            for rel in causes[:5]:
                direction = "positively" if rel['strength'] > 0 else "negatively"
                strength = "strongly" if abs(rel['strength']) > 0.5 else "moderately"
                lag_desc = f"with {rel['lag']}-step lag" if rel['lag'] > 0 else "instantly"
                lines.append(f"  - {rel['cause']} {strength} {direction} {lag_desc} (r={rel['strength']:.3f})")
        
        return "\n".join(lines)
    
    def _chinese_description(self, sorted_rels: List[Dict]) -> str:
        lines = [f"[因果发现] 共发现 {len(sorted_rels)} 个因果关系:", ""]
        
        by_effect = {}
        for rel in sorted_rels:
            by_effect.setdefault(rel['effect'], []).append(rel)
        
        for effect, causes in by_effect.items():
            lines.append(f"• {effect} 受以下因素影响:")
            for rel in causes[:5]:
                direction = "正向" if rel['strength'] > 0 else "负向"
                lag_desc = f"滞后{rel['lag']}步" if rel['lag'] > 0 else "同时"
                lines.append(f"  - {rel['cause']} {lag_desc}{direction}影响 (r={rel['strength']:.3f})")
        
        return "\n".join(lines)
    
    def generate_xml_prompt(self) -> str:
        """生成 XML 格式结构化提示"""
        sorted_rels = sorted(self.relationships, key=lambda x: abs(x['strength']), reverse=True)
        lines = ["<causal_knowledge>"]
        lines.append("  <variables>")
        for var in self.var_names:
            lines.append(f"    <var name='{var}' desc='{self.var_desc.get(var, var)}'/>")
        lines.append("  </variables>")
        lines.append("  <links>")
        for rel in sorted_rels[:20]:
            direction = "positive" if rel['strength'] > 0 else "negative"
            lines.append(f"    <link cause='{rel['cause']}' effect='{rel['effect']}' "
                        f"lag='{rel['lag']}' strength='{rel['strength']:.3f}' dir='{direction}'/>")
        lines.append("  </links>")
        lines.append("</causal_knowledge>")
        return "\n".join(lines)
    
    def generate_json_prompt(self) -> str:
        """生成 JSON 格式提示"""
        sorted_rels = sorted(self.relationships, key=lambda x: abs(x['strength']), reverse=True)
        data = {
            "causal_knowledge": {
                "variables": {v: self.var_desc.get(v, v) for v in self.var_names},
                "links": [
                    {"cause": r['cause'], "effect": r['effect'], "lag": r['lag'],
                     "strength": round(r['strength'], 3)} for r in sorted_rels[:20]
                ]
            }
        }
        return json.dumps(data, indent=2)
    
    def generate_forecast_prompt(self, target: str, pred_len: int) -> str:
        """生成针对特定预测任务的因果提示"""
        rels = [r for r in self.relationships if r['effect'] == target]
        rels = sorted(rels, key=lambda x: abs(x['strength']), reverse=True)[:10]
        
        if not rels:
            return f"<causal_guidance>No known causal factors for {target}</causal_guidance>"
        
        lines = [f"<causal_guidance target='{target}' horizon='{pred_len}'>"]
        for r in rels:
            action = "increases" if r['strength'] > 0 else "decreases"
            lines.append(f"  When {r['cause']} rises, {target} {action} after {r['lag']} steps (r={r['strength']:.2f})")
        lines.append("</causal_guidance>")
        return "\n".join(lines)
    
    def save_prompts(self, output_dir: str = './causal_results'):
        """保存所有格式的提示"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f'{output_dir}/causal_desc_en.txt', 'w') as f:
            f.write(self.generate_natural_description('en'))
        with open(f'{output_dir}/causal_desc_zh.txt', 'w') as f:
            f.write(self.generate_natural_description('zh'))
        with open(f'{output_dir}/causal_prompt.xml', 'w') as f:
            f.write(self.generate_xml_prompt())
        with open(f'{output_dir}/causal_prompt.json', 'w') as f:
            f.write(self.generate_json_prompt())
        
        print(f"✓ 因果提示已保存到 {output_dir}/")


# 测试
if __name__ == "__main__":
    # 模拟因果关系（实际使用时从 PCMCI 获取）
    mock_rels = [
        {'cause': 'HUFL', 'effect': 'OT', 'lag': 1, 'strength': 0.65, 'p_value': 0.001},
        {'cause': 'HULL', 'effect': 'OT', 'lag': 2, 'strength': 0.45, 'p_value': 0.01},
        {'cause': 'MUFL', 'effect': 'OT', 'lag': 1, 'strength': -0.38, 'p_value': 0.02},
        {'cause': 'HUFL', 'effect': 'MUFL', 'lag': 0, 'strength': 0.72, 'p_value': 0.001},
    ]
    var_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    gen = CausalGraphToText(mock_rels, var_names)
    
    print("=== 英文描述 ===")
    print(gen.generate_natural_description('en'))
    print("\n=== 中文描述 ===")
    print(gen.generate_natural_description('zh'))
    print("\n=== XML 提示 ===")
    print(gen.generate_xml_prompt())
    print("\n=== 预测提示 (OT) ===")
    print(gen.generate_forecast_prompt('OT', 96))
    
    gen.save_prompts()