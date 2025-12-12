import torch
from layers.CausalModule import CausalModule

# 模拟 configs
class Config:
    enc_in = 7
    d_model = 16
    llm_dim = 768
    causal_cache_dir = './causal_results'
    causal_tau_max = 4
    causal_pc_alpha = 0.05
    causal_top_k = 10

configs = Config()
causal_module = CausalModule(configs)

# 初始化
var_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
causal_module.initialize_causal_discovery(var_names)

# 测试 forward
tokens = causal_module.forward(batch_size=4)
print(f'✓ Causal tokens shape: {tokens.shape}')

# 测试 prompt
prompt = causal_module.get_causal_prompt(format_type='compact')
print(f'✓ Causal prompt: {prompt[:100]}...' if len(prompt) > 100 else f'✓ Causal prompt: {prompt}')

print('\\n✓ All tests passed!')