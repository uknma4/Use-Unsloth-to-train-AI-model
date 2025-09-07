from peft import PeftModel
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
# 加载原始模型
model = AutoModelForCausalLM.from_pretrained("/home/eddie/models/Deepseek-R1-8b",low_cpu_mem_usage=True)

# 加载微调后的模型（包含 LoRA 适配器）
model = PeftModel.from_pretrained(model, "/home/eddie/deepseek-test-merged")

# 合并 LoRA 适配器权重
model = model.merge_and_unload()


# 保存合并后的模型：

 
model.save_pretrained("/home/eddie/models/merge_deepseek-8b")
