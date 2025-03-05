from diffusers import StableDiffusionPipeline
import os

# 动态设置清华镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model_id = "stabilityai/stable-diffusion-2-1-base"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)

# 保存模型到本地
pipeline.save_pretrained("./stable-diffusion-2-1-base")
print("模型已成功下载并保存到 ./stable-diffusion-2-1-base")
