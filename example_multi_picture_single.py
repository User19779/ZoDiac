# %%
# 标准库
import argparse
import gc
import json
import logging
import os
import shutil

# 第三方库
import numpy as np
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torchvision.transforms import transforms
import yaml

# 自定义模块
from loss.loss import LossProvider
from loss.pytorch_ssim import ssim
from main.attdiffusion import ReSDPipeline
from main.divide_picture import GetDivideMethod, save_adjust_image
from main.utils import *
from main.wmattacker import *
from main.wmdiffusion import WMDetectStableDiffusionPipeline
from main.wmpatch import GTWatermark, GTWatermarkMulti

# 其他第三方库
import diffusers
import imageio
import rawpy
import os.path as osp

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


logger_2 = logging.getLogger('print_to_file_logger')
logger_2.setLevel(logging.DEBUG)  # 设置最低日志级别为DEBUG
file_handler_2 = logging.FileHandler('app_single.log')
file_handler_2.setLevel(logging.INFO)  # 可以为handler单独设置日志级别
logger_2.addHandler(file_handler_2)

os.environ['TORCH_HOME'] = './torch_cache'

# %% [markdown]
# ## Necessary Setup for All Sections

# %%
logger.info(f'===== Load Config =====')
device = torch.device('cuda:1')
# device = torch.device('cpu')

with open('./example/config/config_single.yaml', 'r') as file:
  cfgs = yaml.safe_load(file)
logger.info(cfgs)


# 下采样
def reduce_image_resolution(image_path, output_path):
  # 注意：由于VAE,缩小之后的大小必须是8的整数倍,否则会报错
  # 打开原始图像
  with Image.open(image_path) as img:
    width, height = img.size
    new_width = (width // 12 + 7) // 8 * 8
    new_height = (height // 12 + 7) // 8 * 8
    resized_img = img.resize(
      (new_width, new_height), Image.Resampling.LANCZOS)
    resized_img.save(output_path)


# %%
# ZLW：修改循环
imagename_list = [f'1030_{i}' for i in range(7, -1, -1)]
for imagename in []:
# for imagename in imagename_list:
  # 使用rawpy加载NEF文件
  with rawpy.imread(f'./example/input_raw/{imagename}.NEF') as raw:
    # 将图像转换为RGB模式
    rgb = raw.postprocess()
  # 保存为PNG文件
  imageio.imsave(f'./example/input_big/{imagename}.png', rgb)

  # 因为原来的图片过大难以计算,这里先压缩一遍然后另存为一份新的
  reduce_image_resolution(
    f'./example/input_big/{imagename}.png',
    f'./example/input/{imagename}.png')
  reduce_image_resolution(
    f'./example/input_mask_big/{imagename}.png',
    f'./example/input_mask/{imagename}.png')

  gt_img_tensor = get_img_tensor(
    f'./example/input/{imagename}.png', device,
    mask=f'./example/input_mask/{imagename}.png',mask_value=(255,0,0))
  if True:
    image_after_mask = transforms.ToPILImage("RGB")(gt_img_tensor)
    image_after_mask.save(f'./example/input_after_mask/{imagename}.png')
  gt_img_tensor = gt_img_tensor.unsqueeze(0)

  gt_mask = Image.open(
    f'./example/input_mask/{imagename}.png').convert("1")

  gt_mask_tensor = pil_to_tensor(gt_mask)

  # 确定图像的边界
  # pos : (min_x,min_y,max_x,max_y)
  pos = GetDivideMethod(gt_mask_tensor)

  with open(f'./example/input_mask/{imagename}_info.json','w',encoding='utf-8') as f:
      json.dump({"pos":pos}, f,ensure_ascii=False)

  # 根据边界,切出三个图像,分别将人像部分放在左侧、25%位置,中间、75%位置,右侧
  # 需要考虑横向图片（卧姿）的情况,这个时候需要将人像部分放在上侧,25%位置,中间,75%位置,下侧
  save_adjust_image(
    image_after_mask, gt_mask, pos, f'./example/input_after_mask', imagename,
    background_color=(255,0,0))

  del rgb, gt_img_tensor, image_after_mask, gt_mask, gt_mask_tensor,


scheduler = DDIMScheduler.from_pretrained(
  cfgs['model_id'], subfolder="scheduler")
pipe = WMDetectStableDiffusionPipeline.from_pretrained(
  cfgs['model_id'], scheduler=scheduler).to(device)
pipe.set_progress_bar_config(disable=True)

assert isinstance(pipe, WMDetectStableDiffusionPipeline)
assert isinstance(scheduler, diffusers.schedulers.scheduling_ddim.DDIMScheduler)


# 定义一个二层神经网络
class TwoLayerNetMulti(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNetMulti, self).__init__()
        # 定义第一层（从输入层到隐藏层）
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义第二层（从隐藏层到输出层）
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_res = x
        # 通过第一层并应用激活函数（如 ReLU）
        x = F.tanh(self.fc1(x)) + x
        # 通过第二层（输出层通常不需要激活函数，视任务而定）
        x = F.tanh(self.fc2(x) + x)
        return (x + x_res)


class TwoLayerNet(nn.Module):
    def __init__(self, delta_shape):
        super(TwoLayerNet, self).__init__()
        self.delta_x = nn.Parameter(torch.zeros(delta_shape))  # 使用 nn.Parameter 包装

    def forward(self, x):
        x = x + self.delta_x  # 使用 self.delta_x
        return x


for imagename in imagename_list:
  gt_img_tensors: list[torch.Tensor] = list()
  mask_tensors: list[torch.Tensor] = list()
  for i in range(5):
    gt_img_tensors.append(get_img_tensor(
      f'./example/input_after_mask/{imagename}_pos_{i+1}.png', device,
      mask=f'./example/input_after_mask/{imagename}_mask_{i+1}.png',))
    mask_tensors.append(get_img_tensor(
      f'./example/input_after_mask/{imagename}_mask_{i+1}.png', device,))
  watermark_shape = (
    1, 4, gt_img_tensors[0].shape[-2]//8, gt_img_tensors[0].shape[-1]//8)
  rich.print(f"Watermark Shape:[{watermark_shape}]")
  # 为后续的操作做准备
  for i in range(5):
    gt_img_tensors[i] = torch.unsqueeze(gt_img_tensors[i], 0)
    mask_tensors[i] = torch.unsqueeze(mask_tensors[i], 0)

  wm_path = cfgs['save_img']
  logger.info(f'===== Init Pipeline =====')
  if cfgs['w_type'] == 'single':
    wm_pipe = GTWatermark(
      device, shape=watermark_shape,
      w_channel=cfgs['w_channel'], w_radius=cfgs['w_radius'],
      generator=torch.Generator(device).manual_seed(cfgs['w_seed']),
    )
  elif cfgs['w_type'] == 'multi':
    wm_pipe = GTWatermarkMulti(
      device, shape=watermark_shape,
      w_settings=cfgs['w_settings'],
      generator=torch.Generator(device).manual_seed(cfgs['w_seed']),
    )
  else:
    raise NotImplementedError
  # %% [markdown]
  # ## Image Watermarking

  # %%
  # Step 1: Get init noise

  # FIXME 对所有的 子图片 进行一遍,得到不同的 reversed latents
  def get_init_latent(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
    # DDIM inversion from the given image
    img_latents = pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = pipe.forward_diffusion(
      latents=img_latents, text_embeddings=text_embeddings,
      guidance_scale=guidance_scale, num_inference_steps=50,
    )
    return reversed_latents

  empty_text_embeddings = pipe.get_text_embedding('')

  # %%
  # Step 2: prepare training

  # FIXME 对所有的 子图片 进行一遍,得到不同的 reversed latents
  init_latents = list()

  single_picture_only = True
  if single_picture_only:
    init_latents_approx = get_init_latent(
      gt_img_tensors[i], pipe, empty_text_embeddings)
    init_latents = [
      torch.zeros_like(init_latents_approx,device=device,requires_grad=True) 
      for j in range(5)]
    init_latents[2]=(init_latents_approx.detach().clone())
    # init_latents[2].requires_grad = False
  else:
    for i in range(5):
      init_latents_approx = get_init_latent(
        gt_img_tensors[i], pipe, empty_text_embeddings)
      init_latents.append(init_latents_approx.detach().clone())
      # init_latents[i].requires_grad = False

  # 定义 delta latents
  dim = init_latents[2].shape[-1]
  delta_latent_net = TwoLayerNet(init_latents[2].shape).to(device)

  optimizer = optim.Adam(delta_latent_net.parameters(), lr=0.01)
  scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 80], gamma=0.3)

  totalLoss = LossProvider(cfgs['loss_weights'], device)
  loss_lst = []

  # %%
  # Step 3: train the init latents·
  if True:
    torch.cuda.empty_cache()
    for i in range(cfgs['iters']):
      logger.info(f'iter {i}:')
      # FIXME 对所有的 子图片 分别注入
      # FIXME 可能需要重新写一下 inject_watermark ,适应不同的图片的长度

      init_latents_wms = tuple(wm_pipe.inject_watermark(
        delta_latent_net(init_latents[pos])) for pos in range(5))
      pred_img_tensors = []
      
      loss_across_images = 0.0

      optimizer.zero_grad()
      for pos_num in range(5):
        # FIXME 对所有的 子图片 分别注入
        # FIXME 可能需要重新写一下 inject_watermark ,适应不同的图片的长度
        init_latents_wm = wm_pipe.inject_watermark(
          delta_latent_net(init_latents[pos_num]))
        # for abilation study use
        if single_picture_only and pos_num!=2 and ((i+1) not in cfgs['save_iters']):
          continue
        if cfgs['empty_prompt']:
          pred_img_tensor = pipe(
            '', guidance_scale=1.0, num_inference_steps=50, output_type='tensor',
            use_trainable_latents=True, init_latents=init_latents_wm).images
        else:
          pred_img_tensor = pipe(
            prompt, num_inference_steps=50, output_type='tensor',
            use_trainable_latents=True, init_latents=init_latents_wm).images
        pred_img_tensors.append(pred_img_tensor)

        pos_loss = totalLoss(
          pred_img_tensor, gt_img_tensors[pos_num],
          init_latents_wm, wm_pipe=wm_pipe)
        pos_loss.backward()
        
        loss_across_images += pos_loss.item()
      optimizer.step()
      scheduler.step()

      loss_lst.append(loss_across_images)
      # save watermarked image
      if (i+1) in cfgs['save_iters']:
        for pos_num in range(5):
          path = os.path.join(
            wm_path, f"{imagename.split('.')[0]}_{i+1}_pos_{pos_num+1}.png")
          save_img(path, pred_img_tensors[pos_num], pipe)
    gc.collect()
    torch.cuda.empty_cache()

  def binary_search_theta(
    gt_img_tensor:torch.Tensor,wm_img_tensor:torch.Tensor,
    threshold, lower=0., upper=1., precision=1e-6, max_iter=1000,):
    for i in range(max_iter):
      mid_theta = (lower + upper) / 2
      img_tensor = (gt_img_tensor-wm_img_tensor)*mid_theta+wm_img_tensor
      ssim_value = ssim(
        img_tensor.unsqueeze(0), gt_img_tensor.unsqueeze(0),
        data_range=1.0).item()

      if ssim_value <= threshold:
        lower = mid_theta
      else:
        upper = mid_theta
      if upper - lower < precision:
        break
    return lower

  # %%
  # hyperparameter
  ssim_threshold = cfgs['ssim_threshold']
  
  # 运算获得最终的图像
  mask_tensor = get_img_tensor(
    f'./example/input_mask/{imagename}.png', device,return_int=True)
  gt_img_tensor = get_img_tensor(
    f'./example/input/{imagename}.png', device,)
  with open(f'./example/input_mask/{imagename}_info.json','r',encoding='utf-8') as f:
    obj = json.load(f)
    pos = obj["pos"]
    min_x,min_y,max_x,max_y = pos
  original_rectangle = gt_img_tensor[:,min_y:max_y+1,min_x:max_x+1].detach().clone()
  
  # 取 5 个的平均最终水印的加密结果,同时取得它对应的mask
  wm_img_path = os.path.join(
    wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_pos_3.png")
  wm_img_tensor = get_img_tensor(wm_img_path, device)
  
  wm_mask_path = f'./example/input_after_mask/{imagename}_mask_3.png'
  wm_mask_tensor = get_img_tensor(wm_mask_path, device,return_int=True)
  with open(f'./example/input_after_mask/{imagename}_info_3.json','r',encoding='utf-8') as f:
    obj = json.load(f)
    pos = obj["pos"]
    pos_min_x,pos_min_y,pos_max_x,pos_max_y = pos
  valid_rectangle_watermarked = wm_img_tensor[:,pos_min_y:pos_max_y+1,pos_min_x:pos_max_x+1].clone().detach()
  valid_rectangle_mask = wm_mask_tensor[:,pos_min_y:pos_max_y+1,pos_min_x:pos_max_x+1].clone().detach()
  
  valid_rectangle = torch.where(
    valid_rectangle_mask==255,valid_rectangle_watermarked,original_rectangle
  )
  edited_img_tensor = gt_img_tensor.clone().detach()
  edited_img_tensor[:,min_y:max_y+1,min_x:max_x+1] = valid_rectangle
  
  ssim_value = ssim( 
    gt_img_tensor.unsqueeze(0),
    edited_img_tensor.unsqueeze(0),data_range=1.0).item()
  compute_psnr(gt_img_tensor.unsqueeze(0),edited_img_tensor.unsqueeze(0))
  
  logger.info(f'Original SSIM {ssim_value}')
  
  # %%
  
  optimal_theta = binary_search_theta(
    gt_img_tensor,edited_img_tensor,
    ssim_threshold, precision=0.01)
  
  logger.info(f'Optimal Theta {optimal_theta}')
  img_tensor = (gt_img_tensor-edited_img_tensor)*optimal_theta+edited_img_tensor
  
  image_after_mask_name = f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}"

  path = os.path.join(
    wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}_final.png")
  save_img(path, img_tensor.unsqueeze(0), pipe)

  ssim_value = ssim(
    img_tensor.unsqueeze(0), gt_img_tensor.unsqueeze(0),data_range=1.0).item()
  psnr_value = compute_psnr(img_tensor, gt_img_tensor)

  tester_prompt = ''
  text_embeddings = pipe.get_text_embedding(tester_prompt)
  # FIXME
  # 所有的 watermark_prob 之前都需要再进行裁切和遮罩
  image_after_mask = get_img_tensor(
    path,device,f'./example/input_mask/{imagename}.png',255)
  image_after_mask = pipe.numpy_to_pil(
    pipe.img_tensor_to_numpy(image_after_mask.unsqueeze(0)))[0]
  gt_mask = Image.open(
    f'./example/input_mask/{imagename}.png').convert("1")
  
  with open(f'./example/input_mask/{imagename}_info.json','r',encoding='utf-8') as f:
    obj = json.load(f)
    pos = obj["pos"]
    min_x,min_y,max_x,max_y = pos
  save_adjust_image(
    image_after_mask, gt_mask, pos, wm_path, 
    image_after_mask_name)
  
  img_tensors: list[torch.Tensor] = list()
  for i in range(5):
    single_img_tensor = get_img_tensor(
      osp.join(wm_path,f'{image_after_mask_name}_pos_{i+1}.png'), device,
      mask=f'./example/input_after_mask/{imagename}_mask_{i+1}.png',)
    img_tensors.append(single_img_tensor)
    
  det_prob = 1 - watermark_prob(img_tensors[2].unsqueeze(0), pipe,
                  wm_pipe, text_embeddings, device=device)

  logger.info(
    f'SSIM {ssim_value}, PSNR, {psnr_value}, Detect Prob: {det_prob} after Adaptive Enhance')


  # %% [markdown]
  # ## Attack Watermarked Image with Individual Attacks

  # %%
  logger.info(f'===== Init Attackers =====')
  att_pipe = ReSDPipeline.from_pretrained(
    "./stable-diffusion-2-1-base/", torch_dtype=torch.float16)
  att_pipe.set_progress_bar_config(disable=True)
  att_pipe.to(device)

  attackers = {
    'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
    'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
    'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
    'jpeg_attacker_50': JPEGAttacker(quality=50),
    # 'rotate_90': RotateAttacker(degree=90),
    'brightness_0.5': BrightnessAttacker(brightness=0.5),
    'contrast_0.5': ContrastAttacker(contrast=0.5),
    'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
    'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
    'bm3d': BM3DAttacker(),
  }

  # %%
  logger.info(f'===== Start Attacking... =====')

  post_img = os.path.join(
    wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}_final.png")
  for attacker_name, attacker in attackers.items():
    print(f'Attacking with {attacker_name}')
    os.makedirs(os.path.join(wm_path, attacker_name), exist_ok=True)
    att_img_path = os.path.join(
      wm_path, attacker_name, os.path.basename(post_img))
    attackers[attacker_name].attack([post_img], [att_img_path])

  # %% [markdown]
  # ## Attack Watermarked Image with Combined Attacks

  # %%

  # case_list = ['w/ rot', 'w/o rot']
  case_list = ['w/o rot',]

  logger.info(f'===== Init Attackers(\'all\') =====')
  att_pipe = ReSDPipeline.from_pretrained(
    "./stable-diffusion-2-1-base/", torch_dtype=torch.float16)
  att_pipe.set_progress_bar_config(disable=True)
  att_pipe.to(device)

  # %%
  # post_img = os.path.join(
  #   wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}_final.png")

  for case in case_list:
    print(f'Case: {case}')
    if case == 'w/ rot':
      attackers = {
        'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
        'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
        'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
        'jpeg_attacker_50': JPEGAttacker(quality=50),
        'rotate_90': RotateAttacker(degree=90),
        'brightness_0.5': BrightnessAttacker(brightness=0.5),
        'contrast_0.5': ContrastAttacker(contrast=0.5),
        'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
        'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
        'bm3d': BM3DAttacker(),
      }
      multi_name = 'all'
    elif case == 'w/o rot':
      attackers = {
        'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
        'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
        'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
        'jpeg_attacker_50': JPEGAttacker(quality=50),
        'brightness_0.5': BrightnessAttacker(brightness=0.5),
        'contrast_0.5': ContrastAttacker(contrast=0.5),
        'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
        'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
        'bm3d': BM3DAttacker(),
      }
      multi_name = 'all_norot'

    os.makedirs(os.path.join(wm_path, multi_name), exist_ok=True)
    att_img_path = os.path.join(
      wm_path, multi_name, os.path.basename(post_img))
    for i, (attacker_name, attacker) in enumerate(attackers.items()):
      print(f'Attacking with No[{i}]: {attacker_name}')
      if i == 0:
        attackers[attacker_name].attack(
          [post_img], [att_img_path], multi=True)
      else:
        attackers[attacker_name].attack(
          [att_img_path], [att_img_path], multi=True)

  # %% [markdown]
  # ## Detect Watermark

  # %%
  # post_img = os.path.join(
  #   wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}_final.png")

  attackers = ['diff_attacker_60', 'cheng2020-anchor_3', 'bmshj2018-factorized_3', 'jpeg_attacker_50',
    'brightness_0.5', 'contrast_0.5', 'Gaussian_noise', 'Gaussian_blur', 'rotate_90', 'bm3d',
    'all', 'all_norot']

  tester_prompt = ''  # assume at the detection time, the original prompt is unknown
  text_embeddings = pipe.get_text_embedding(tester_prompt)

  # %%
  logger.info(f'===== Testing the Watermarked Images {post_img} =====')
  
  # 记录每一种处理方法的 det_prob
  det_prob_dict = {"Original": det_prob}

  # %%
  logger.info(f'===== Testing the Attacked Watermarked Images =====')
  # 回收缓存
  gc.collect()
  torch.cuda.empty_cache()
  for attacker_name in attackers:
    if not os.path.exists(os.path.join(wm_path, attacker_name)):
      logger.info(f'Attacked images under {attacker_name} not exist.')
      continue

    logger.info(f'=== Attacker Name: {attacker_name} ===')
    
    # FIXME
    # 所有的 watermark_prob 之前都需要再进行裁切和遮罩
    base_name = f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}_final"
    
    image_after_mask = get_img_tensor(
      os.path.join(wm_path, attacker_name, base_name+".png"), device, 
      f'./example/input_mask/{imagename}.png',255,)
    image_after_mask = pipe.numpy_to_pil(
      pipe.img_tensor_to_numpy(image_after_mask.unsqueeze(0)))[0]
    
    gt_mask = Image.open(
      f'./example/input_mask/{imagename}.png').convert("1")
    save_adjust_image(
      image_after_mask, gt_mask, pos, os.path.join(wm_path, attacker_name), 
      base_name)
    
    img_tensors: list[torch.Tensor] = list()
    for i in range(5):
      single_img_tensor = get_img_tensor(
        osp.join(wm_path,attacker_name,f'{base_name}_pos_{i+1}.png'), device,
        mask=f'./example/input_after_mask/{imagename}_mask_{i+1}.png',)
      img_tensors.append(single_img_tensor)
    
  
    det_prob = 1 - watermark_prob(
      img_tensors[2].unsqueeze(0),
      pipe, wm_pipe, text_embeddings, device=device)
    logger.info(f'Watermark Presence Prob.: {det_prob}')

    det_prob_dict[attacker_name] = det_prob

  logger_2.info(f"IMAGE:{imagename}")
  if imagename==imagename_list[-1]:
    logger_2.info(",".join(det_prob_dict.keys()))
  logger_2.info(",".join((f"{i:.5f}" for i in det_prob_dict.values())))
  logger_2.info("")  # newline

  # 回收缓存
  gc.collect()
  torch.cuda.empty_cache()
