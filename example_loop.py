# %%
from main.attdiffusion import ReSDPipeline
from main.wmattacker import *
import rich
from loss.pytorch_ssim import ssim
from loss.loss import LossProvider
from main.utils import *
from main.wmpatch import GTWatermark, GTWatermarkMulti
from main.wmdiffusion import WMDetectStableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from datasets import load_dataset
from diffusers import DDIMScheduler
import torchvision.transforms as transforms
import torch.optim as optim
import torch
import argparse
import yaml
import os
import logging
import shutil
import numpy as np
from PIL import Image

import rawpy
import imageio
import gc

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


logger_2 = logging.getLogger('print_to_file_logger')
logger_2.setLevel(logging.DEBUG)  # 设置最低日志级别为DEBUG
file_handler_2 = logging.FileHandler('app.log')
file_handler_2.setLevel(logging.INFO)  # 可以为handler单独设置日志级别
logger_2.addHandler(file_handler_2)

os.environ['TORCH_HOME'] = './torch_cache'

# %% [markdown]
# ## Necessary Setup for All Sections

# %%
logger.info(f'===== Load Config =====')
device = torch.device('cuda:1')
# device = torch.device('cpu')

with open('./example/config/config.yaml', 'r') as file:
    cfgs = yaml.safe_load(file)
logger.info(cfgs)


# 下采样
def reduce_image_resolution(image_path, output_path):
    # 打开原始图像
    with Image.open(image_path) as img:
        width, height = img.size
        new_width = width // 10
        new_height = height // 10
        resized_img = img.resize(
            (new_width, new_height), Image.Resampling.LANCZOS)
        resized_img.save(output_path)


def binary_search_theta(threshold, lower=0., upper=1., precision=1e-6, max_iter=1000):
    for i in range(max_iter):
        mid_theta = (lower + upper) / 2
        img_tensor = (gt_img_tensor-wm_img_tensor)*mid_theta+wm_img_tensor
        ssim_value = ssim(img_tensor, gt_img_tensor).item()

        if ssim_value <= threshold:
            lower = mid_theta
        else:
            upper = mid_theta
        if upper - lower < precision:
            break
    return lower


def compare_image_sizes(path1, path2):
    with Image.open(path1) as img1:
        width1, height1 = img1.size
    with Image.open(path2) as img2:
        width2, height2 = img2.size
    return (width2 >= width1 and height2 >= height1)


# %%
# ZLW：修改循环
imagename_list = [f'1030_{i}' for i in range(1,8,1)]
for imagename in imagename_list:
    # 使用rawpy加载NEF文件
    with rawpy.imread(f'./example/input_raw/{imagename}.NEF') as raw:
        # 将图像转换为RGB模式
        rgb = raw.postprocess()
    # 保存为PNG文件
    imageio.imsave(f'./example/input_big/{imagename}.png', rgb)

    reduce_image_resolution(
        f'./example/input_big/{imagename}.png',
        f'./example/input/{imagename}.png')
    reduce_image_resolution(
        f'./example/input_mask_big/{imagename}.png',
        f'./example/input_mask/{imagename}.png')

    gt_img_tensor = get_img_tensor(
        f'./example/input/{imagename}.png', device,
        mask=f'./example/input_mask/{imagename}.png', )

    # imagename = 'pepper'
    # gt_img_tensor = get_img_tensor(
    #     f'./example/input/{imagename}.tiff', device, )

    wm_path = cfgs['save_img']
    if True:
        image_after_mask = transforms.ToPILImage("RGB")(gt_img_tensor)
        image_after_mask.save(f'./example/input_after_mask/{imagename}.png')
    gt_img_tensor = gt_img_tensor.unsqueeze(0)

    watermark_shape = (
        1, 4, gt_img_tensor.shape[2]//8, gt_img_tensor.shape[3]//8)
    rich.print(f"Watermark Shape:[{watermark_shape}]")

    # %%
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

    scheduler = DDIMScheduler.from_pretrained(
        cfgs['model_id'], subfolder="scheduler")
    pipe = WMDetectStableDiffusionPipeline.from_pretrained(
        cfgs['model_id'], scheduler=scheduler).to(device)
    pipe.set_progress_bar_config(disable=True)

    # %% [markdown]
    # ## Image Watermarking

    # %%
    # Step 1: Get init noise

    def get_init_latent(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
        # DDIM inversion from the given image
        img_latents = pipe.get_image_latents(img_tensor, sample=False)
        reversed_latents = pipe.forward_diffusion(
            latents=img_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
        )
        return reversed_latents

    empty_text_embeddings = pipe.get_text_embedding('')
    init_latents_approx = get_init_latent(
        gt_img_tensor, pipe, empty_text_embeddings)

    # %%
    # Step 2: prepare training
    init_latents = init_latents_approx.detach().clone()
    init_latents.requires_grad = True
    optimizer = optim.Adam([init_latents], lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 80], gamma=0.3)

    totalLoss = LossProvider(cfgs['loss_weights'], device)
    loss_lst = []

    # %%
    # Step 3: train the init latents
    if True:
        for i in range(cfgs['iters']):
            logger.info(f'iter {i}:')
            init_latents_wm = wm_pipe.inject_watermark(init_latents)
            if cfgs['empty_prompt']:
                pred_img_tensor = pipe('', guidance_scale=1.0, num_inference_steps=50, output_type='tensor',
                                       use_trainable_latents=True, init_latents=init_latents_wm).images  # type: ignore
            else:
                pred_img_tensor = pipe(prompt, num_inference_steps=50, output_type='tensor',
                                       use_trainable_latents=True, init_latents=init_latents_wm).images  # type: ignore
            loss = totalLoss(pred_img_tensor, gt_img_tensor,
                             init_latents_wm, wm_pipe)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_lst.append(loss.item())
            # save watermarked image
            if (i+1) in cfgs['save_iters']:
                path = os.path.join(
                    wm_path, f"{imagename.split('.')[0]}_{i+1}.png")
                save_img(path, pred_img_tensor, pipe)
        torch.cuda.empty_cache()

    # %% [markdown]
    # ## Postprocessing with Adaptive Enhancement

    # %%
    # hyperparameter
    ssim_threshold = cfgs['ssim_threshold']

    # %%
    wm_img_path = os.path.join(
        wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}.png")
    wm_img_tensor = get_img_tensor(wm_img_path, device).unsqueeze(0)

    ssim_value = ssim(wm_img_tensor, gt_img_tensor).item()
    logger.info(f'Original SSIM {ssim_value}')

    # %%
    optimal_theta = binary_search_theta(ssim_threshold, precision=0.01)
    logger.info(f'Optimal Theta {optimal_theta}')

    img_tensor = (gt_img_tensor-wm_img_tensor)*optimal_theta+wm_img_tensor

    ssim_value = ssim(img_tensor, gt_img_tensor).item()
    psnr_value = compute_psnr(img_tensor, gt_img_tensor)

    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    det_prob = 1 - watermark_prob(img_tensor, pipe,
                                  wm_pipe, text_embeddings, device=device)

    path = os.path.join(
        wm_path, f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{ssim_threshold}_unmasked.png")
    save_img(path, img_tensor, pipe)
    logger.info(
        f'SSIM {ssim_value}, PSNR, {psnr_value}, Detect Prob: {det_prob} after postprocessing')

    # %% 为图像中添加遮照
    masked_img_tensor = get_img_tensor(
        path, device, f'./example/input_mask/{imagename}.png', mask_value=127,
        background_image=None,)

    masked_img_tensor = masked_img_tensor.unsqueeze(0)
    masked_path = os.path.join(
        wm_path, f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{ssim_threshold}.png")
    save_img(masked_path, masked_img_tensor, pipe)
    # 回收缓存
    gc.collect()
    torch.cuda.empty_cache()

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
        wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}.png")
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
    post_img = os.path.join(
        wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}.png")

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
    post_img = os.path.join(
        wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}.png")

    attackers = ['diff_attacker_60', 'cheng2020-anchor_3', 'bmshj2018-factorized_3', 'jpeg_attacker_50',
                 'brightness_0.5', 'contrast_0.5', 'Gaussian_noise', 'Gaussian_blur', 'rotate_90', 'bm3d',
                 'all', 'all_norot']

    tester_prompt = ''  # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # %%
    logger.info(f'===== Testing the Watermarked Images {post_img} =====')
    det_prob = 1 - watermark_prob(post_img, pipe,
                                  wm_pipe, text_embeddings, device=device)
    logger.info(f'Watermark Presence Prob.: {det_prob}')

    # 记录每一种处理方法的 det_prob
    det_prob_dict = {"Original": det_prob}

    # %%
    logger.info(f'===== Testing the Attacked Watermarked Images =====')
    for attacker_name in attackers:
        if not os.path.exists(os.path.join(wm_path, attacker_name)):
            logger.info(f'Attacked images under {attacker_name} not exist.')
            continue

        logger.info(f'=== Attacker Name: {attacker_name} ===')
        det_prob = 1 - watermark_prob(
            os.path.join(wm_path, attacker_name, os.path.basename(post_img)),
            pipe, wm_pipe, text_embeddings, device=device)
        logger.info(f'Watermark Presence Prob.: {det_prob}')

        det_prob_dict[attacker_name] = det_prob

    logger_2.info(f"IMAGE:[{imagename}]")
    logger_2.info(str(det_prob_dict))
    logger_2.info("")  # newline

    # 回收缓存
    gc.collect()
    torch.cuda.empty_cache()

