from PIL import Image
import math
import os
from pytorch_msssim import ssim, ms_ssim
import matplotlib.pyplot as plt

from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms
import torch

from typing import Optional



def show_images_side_by_side(images, titles=None, figsize=(8,4)):
    """
    Display a list of images side by side.
    
    Args:
    images (list of numpy arrays): List of images to display.
    titles (list of str, optional): List of titles for each image. Default is None.
    """
    num_images = len(images)
    
    if titles is not None:
        if len(titles) != num_images:
            raise ValueError("Number of titles must match the number of images.")
    
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i]) 
        ax.axis('off')
        
        if titles is not None:
            ax.set_title(titles[i])
    
    plt.tight_layout()
    plt.show()
    return

def show_latent_and_final_img(latent: torch.Tensor, img: torch.Tensor, pipe):
    with torch.no_grad():
        latents_pil_img = pipe.numpy_to_pil(pipe.decode_latents(latent.detach()))[0]
        pil_img = pipe.numpy_to_pil(pipe.img_tensor_to_numpy(img))[0]
    show_images_side_by_side([latents_pil_img, pil_img], ['Latent','Generated Image'])
    return

def save_img(path:str|None, img: torch.Tensor, pipe):
    pil_img = pipe.numpy_to_pil(pipe.img_tensor_to_numpy(img))[0]
    if path:
        pil_img.save(path)
    return pil_img

def _round_up_to_8(n):
    return (n + 7) // 8 * 8


def _make_square(img:Image.Image, fill_color=(255, 255, 255))->Image.Image:
    width, height = img.size
    # 计算长边
    max_size = _round_up_to_8(max(width, height))
    
    # 创建新的方形图像
    new_img = Image.new("RGB", (max_size, max_size), fill_color)
    paste_position = ((max_size - width) // 2, (max_size - height) // 2)
    new_img.paste(img, paste_position)
    return new_img



def adjust_size(img: Image.Image, output_size:tuple[int]|None=None ,fill_color:str|int=0) -> Image.Image:
    # 确保输入图像是二值图像（黑白)
    width, height = img.size
    if isinstance(output_size,tuple):
        target_width,target_height = output_size
    elif output_size==None:
        target_width = target_height = _round_up_to_8(max(width, height))
    
    # 创建新的方形图像，使用白色背景
    new_img = Image.new(img.mode, (target_width, target_height), fill_color)  # 使用灰度模式"L"    
    paste_position = ((target_width - width) // 2, (target_height - height) // 2)
    new_img.paste(img, paste_position)
    new_img.crop((0,0,target_width-1,target_height-1))

    return new_img


def mask_img_tensor(
    img_tensor, mask_path:str, mask_value:Optional[int]=255,background_image:Optional[str]=None):
    
    img_mask = Image.open(mask_path).convert("1")
    mask_width = img_mask.size[-2]
    mask_height = img_mask.size[-1]
    
    # 如果mask和img_tensor存在尺寸差异，调整mask来保证和img_tensor一致
    if (mask_width != img_tensor.shape[-1] or mask_height != img_tensor.shape[-2]):
        img_mask = adjust_size(img_mask,tuple(img_tensor.shape[-1:-3:-1]),0)
    img_mask_tensor = pil_to_tensor(img_mask).repeat((3,1,1))
    
    background_value = mask_value
    if bool(background_image):
        # 加载图片，随机截取和目标区域等大的区域，转换为图片
        background = Image.open(background_image).convert("RGB")
        background = adjust_size(background,tuple(img_tensor.shape[-1:-3:-1]),"white")
        background_value_tensor = pil_to_tensor(background)
    else:
        background_value_tensor = torch.full_like(
            img_tensor, background_value,)
        
    img_tensor = torch.where(
        img_mask_tensor,img_tensor,background_value_tensor)
    return img_tensor

def get_img_tensor(
    img_path, device, mask:Optional[str]=None, mask_value:Optional[int]=255,
    background_image:Optional[str]=None,
    return_int:bool=False):
    img = Image.open(img_path).convert("RGB")

    img_tensor = pil_to_tensor(img)
    
    if bool(mask):
        img_tensor=mask_img_tensor(
            img_tensor,mask,mask_value,background_image)
    if return_int:
        pass
    else:
        img_tensor = (img_tensor/255)
    return img_tensor.to(device)

def create_output_folder(cfgs):
    parent = os.path.join(cfgs['save_img'], cfgs['dataset'])
    wm_path = os.path.join(parent, cfgs['method'], cfgs['case'])
    
    special_model = ['CompVis']
    for key in special_model:
        if key in cfgs['model_id']:
            wm_path = os.path.join(parent, cfgs['method'], '_'.join([cfgs['case'][:-1], key+'/']))
            break
        
    os.makedirs(wm_path, exist_ok=True)
    ori_path = os.path.join(parent, 'OriImgs/')
    os.makedirs(ori_path, exist_ok=True)
    return wm_path, ori_path

# Metrics for similarity
def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100
    return 20 * math.log10(1.) - 10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_ssim(a, b):
    return ssim(a, b, data_range=1.).item()

def load_img(img_path, device):
    img = Image.open(img_path).convert('RGB')
    x = (transforms.ToTensor()(img)).unsqueeze(0).to(device)
    return x

def eval_psnr_ssim_msssim(ori_img_path, new_img_path, device):
    ori_x, new_x = load_img(ori_img_path, device), load_img(new_img_path, device)
    return compute_psnr(ori_x, new_x), compute_ssim(ori_x, new_x), compute_msssim(ori_x, new_x)

def eval_lpips(ori_img_path, new_img_path, metric, device):
    ori_x, new_x = load_img(ori_img_path, device), load_img(new_img_path, device)
    return metric(ori_x, new_x).item()

# Detect watermark from one image
def watermark_prob(img, dect_pipe, wm_pipe, text_embeddings, tree_ring=True, device=torch.device('cuda')):
    if isinstance(img, str):
        img_tensor = get_img_tensor(img,device=device,)
        img_tensor = img_tensor.unsqueeze(0).to(device)
    elif isinstance(img, torch.Tensor):
        img_tensor = img

    img_latents = dect_pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = dect_pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1.0,
        num_inference_steps=50,
    )
    det_prob = wm_pipe.one_minus_p_value(reversed_latents) if not tree_ring else wm_pipe.tree_ring_p_value(reversed_latents)
    return det_prob
