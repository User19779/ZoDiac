from math import sqrt
import os.path as osp
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
from scipy.spatial import distance
import torch
import torch.nn.functional as F

from main.utils import Image
from main.wmattacker import Image

import json


def GetDivideMethod(
    mask_tensor: torch.Tensor
) -> tuple[int, ...]:
    """_summary_

    Args:
        mask_tensor (torch.Tensor): mask的黑白tensor

    Returns:
        tuple[int, ...]: (min_x, min_y, max_x, max_y),包含边界
    """
    print(mask_tensor.shape)

    check_cover_rate = False
    if check_cover_rate:
        white_pixel_total = mask_tensor.sum().item()
        target_white_pixels = int(white_pixel_total * 0.97)

    white_positions = torch.nonzero(mask_tensor)
    min_y, min_x = white_positions[:, 1].min().item(), \
        white_positions[:, 2].min().item()
    max_y, max_x = white_positions[:, 1].max().item(), \
        white_positions[:, 2].max().item()
    ans = (min_x, min_y, max_x, max_y)
    ans = tuple(int(i) for i in ans)
    return ans


def save_adjust_image(
        img: Image.Image, mask: Image.Image, area_pos: tuple[int, ...],
        save_path: str, img_name: str, background_color="white"):
    min_x, min_y, max_x, max_y = area_pos
    # 加一是因为 area pos 是闭区间
    region_width = (max_x - min_x + 1 + 7) // 8 * 8
    region_height = (max_y - min_y + 1 + 7) // 8 * 8

    for i, pos in enumerate([0, 0.25, 0.5, 0.75, 1]):
        if region_width > region_height:
            # Width is greater than height, horizontal alignment
            square_size = region_width
            paste_pos_y = int(pos * (square_size - region_height))
            img_temp = Image.new('RGB', (square_size, square_size), background_color)
            img_temp.paste(
                img.crop((min_x, min_y, max_x, max_y)), (0, paste_pos_y))
            img_temp.save(osp.join(save_path, f'{img_name}_pos_{i+1}.png'))

            mask_temp = Image.new('RGB', (square_size, square_size), 'black')
            mask_temp.paste(
                mask.crop((min_x, min_y, max_x, max_y)), (0, paste_pos_y))
            # 计算新图片的位置
            # pos : (min_x,min_y,max_x,max_y), 包含边界
            pos = (0,paste_pos_y,max_x-min_x,max_y-min_y+paste_pos_y)
        else:
            # Height is greater or equal to width, vertical alignment
            square_size = region_height
            paste_pos_x = int(pos * (square_size - region_width))
            img_temp = Image.new('RGB', (square_size, square_size), background_color)
            img_temp.paste(
                img.crop((min_x, min_y, max_x, max_y)), (paste_pos_x, 0))
            img_temp.save(osp.join(save_path, f'{img_name}_pos_{i+1}.png'))

            mask_temp = Image.new('RGB', (square_size, square_size), 'black')
            mask_temp.paste(
                mask.crop((min_x, min_y, max_x, max_y)), (paste_pos_x, 0))
                        # 计算新图片的位置
            # pos : (min_x,min_y,max_x,max_y), 包含边界
            pos = (paste_pos_x,0,max_x-min_x+paste_pos_x,max_y-min_y)
            
        mask_temp.save(osp.join(save_path, f'{img_name}_mask_{i+1}.png'))
        with open(osp.join(save_path, f'{img_name}_info_{i+1}.json'),'w',encoding='utf-8') as f:
            json.dump({"pos":pos}, f,ensure_ascii=False)
