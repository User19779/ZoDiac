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


def GetDivideMethod(
    mask_tensor: torch.Tensor
) -> tuple[int, ...]:
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
        save_path: str, img_name: str):
    min_x, min_y, max_x, max_y = area_pos
    region_width = max_x - min_x
    region_height = max_y - min_y

    if region_width > region_height:
        # Width is greater than height, horizontal alignment
        square_size = region_width
        for i, pos in enumerate([0, 0.25, 0.5, 0.75, 1]):
            paste_pos_y = int(pos * (square_size - region_height))
            img_temp = Image.new('RGB', (square_size, square_size), 'white')
            img_temp.paste(
                img.crop((min_x, min_y, max_x, max_y)), (0, paste_pos_y))
            img_temp.save(osp.join(save_path, f'{img_name}_pos_{i+1}.png'))

            mask_temp = Image.new('RGB', (square_size, square_size), 'black')
            mask_temp.paste(
                mask.crop((min_x, min_y, max_x, max_y)), (0, paste_pos_y))
            mask_temp.save(osp.join(save_path, f'{img_name}_mask_{i+1}.png'))
    else:
        # Height is greater or equal to width, vertical alignment
        square_size = region_height
        for i, pos in enumerate([0, 0.25, 0.5, 0.75, 1]):
            paste_pos_x = int(pos * (square_size - region_width))
            img_temp = Image.new('RGB', (square_size, square_size), 'white')
            img_temp.paste(
                img.crop((min_x, min_y, max_x, max_y)), (paste_pos_x, 0))
            img_temp.save(osp.join(save_path, f'{img_name}_pos_{i+1}.png'))

            mask_temp = Image.new('RGB', (square_size, square_size), 'black')
            mask_temp.paste(
                mask.crop((min_x, min_y, max_x, max_y)), (paste_pos_x, 0))
            mask_temp.save(osp.join(save_path, f'{img_name}_mask_{i+1}.png'))
