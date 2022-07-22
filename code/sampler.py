import sys
import os
import openslide
import glob
import numpy as np
import torch
import cv2
import tqdm
import unet_studio

class Sampler():
    def __init__(self, path, seg_net, mask_level=5):
        self.path = path
        self.slide = openslide.open_slide(path)
        self.seg_net = seg_net

        downsamples = self.slide.level_downsamples
        self.target_zoom = 2**mask_level
        if downsamples[-1] < self.target_zoom:
            self.mask_level = len(downsamples) - 1
        else:
            self.mask_level = [i for i in range(len(downsamples)) if int(round(downsamples[i])) >= self.target_zoom][0]
        self.level_factor = int(downsamples[self.mask_level])

        self.tissue_mask = self.get_tissue_mask()
        self.tumor_mask  = self.get_tumor_mask()
        self.tissue_mask = cv2.bitwise_and(self.tissue_mask, self.tumor_mask)

    def get_thumb(self, level):
        width, height = self.slide.dimensions
        x, y = 0, 0
        width = width / self.level_factor
        height = height / self.level_factor
        x, y, width, height = list(map(int, [x, y, width, height]))
        thumb = self.slide.read_region((x, y), level, (width, height))
        thumb = np.array(thumb)
        thumb = thumb[:, :, 0:3][:, :, ::-1]
        return thumb


    def get_tissue_mask(self):
        self.thumb = self.get_thumb(self.mask_level)
        img = self.thumb

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean = np.mean(img_yuv)

        if mean < 20:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        y, u, v = cv2.split(img_yuv)
        _,thresh = cv2.threshold(u, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tissue_mask = cv2.erode(thresh, kernel, iterations=1)
        N = 21
        tissue_mask = cv2.dilate(tissue_mask, kernel, iterations=N)
        tissue_mask = cv2.erode(tissue_mask, kernel, iterations=N)
        return tissue_mask

    def segment_image(self, rgb_img):
        img = np.float32(rgb_img)/255
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img[None,:]).cuda()
        with torch.no_grad():
            tile_mask = self.seg_net(img).squeeze()
        tile_mask = tile_mask.argmax(dim=0).data.cpu().numpy().squeeze()
        tile_mask = np.uint8(tile_mask)
        tile_mask = np.uint8(tile_mask==1)*255
        return tile_mask

    def get_tumor_mask(self, tumor_net_image_level=3, step=512, tile_width=512):
        self.tumor_mask = self.tissue_mask*0

        width_level_0, height_level_0 = self.slide.dimensions
        tumor_level_factor = int(self.slide.level_downsamples[tumor_net_image_level])
        width, height = int(width_level_0 / tumor_level_factor), int(height_level_0/tumor_level_factor)

        xs = range(0, width-step, step)
        ys = range(0, height-step, step)
        for x in xs:
            for y in ys:
                x_level_0, y_level_0 = int(x*tumor_level_factor), int(y*tumor_level_factor)
                x_level_mask, y_level_mask = int(x*tumor_level_factor/self.level_factor), int(y*tumor_level_factor/self.level_factor)
                width_level_mask = int(tile_width*tumor_level_factor/self.level_factor)
                tissue = self.tissue_mask[y_level_mask:y_level_mask+width_level_mask,\
                    x_level_mask:x_level_mask+width_level_mask]

                if np.count_nonzero(tissue) > tissue.shape[0]*tissue.shape[1]/2:
                    img = self.slide.read_region((x_level_0, y_level_0), tumor_net_image_level, (tile_width, tile_width))
                    img = np.array(img)[:, :, :3]
                    tumor_mask = self.segment_image(img)                    
                    resize_factor = int(self.level_factor/tumor_level_factor)
                    tumor_mask = cv2.resize(tumor_mask, (width_level_mask, width_level_mask), 0)
                    self.tumor_mask[y_level_mask:y_level_mask+width_level_mask, \
                        x_level_mask:x_level_mask+width_level_mask] = tumor_mask

        return self.tumor_mask

    def sample(self, image_quality=0, step=256, tile_width=256):
        width_level_0, height_level_0 = self.slide.dimensions
        level_factor = int(self.slide.level_downsamples[image_quality])
        width, height = int(width_level_0 / level_factor), int(height_level_0/level_factor)

        xs = range(0, width-step, step)
        ys = range(0, height-step, step)
        for x in xs:
            for y in ys:
                x_level_0, y_level_0 = int(x*level_factor), int(y*level_factor)
                x_level_mask, y_level_mask = int(x_level_0 // self.level_factor), int(y_level_0 // self.level_factor)

                SIZE = int(tile_width*level_factor/self.level_factor)
                mask = self.tissue_mask[y_level_mask:y_level_mask+SIZE, x_level_mask:x_level_mask+SIZE]
                if np.count_nonzero(mask) > SIZE*SIZE*0.95:
                    img = self.slide.read_region((x_level_0, y_level_0), image_quality, (tile_width, tile_width))
                    img = np.array(img)[:, :, :3]
                    yield x_level_0, y_level_0, img