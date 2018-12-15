#!/usr/bin/env python

from PIL import Image
import os

# image_root = '/nas/dataset/ILSVRC2012/train/'
image_root = '/data/common_datasets/ILSVRC2012/train/'

cmyk_img_subpaths = ["n01739381/n01739381_1309.JPEG"
                    ,"n02077923/n02077923_14822.JPEG"
                    ,"n02447366/n02447366_23489.JPEG"
                    ,"n02492035/n02492035_15739.JPEG"
                    ,"n02747177/n02747177_10752.JPEG"
                    ,"n03018349/n03018349_4028.JPEG"
                    ,"n03062245/n03062245_4620.JPEG"
                    ,"n03347037/n03347037_9675.JPEG"
                    ,"n03467068/n03467068_12171.JPEG"
                    ,"n03529860/n03529860_11437.JPEG"
                    ,"n03544143/n03544143_17228.JPEG"
                    ,"n03633091/n03633091_5218.JPEG"
                    ,"n03710637/n03710637_5125.JPEG"
                    ,"n03961711/n03961711_5286.JPEG"
                    ,"n04033995/n04033995_2932.JPEG"
                    ,"n04258138/n04258138_17003.JPEG"
                    ,"n04264628/n04264628_27969.JPEG"
                    ,"n04336792/n04336792_7448.JPEG"
                    ,"n04371774/n04371774_5854.JPEG"
                    ,"n04596742/n04596742_4225.JPEG"
                    ,"n07583066/n07583066_647.JPEG"
                    ,"n13037406/n13037406_4650.JPEG"]
png_file_subpath = 'n02105855/n02105855_2933.JPEG'

for subpath in cmyk_img_subpaths:
    image_path = os.path.join(image_root, subpath)
    print('Convert %s(CMYK->RGB)' % image_path)
    img_cmyk = Image.open(image_path)
    img_rgb = img_cmyk.convert('RGB')
    img_rgb.save(image_path, 'jpeg')

png_file_path = os.path.join(image_root,png_file_subpath)
print('Convert %s(PNG->JPEG)' % png_file_path)
img_png = Image.open(png_file_path)
background = Image.new(img_png.mode[:-1], img_png.size, '#FFFFFF')
background.paste(img_png, img_png.split()[-1])
img_png = background
img_png.save(png_file_path, 'jpeg')


print('done')