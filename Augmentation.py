import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from tqdm.notebook import tqdm

def compress(imk, scale_percent = 30):
    img = imk.copy()
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def add_noise(img):
    images = []
    l = 20
    for k in range(l):
        img1 = img.copy() 
        cv2.randn(img1,(0,0,0),(50,50,50))
        images.append(img+img1)
    img_avg=np.zeros((img.shape[0],img.shape[1],img.shape[2]),np.float32)
    for im in images:
        img_avg=img_avg+im/l
    img_avg=np.array(np.round(img_avg),dtype=np.uint8)
    
    return img_avg

def increase_brightness(img, value=85):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def increase_contrast(img, factor = 1.5):
    c_img = np.array(img*factor, dtype = int)
    c_img[c_img > 255] = 255
    return c_img

def posterize(imk):
    img = imk.copy()
    img[img > 128] = 255
    img[img <= 128] = 0
    return img

def random_rotate(img):
    import random
    rows, cols, chn = img.shape

    angle = random.randrange(15, 80)
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    dst = cv2.warpAffine(img,M,(int(cols/1.5),int(rows/1.5)))
    
    return dst

def process(o):
    ids = ['o', 'h', 'osp', 'hsp', 'obc', 'hbc', 'ospbc', 'hspbc', 
           'obc2', 'hbc2', 'oc', 'hc', 'op', 'hp', 'ospp', 'hspp', 'or', 'ospr']
    imgs = []

    imgs.append(o)

    h = cv2.flip(o, 1)
    imgs.append(h)

    osp = add_noise(o)
    imgs.append(osp)

    hsp = add_noise(h)
    imgs.append(hsp)

    obc = increase_brightness(o)
    imgs.append(obc)

    hbc = increase_brightness(h)
    imgs.append(hbc)

    ospbc = increase_brightness(osp)
    imgs.append(ospbc)

    hspbc = increase_brightness(hsp)
    imgs.append(hspbc)

    obc2 = increase_brightness(o, 185)
    imgs.append(obc2)

    hbc2 = increase_brightness(h, 185)
    imgs.append(hbc2)

    oc = increase_contrast(o, 1.75)
    imgs.append(oc)

    hc = increase_contrast(h, 1.75)
    imgs.append(hc)

    op = posterize(o)
    imgs.append(op)

    hp = posterize(h)
    imgs.append(hp)

    ospp = posterize(osp)
    imgs.append(ospp)

    hspp = posterize(hsp)
    imgs.append(hspp)

    orr = random_rotate(o)
    imgs.append(orr)

    ospr = random_rotate(osp)
    imgs.append(ospr)
    
    return imgs, ids

def get_str(num):
    n = str(num)
    s = n.zfill(10)
    return s + '_'

count = 0;

folders = os.listdir('./first/')
for folder in tqdm(folders):
    print(folder)
    path = './first/' + folder + '/'
    path2 = './second/' + folder + '/'
    for file in tqdm(os.listdir(path)):
        img = cv2.imread(path + file)
        try:
            imgs, ids = process(compress(img))

            for i in range(len(imgs)):
                count += 1
                name = path2 + get_str(count) + ids[i] + '.png'
                cv2.imwrite(name, imgs[i])
        except e:
            print(e)