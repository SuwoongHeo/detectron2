#Temporary code for Tex2Shape compatiblity

import os
import sys
import glob
import cv2
import pickle as pkl
import numpy as np

sys.path.append('/projects/DensePose/')

image_root = '/ssd2/swheo/db/lg_project/synthetic_test/images/' #'/ssd2/swheo/db/lg_project/test/images/'
dp_root = '/ssd2/swheo/db/lg_project/synthetic_test/DensePose/'#'/ssd2/swheo/db/lg_project/test/DensePose/'
dp_name = 'result.pkl'
out_dir = '/ssd2/swheo/db/lg_project/synthetic_test/tex2shape_1024x/'#'/ssd2/swheo/db/lg_project/test/tex2shape_1024x/'
target_size = (1024,1024,3)
y_const = 920 # constraint along with height

def get_image_list(input_spec, exp=None):
    if os.path.isdir(input_spec):
        file_list = [
            os.path.join(input_spec, fname)
            for fname in os.listdir(input_spec)
            if os.path.isfile(os.path.join(input_spec, fname))
        ]
    elif os.path.isfile(input_spec):
        file_list = [input_spec]
    else:
        file_list = glob.glob(input_spec)
    if exp!=None:
        for idx_, file_ in enumerate(file_list):
            if exp not in file_.split('/')[-1] :
                file_list.pop(idx_)
    return file_list

image_list = get_image_list(image_root)
iuv_list = get_image_list(dp_root, 'IUV')
with open(dp_root+dp_name, 'rb') as f:
    data = pkl.load(f)
os.makedirs(out_dir, exist_ok=True)
for datum in data:
    fname = datum['file_name'].split('/')[-1].split('.')[0]
    iuv_file = list(filter(lambda x : fname in x.split('/')[-1].split('.')[0], iuv_list))
    image_file = list(filter(lambda x : fname in x.split('/')[-1].split('.')[0], image_list))

    rgb_image = cv2.imread(datum['file_name'])
    iuv_image = cv2.imread(iuv_file[0])
    box_idx = np.argmax(datum['scores'].cpu().numpy())
    box = datum['pred_boxes_XYXY'].cpu().numpy()[box_idx]
    box[2] -= box[0]
    box[3] -= box[1]

    x, y, w, h = [int(v) for v in box]
    ratio = y_const / h if h > y_const else 1
    rgb_crop = cv2.resize(rgb_image[y:y+h, x:x+w,:], (int(w*ratio),int(h*ratio)))
    iuv_crop = cv2.resize(iuv_image[y:y+h, x:x+w,:], (int(w*ratio),int(h*ratio)))
    top, bottom = np.floor((target_size[0] - rgb_crop.shape[0]) / 2).astype(np.int32), \
                  np.ceil((target_size[0] - rgb_crop.shape[0]) / 2).astype(np.int32)
    left, right = np.floor((target_size[1] - rgb_crop.shape[1]) / 2).astype(np.int32), \
                  np.ceil((target_size[1] - rgb_crop.shape[1]) / 2).astype(np.int32)
    rgb_out = cv2.copyMakeBorder(rgb_crop, top, bottom, left, right, cv2.BORDER_CONSTANT)
    iuv_out = cv2.copyMakeBorder(iuv_crop, top, bottom, left, right, cv2.BORDER_CONSTANT)

    cv2.imwrite(out_dir + fname + "_1024x1024.jpg", rgb_out)
    cv2.imwrite(out_dir + fname + "_IUV_1024x1024.png", iuv_out)
    foo = 1