import os
import sys
import cv2
import densepose
import argparse
import pickle as pkl
import numpy as np

import matplotlib.pyplot as plt

def main(args):
    with open(args.input_path, 'rb') as f:
        data = pkl.load(f)
    num_data = len(data)
    for datum in data:
        wh = datum['shape']
        image = np.zeros((wh[0], wh[1], 3))
        scores = datum['scores'].cpu().numpy()
        boxes = datum['pred_boxes_XYXY'].cpu().numpy()
        preds = datum['pred_densepose']
        for idx, score in enumerate(scores):
            if score > args.score_th:
                entry = boxes[idx,:]
                entry[2] -= entry[0]
                entry[3] -= entry[1]
                x, y, w, h = [int(v) for v in entry]
                I = preds[idx].labels.cpu().numpy()
                UV = preds[idx].uv.cpu().numpy()
                image[y: y + h, x: x + w,0:2] += UV.transpose((1,2,0))*255
                image[y: y + h, x: x + w,2] += I
        filename = datum['file_name'].split('/')[-1].split('.')[0]

        cv2.imwrite(args.output_path+filename+"_IUV.png", image[:,:,[2,0,1]].astype(np.uint8))

    # Temporary code for saving python2 pickle of data
    if False:
        name = args.input_path.split('.')[0]+"_py2.pkl"
        with open(name, 'wb') as f:
            pkl.dump(data, f, protocol=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', default="/ssd2/swheo/db/lg_project/test/DensePose/result.pkl",
        type=str, help="PKL directory")
    parser.add_argument(
        '--output_path', default="/ssd2/swheo/db/lg_project/test/DensePose/",
        type=str, help="Directory where IUV images are writen")
    parser.add_argument(
        '--score_th', default=0.8,
        type=float, help="Threshold for removing boxes with low score")
    args = parser.parse_args()
    main(args)