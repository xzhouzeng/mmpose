
import os
import warnings

from dataset_skeleton import get_skeleton
import cv2
import mmcv
import numpy as np
import torch
from PIL import Image

from mmpose.datasets.dataset_info import DatasetInfo


def visusalize_pose(img_or_path,
                    pose_result,  # n,k,3
                    radius=4,
                    thickness=1,
                    kpt_score_thr=0.3,
                    dataset_type="TopDownCocoDataset"):


    skeleton,pose_kpt_color,pose_link_color=get_skeleton(dataset_type)

    img = mmcv.imread(img_or_path)
    img = img.copy()

    img_h, img_w, _ = img.shape

    for kpts in pose_result:

        kpts = np.array(kpts, copy=False)

        # draw each point on image
        assert len(pose_kpt_color) == len(kpts)
        for kid, kpt in enumerate(kpts):
            x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
            if kpt_score > kpt_score_thr:
                r, g, b = pose_kpt_color[kid]
                cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                            (int(r), int(g), int(b)), -1)

        # draw links
        assert len(pose_link_color) == len(skeleton)
        for sk_id, sk in enumerate(skeleton):
            pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
            pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
            if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                    and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                    and pos2[1] > 0 and pos2[1] < img_h
                    and kpts[sk[0], 2] > kpt_score_thr
                    and kpts[sk[1], 2] > kpt_score_thr):
                r, g, b = pose_link_color[sk_id]
                cv2.line(
                    img,
                    pos1,
                    pos2, (int(r), int(g), int(b)),
                    thickness=thickness)


    return img


    