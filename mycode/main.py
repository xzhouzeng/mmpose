    
import custom_tools
import cv2
import mmcv
import os
import time
import warnings
from torchstat import stat
from torchsummary import summary

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def test_model_image():

    pose_config="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/cpm_mpii_368x368.py"
    pose_checkpoint="checkpoints/cpm_mpii_368x368-116e62b8_20200822.pth"
    device='cuda:0'
    image_path="mycode/mydata/image/test-p4.jpg"
    out_img_root="vis_results/mytest"

    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)


    #    pose_result [{'keypoints':(17,3),'bbox':(4)}，...]
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_path,
        None,
        None,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=False,
        outputs=None)

    if out_img_root == "":
        out_file = None
    else:
        os.makedirs(out_img_root, exist_ok=True)
        out_file = os.path.join(out_img_root, 'visualize_pose4.jpg')

    pose_result = []   # (N,K,3)
    for res in pose_results:
        pose_result.append(res['keypoints'])

    # show the results
    img=custom_tools.visusalize_pose(
        image_path,
        pose_result,  # n,k,3
        radius=4,
        thickness=1,
        kpt_score_thr=0.3,
        dataset_type="TopDownCocoDataset"
    )

    # 保存可视化后的图片
    if out_file is not None:
        mmcv.image.imwrite(img, out_file)

def test_model_video():

    # det_config=""
    # det_checkpoint=""

    # cpm
    # pose_config="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/cpm_mpii_368x368.py"
    # pose_checkpoint="checkpoints/cpm_mpii_368x368-116e62b8_20200822.pth"

    # hrnet
    pose_config="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_coarsedropout.py"
    pose_checkpoint="checkpoints/hrnet_w32_coco_256x192_coarsedropout-0f16a0ce_20210320.pth"

    device='cuda:0'
    video_path="mycode/mydata/video/Arrow_step.mp4"
    out_video_root="vis_results/video"

    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Faild to load video file {video_path}'

    if out_video_root == "":
        save_out_video = False
    else:
        os.makedirs(out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(out_video_root,
                         f'vis_{os.path.basename(video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break

        # test a single image, with a list of bboxes.
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            img,
            None,
            None,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        pose_result = []   # (N,K,3)
        for res in pose_results:
            pose_result.append(res['keypoints'])

        # show the results
        vis_img=custom_tools.visusalize_pose(
            img,
            pose_result,  # n,k,3
            radius=4,
            thickness=1,
            kpt_score_thr=0.3,
            dataset_type="TopDownCocoDataset")

        if save_out_video:
            videoWriter.write(vis_img)
            
        # if show:
        #     cv2.imshow('Image', vis_img)
        # if show and cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    if save_out_video:
        videoWriter.release()
    # if show:
    #     cv2.destroyAllWindows()

def test_model_video_with_detection():

    det_config="demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
    det_checkpoint="checkpoints/dect/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

    # cpm
    # pose_config="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/cpm_mpii_368x368.py"
    # pose_checkpoint="checkpoints/cpm_mpii_368x368-116e62b8_20200822.pth"

    # hrnet
    pose_config="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_coarsedropout.py"
    pose_checkpoint="checkpoints/hrnet_w32_coco_256x192_coarsedropout-0f16a0ce_20210320.pth"

    device='cuda:0'
    video_path="mycode/mydata/video/Arrow_step.mp4"
    out_video_root="vis_results/video"
    out_video_name="vis_Arrow_step_HRNet_w32_with_detection.mp4"

    dataset_type="TopDownCocoDataset"

    det_model = init_detector(
        det_config, det_checkpoint, device=device.lower())
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Faild to load video file {video_path}'

    if out_video_root == "":
        save_out_video = False
    else:
        os.makedirs(out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(out_video_root,out_video_name), fourcc, fps, size)

    count_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("count_frames:",count_frames)
    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    
    # 统计运行时间
    start = time.time()     #代码运行之前 获取一下时间
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, 1)

        # test a single image, with a list of bboxes.
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=0.5,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        pose_result = []   # (N,K,3)
        for res in pose_results:
            pose_result.append(res['keypoints'])

        # show the results
        vis_img=custom_tools.visusalize_pose(
            img,
            pose_result,  # n,k,3
            radius=4,
            thickness=1,
            kpt_score_thr=0.3,
            dataset_type=dataset_type)

        if save_out_video:
            videoWriter.write(vis_img)
            
        # if show:
        #     cv2.imshow('Image', vis_img)
        # if show and cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    end = time.time()
    whole_time=end-start
    print("whole_time:",whole_time)

    fr=count_frames/whole_time
    print("pred_fr:",fr)

    cap.release()
    if save_out_video:
        videoWriter.release()
    # if show:
    #     cv2.destroyAllWindows()

def test_model_FLOPs_image():

    pose_config="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/cpm_mpii_368x368.py"
    device='cuda:0'

    pose_model = init_pose_model(
        pose_config, None, device=device.lower())
    # print(pose_model)
    summary(pose_model.backbone, (3, 368, 368))
    
if __name__ == '__main__':
    test_model_video_with_detection()

    
    