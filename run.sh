# python demo/top_down_img_demo.py \
#     configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
#     https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
#     --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
#     --out-img-root vis_results

# ./tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w48_coco_512x512.py \
#     checkpoints/hrnet_w48_coco_512x512-cf72fcdf_20200816.pth 2 \
#     --eval mAP

# python tools/test.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/cpm_mpii_368x368.py \
#     checkpoints/cpm_mpii_368x368-116e62b8_20200822.pth \

# python demo/top_down_img_demo_with_mmdet.py \
#     demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
#     checkpoints/dect/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
#     configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/cpm_mpii_368x368.py \
#     checkpoints/cpm_mpii_368x368-116e62b8_20200822.pth \
#     --img-root mypicture/ \
#     --img test-p3.jpg \
#     --out-img-root vis_results \
#     --bbox-thr 0.8

# python demo/top_down_img_demo.py \
#     configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/cpm_mpii_368x368.py \
#     checkpoints/cpm_mpii_368x368-116e62b8_20200822.pth \
#     --img-root mypicture/ \
#     --json-file tests/data/mpii/test_mpii.json \
#     --out-img-root vis_results \

# python demo/single_person_img_demo.py \
#     configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/cpm_mpii_368x368.py \
#     checkpoints/cpm_mpii_368x368-116e62b8_20200822.pth \
#     --img-root mypicture/ \
#     --img test-p1.jpg \
#     --out-img-root vis_results \

# python demo/single_person_img_demo.py \
#     configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/cpm_mpii_368x368.py \
#     checkpoints/cpm_mpii_368x368-116e62b8_20200822.pth \
#     --img-root mycode/mydata/image \
#     --img test-p1.jpg \
#     --out-img-root vis_results/mytest \

python tools/analysis/get_flops.py \
        configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/cpm_mpii_368x368.py \
        --shape 368 368 \
        -n
