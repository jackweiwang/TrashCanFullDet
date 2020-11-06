_base_ = './mask_rcnn_hrnetv2p_w32_1x_coco.py'
# learning policy
lr_config = dict(step=[150, 170])
total_epochs = 200
