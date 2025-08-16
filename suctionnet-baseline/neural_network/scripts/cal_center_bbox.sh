# 获取当前sh文件所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 获取上一级目录
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
python $PARENT_DIR/cal_center_bbox.py \
--data_root /home/lixinlong/Project/pose_detect_train/Data/GraspNet \
--saveroot /home/lixinlong/Project/pose_detect_train/Data/GraspNet/additional_label \
--camera realsense \
--pool_size 10 \
--start_scene_idx 00 \
--end_scene_idx 99 \
--save_visu