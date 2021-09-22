config_file=$1
set -x
set -e
python3 -m pdb tools/train.py  -n 1 -b 2 -f configs/${config_file}.py -d dataset-2805  # -w weights/faster_rcnn_res50_coco_3x_800size_40dot1_8682ff1a.pkl
