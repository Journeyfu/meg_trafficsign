config_file=$1
set -x
set -e
python3 tools/train.py  -n 2 -b 2 -f configs/${config_file}.py -d dataset-2805 -w weights/faster_rcnn_res50_coco_3x_800size_40dot1_8682ff1a.pkl
python3 tools/test.py  -n 2 -se 0 -f configs/${config_file}.py -d dataset-2805 -w logs/${config_file}_gpus2/epoch_23.pkl
