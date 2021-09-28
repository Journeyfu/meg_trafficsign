config_file=$1
set -x
set -e

python3 tools/test_final.py  -n 1 -se 0 -f configs/${config_file}.py -d traffic5 -w logs/${config_file}_gpus1/epoch_23.pkl
