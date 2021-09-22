config_file=$1
set -x
set -e

python3 -m pdb tools/test.py  -n 1 -se 0 -f configs/${config_file}.py -d dataset-2805 -w logs/${config_file}_gpus2/epoch_23.pkl
