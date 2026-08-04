#source /home/jovyan/dzhivelikian/test_time_gd/.envrc
#conda activate /home/jovyan/kuratov/envs/py311_pt2.6_cu12.4
python run_from_config.py --config configs/gradmemgpt/kv_retrieval/hopfield_curriculum.yaml > /dev/null 2>&1 &
 python run_from_config.py --config configs/gradmemgpt/babi/hopfield.yaml > /dev/null 2>&1
python run_from_config.py --config configs/gradmemgpt/kv_retrieval/hopfield_curriculum.yaml overrides seed=143 hopfield_proj_dim=128 > /dev/null 2>&1