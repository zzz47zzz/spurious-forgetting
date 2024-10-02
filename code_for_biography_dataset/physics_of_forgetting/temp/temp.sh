export WANDB_API=_
export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --config_path ./temp/fine_tuning_config.json --run_name 0_0
export WANDB_API=_
export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --config_path ./temp/fine_tuning_config.json --run_name 1_0
export WANDB_API=_
export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --config_path ./temp/fine_tuning_config.json --run_name 1_1
