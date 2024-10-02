export WANDB_API=5139c64ae54ccc30c6ab755a670a5d35a2666560
export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --config_path ./temp/fine_tuning_config.json --run_name 0_0
export WANDB_API=5139c64ae54ccc30c6ab755a670a5d35a2666560
export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --config_path ./temp/fine_tuning_config.json --run_name 1_0
export WANDB_API=5139c64ae54ccc30c6ab755a670a5d35a2666560
export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --config_path ./temp/fine_tuning_config.json --run_name 1_1
