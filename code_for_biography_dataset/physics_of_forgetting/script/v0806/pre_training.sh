for task_id in {1..5};
do
  export WANDB_API=5139c64ae54ccc30c6ab755a670a5d35a2666560
  export PYTHONPATH=./
  python ./training/pre_training.py --config_path ./config/v0806/multi5_permute_fullname/pre_training.json --task_name task_"${task_id}"
done

export WANDB_API=5139c64ae54ccc30c6ab755a670a5d35a2666560
export PYTHONPATH=./