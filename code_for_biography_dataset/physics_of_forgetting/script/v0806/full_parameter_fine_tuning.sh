for task_id in {0..5};
do
  for ((data_id=0; data_id<=task_id; data_id++))
  do
    export WANDB_API=_
    export PYTHONPATH=./
    python ./training/full_parameter_fine_tuning.py --config_path ./config/v0806/multi5_permute_fullname/fine_tuning.json --run_name "${task_id}"_"${data_id}"
  done
done

#python ./training/full_parameter_fine_tuning.py --config_path ./config/v0806/multi5_permute_fullname/fine_tuning.json --run_name 3_0 &
#python ./training/full_parameter_fine_tuning.py --config_path ./config/v0806/multi5_permute_fullname/fine_tuning.json --run_name 3_1 &
#python ./training/full_parameter_fine_tuning.py --config_path ./config/v0806/multi5_permute_fullname/fine_tuning.json --run_name 3_2 &
#python ./training/full_parameter_fine_tuning.py --config_path ./config/v0806/multi5_permute_fullname/fine_tuning.json --run_name 3_3 &
#python ./training/full_parameter_fine_tuning.py --config_path ./config/v0806/multi5_permute_fullname/fine_tuning.json --run_name 4_0