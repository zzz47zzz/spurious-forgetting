for checkpoint_index in $(seq 6250 6250 62500);
do
  export WANDB_API=5139c64ae54ccc30c6ab755a670a5d35a2666560
  export PYTHONPATH=./
  export CUDA_VISIBLE_DEVICES=1
  pre_trained_model_path=./model/gpt-neox/processed_0720_v0730/config_v0903/multi5_permute_fullname/task1_fine_tuning_62500step/checkpoint-"${checkpoint_index}"
  output_dir="${pre_trained_model_path}"/recovery
  python ./training/full_parameter_fine_tuning.py \
    --config_path ./config/v0903/multi5_permute_fullname/recovery.json \
    --run_config_dict_key 1_0 \
    --wandb_run_name task1_checkpoint"${checkpoint_index}"_recover_to_task0 \
    --pre_trained_model_path "${pre_trained_model_path}" \
    --output_dir "${output_dir}"
done