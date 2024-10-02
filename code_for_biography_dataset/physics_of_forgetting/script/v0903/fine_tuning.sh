for data_id in {4..5};
do
  export WANDB_API=5139c64ae54ccc30c6ab755a670a5d35a2666560
  export PYTHONPATH=./
  export CUDA_VISIBLE_DEVICES=0
  python ./training/full_parameter_fine_tuning.py \
    --config_path ./config/v0903/multi5_permute_fullname/fine_tuning.json \
    --run_config_dict_key 0_"${data_id}"
done


