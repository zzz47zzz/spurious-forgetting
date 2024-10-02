export PYTHONPATH=./
python ./training/pre_training.py \
--output_dir ./model/gpt-neox/v_0720/single \
--biography_data_path ./data/processed_0720/biography/single.json \
--train_qa_data_path ./data/processed_0703/qa/train.json \
--test_qa_data_path ./data/processed_0703/qa/test.json

export PYTHONPATH=./
python ./training/pre_training.py \
--output_dir ./model/gpt-neox/v_0720/single_fullname \
--biography_data_path ./data/processed_0720/biography/single_fullname.json \
--train_qa_data_path ./data/processed_0703/qa/train.json \
--test_qa_data_path ./data/processed_0703/qa/test.json

export PYTHONPATH=./
python ./training/pre_training.py \
--output_dir ./model/gpt-neox/v_0720/single_permute \
--biography_data_path ./data/processed_0720/biography/single_permute.json \
--train_qa_data_path ./data/processed_0703/qa/train.json \
--test_qa_data_path ./data/processed_0703/qa/test.json

export PYTHONPATH=./
python ./training/pre_training.py \
--output_dir ./model/gpt-neox/v_0720/single_permute_fullname \
--biography_data_path ./data/processed_0720/biography/single_permute_fullname.json \
--train_qa_data_path ./data/processed_0703/qa/train.json \
--test_qa_data_path ./data/processed_0703/qa/test.json


export PYTHONPATH=./
python ./training/pre_training.py \
--output_dir ./model/gpt-neox/v_0720/multi5 \
--biography_data_path ./data/processed_0720/biography/multi5.json \
--train_qa_data_path ./data/processed_0703/qa/train.json \
--test_qa_data_path ./data/processed_0703/qa/test.json

export PYTHONPATH=./
python ./training/pre_training.py \
--output_dir ./model/gpt-neox/v_0720/multi5_fullname \
--biography_data_path ./data/processed_0720/biography/multi5_fullname.json \
--train_qa_data_path ./data/processed_0703/qa/train.json \
--test_qa_data_path ./data/processed_0703/qa/test.json

export PYTHONPATH=./
python ./training/pre_training.py \
--output_dir ./model/gpt-neox/v_0720/multi5_permute \
--biography_data_path ./data/processed_0720/biography/multi5_permute.json \
--train_qa_data_path ./data/processed_0703/qa/train.json \
--test_qa_data_path ./data/processed_0703/qa/test.json

export PYTHONPATH=./
python ./training/pre_training.py \
--output_dir ./model/gpt-neox/v_0720/multi5_permute_fullname \
--biography_data_path ./data/processed_0720/biography/multi5_permute_fullname.json \
--train_qa_data_path ./data/processed_0703/qa/train.json \
--test_qa_data_path ./data/processed_0703/qa/test.json
