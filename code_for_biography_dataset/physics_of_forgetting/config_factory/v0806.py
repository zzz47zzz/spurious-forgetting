import json
import os

PROJECT = 'forgetting'
CONFIG_ID = "v0806"
BIOGRAPHY_TYPE = "multi5_permute_fullname"
CONTINUAL_LEARNING_EXP_ID = f"processed_0720_v0730__{CONFIG_ID}_v0806__{BIOGRAPHY_TYPE}"
# pre-training
MODEL_PATH = "./model/gpt-neox/pythia-160m"  # the checkpoint is used to indicate the model architecture
BIOGRAPHY_DATA_PATH = f"./data/processed_0720_v0730/biography/{BIOGRAPHY_TYPE}.json"

PRE_TRAINING_OUTPUT_ROOT_DIR = "./model/gpt-neox/processed_0720_v0730/config_v0806/multi5_permute_fullname"
# fine-tuning
ALL_QA_DATA_PATH = "./data/processed_0720_v0730/qa/all.json"


def construct_pre_training_config():
    task_0_output_dir = os.path.join(PRE_TRAINING_OUTPUT_ROOT_DIR, "task_0")
    result = {
        'wandb': {
            "project": PROJECT,
            "continual_learning_exp_id": CONTINUAL_LEARNING_EXP_ID,
            'phase': 'pre_training',
            'run_name': 'set_in_code (e.g. task_0, task_1, ...)',
        },
        'shared': {
            'model_path': MODEL_PATH,
            'biography_data_path': BIOGRAPHY_DATA_PATH,
        },
        'task': {
            'task_0': {
                "person_index_info_list": [
                    {
                        "start": 0,
                        "end": 100000
                    }
                ],
                "previous_output_dir": "",  # set to "" if this is the first task
                "output_dir": task_0_output_dir,
                "max_steps": 80000,
                "warmup_steps": 1000
            },
        }
    }
    previous_output_dir = task_0_output_dir
    increment_task_num = 5
    increment_biography_num = int(100000 / increment_task_num)  # 100000 means additional 100k biographies
    increment_max_steps = int(80000 / increment_task_num)
    increment_warmup_steps = int(1000 / increment_task_num)
    for task_id in range(1, increment_task_num + 1):
        task_i_output_dir = os.path.join(PRE_TRAINING_OUTPUT_ROOT_DIR, f"task_{task_id}")
        result['task']['task_' + str(task_id)] = {
            "person_index_info_list": [
                {
                    "start": 100000 + (task_id - 1) * increment_biography_num,
                    "end": 100000 + task_id * increment_biography_num
                }
            ],
            "previous_output_dir": previous_output_dir,
            "output_dir": task_i_output_dir,
            "max_steps": increment_max_steps,
            "warmup_steps": increment_warmup_steps
        }
        previous_output_dir = task_i_output_dir
    if not os.path.exists(os.path.join('./config', CONFIG_ID)):
        os.mkdir(os.path.join('./config', CONFIG_ID))
    if not os.path.exists(os.path.join('./config', CONFIG_ID, BIOGRAPHY_TYPE)):
        os.mkdir(os.path.join('./config', CONFIG_ID, BIOGRAPHY_TYPE))
    json.dump(result, open(os.path.join('./config', CONFIG_ID, BIOGRAPHY_TYPE, 'pre_training.json'), 'w'), indent=4)


def construct_fine_tuning_config():
    current_test_person_index_info_dict = {
        "train": [
            {
                "start": 0,
                "end": 500
            }
        ],
        "task_0": [
            {
                "start": 50000,
                "end": 50500
            }
        ]
    }
    current_validation_person_index_info_dict = current_test_person_index_info_dict.copy()
    result = {
        'wandb': {
            "project": PROJECT,
            "continual_learning_exp_id": CONTINUAL_LEARNING_EXP_ID,
            "phase": "fine_tuning",
            'run_name': 'set_in_code (e.g. 0_0, 1_0, 1_1, ...)',
            'pre_trained_model_identifier': 'set_in_code (e.g. task_0, task_1, ...)',
            "data_identifier": "set_in_code (e.g. task_0, task_1, ...)",
        },
        'shared': {
            'all_qa_data_path': ALL_QA_DATA_PATH,
        },
        'run': {
            "0_0": {
                "training_person_index_info_list": [
                    {
                        "start": 0,
                        "end": 50000
                    }
                ],
                "test_person_index_info_dict": current_test_person_index_info_dict.copy(),
                "validation_person_index_info_dict": current_validation_person_index_info_dict.copy(),
                'pre_trained_model_path': os.path.join(PRE_TRAINING_OUTPUT_ROOT_DIR, "task_0"),
                'output_dir': os.path.join(PRE_TRAINING_OUTPUT_ROOT_DIR, "task_0", "fine_tuning", "0_0"),
                "max_steps": 1500,
                "learning_rate": 5e-6,
                "weight_decay": 0.01,
                "num_train_epochs": -1,
                "first_token_accuracy_calculation_strategy": "STEP",
                "first_token_accuracy_calculation_interval": 150
            },
        }
    }
    for task_id in range(1, 6):
        for data_id in range(0, task_id + 1):
            # set current_validation_person_index_info_dict
            if data_id == 0:
                training_person_index_info_list = [{
                    "start": 0,
                    "end": 50_000
                }]
                # update current_test_person_index_info_dict
                current_test_person_index_info_dict.update({
                    f"task_{task_id}": [{
                        "start": 100_000 + (task_id - 1) * 20_000 + 10_000,
                        # "end": 100_000 + (task_id - 1) * 20_000 + 20_000,
                        "end": 100_000 + (task_id - 1) * 20_000 + 10_500,
                    }],
                    'train': [{
                        "start": 0,
                        "end": 500,
                    }]
                })
                # update current_validation_person_index_info_dict
                current_validation_person_index_info_dict = current_test_person_index_info_dict.copy()
            else:
                training_person_index_info_list = [{
                    "start": 100_000 + (data_id - 1) * 20_000,
                    "end": 100_000 + (data_id - 1) * 20_000 + 10_000
                }]
                # update current_test_person_index_info_dict
                current_test_person_index_info_dict.update({
                    f"task_{task_id}": [{
                        "start": 100_000 + (task_id - 1) * 20_000 + 10_000,
                        # "end": 100_000 + (task_id - 1) * 20_000 + 20_000,
                        "end": 100_000 + (task_id - 1) * 20_000 + 10_500,
                    }],
                    'train': [{
                        "start": 100_000 + (data_id - 1) * 20_000,
                        # "end": 100_000 + (data_id - 1) * 20_000 + 10_000,
                        "end": 100_000 + (data_id - 1) * 20_000 + 500,
                    }]
                })
                # update current_validation_person_index_info_dict
                current_validation_person_index_info_dict = current_test_person_index_info_dict.copy()
            # write to result
            result['run'][f'{task_id}_{data_id}'] = {
                'training_person_index_info_list': training_person_index_info_list,
                'validation_person_index_info_dict': current_validation_person_index_info_dict.copy(),
                'test_person_index_info_dict': current_test_person_index_info_dict.copy(),
                'pre_trained_model_path': os.path.join(PRE_TRAINING_OUTPUT_ROOT_DIR, f"task_{task_id}"),
                'output_dir': os.path.join(PRE_TRAINING_OUTPUT_ROOT_DIR, f"task_{task_id}",
                                           "fine_tuning", f"{task_id}_{data_id}"),
                'max_steps': 1500,
                'learning_rate': 5e-6,
                'weight_decay': 0.01,
                'num_train_epochs': -1,
                'first_token_accuracy_calculation_strategy': "STEP",
                'first_token_accuracy_calculation_interval': 150
            }
    json.dump(result, open(os.path.join('./config', CONFIG_ID, BIOGRAPHY_TYPE, 'fine_tuning.json'), 'w'), indent=4)


if __name__ == '__main__':
    # construct_pre_training_config()
    construct_fine_tuning_config()
