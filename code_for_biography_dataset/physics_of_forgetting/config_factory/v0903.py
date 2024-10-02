import json
import os

# PROJECT = 'forgetting'
CONFIG_ID = "v0903"
BIOGRAPHY_TYPE = "multi5_permute_fullname"
CONTINUAL_LEARNING_EXP_ID = f"continual_fine_tuning__processed_0720_v0730__config_v{CONFIG_ID}__{BIOGRAPHY_TYPE}"

# fine-tuning
ALL_QA_DATA_PATH = "./data/processed_0720_v0730/qa/all.json"
SAVE_MODEL_ROOT_DIR = "./model/gpt-neox/processed_0720_v0730/config_v0903/multi5_permute_fullname/"


def construct_fine_tuning_config(output_file_name: str,
                                 first_token_accuracy_calculation_interval: int,
                                 save_steps: int):
    interval_list_of_storing_checkpoint_and_calculating_first_token_accuracy = []
    for i in range(0, 201, 5):
        if i == 0:
            continue
        interval_list_of_storing_checkpoint_and_calculating_first_token_accuracy.append({"start": i, "end": i + 1})
    test_person_index_info_dict = {
        "train": [
            {
                "start": 0,
                "end": 50000
            }
        ],
        "task_0": [
            {
                "start": 50000,
                "end": 100000
            }
        ]
    }
    validation_person_index_info_dict = {
        'train': [
            {
                'start': 0,
                'end': 500
            }
        ],
        'task_0': [
            {
                'start': 50000,
                'end': 50500
            }
        ]
    }
    result = {
        "wandb": {
            "project": 'forgetting',  # PROJECT
            "continual_learning_exp_id": CONTINUAL_LEARNING_EXP_ID,
            "phase": "fine_tuning",
            "run_name": "set_in_code (e.g. 0_0, 1_0, 1_1, ...)",
            "pre_trained_model_identifier": "set_in_code (e.g. task_0, task_1, ...)",
            "data_identifier": "set_in_code (e.g. task_0, task_1, ...)"
        },
        "shared": {
            "all_qa_data_path": ALL_QA_DATA_PATH
        },
        'run': {
            '0_0': {
                'training_person_index_info_list': [
                    {
                        'start': 0,
                        'end': 50_000
                    }
                ],
                "test_person_index_info_dict": test_person_index_info_dict.copy(),
                'validation_person_index_info_dict': validation_person_index_info_dict.copy(),
                "selected_step_interval_list_to_save_checkpoint":
                    interval_list_of_storing_checkpoint_and_calculating_first_token_accuracy,
                'additional_step_interval_list_to_calculate_first_token_accuracy':
                    interval_list_of_storing_checkpoint_and_calculating_first_token_accuracy,
                'pre_trained_model_path': os.path.join(SAVE_MODEL_ROOT_DIR, 'task0_pre_training'),
                "output_dir": os.path.join(SAVE_MODEL_ROOT_DIR, "task0_fine_tuning"),
                "max_steps": 62_500,
                "learning_rate": 5e-06,
                "weight_decay": 0.01,
                "num_train_epochs": -1,
                'save_steps': save_steps,
                "first_token_accuracy_calculation_strategy": "STEP",
                "first_token_accuracy_calculation_interval": first_token_accuracy_calculation_interval,
                'remove_all_checkpoint_when_finish': False,
            },
        }
    }
    for i in range(1, 6):
        test_person_index_info_dict['train'] = [
            {
                'start': 100_000 + 20_000 * (i - 1),
                'end': 100_000 + 20_000 * i
            }
        ]
        test_person_index_info_dict[f'task_{i}'] = [
            {
                'start': 100_000 + 20_000 * (i - 1),
                'end': 100_000 + 20_000 * i
            }
        ]
        validation_person_index_info_dict['train'] = [
            {
                'start': 100_000 + 20_000 * (i - 1),
                'end': 100_000 + 20_000 * (i - 1) + 500
            }
        ]
        validation_person_index_info_dict[f'task_{i}'] = [
            {
                'start': 100_000 + 20_000 * (i - 1),
                'end': 100_000 + 20_000 * (i - 1) + 500
            }
        ]
        result['run'][f'0_{i}'] = {
            'training_person_index_info_list': [
                {
                    'start': 100_000 + 20_000 * (i - 1),
                    'end': 100_000 + 20_000 * i
                }
            ],
            'test_person_index_info_dict': test_person_index_info_dict.copy(),
            'validation_person_index_info_dict': validation_person_index_info_dict.copy(),
            "selected_step_interval_list_to_save_checkpoint":
                interval_list_of_storing_checkpoint_and_calculating_first_token_accuracy,
            'additional_step_interval_list_to_calculate_first_token_accuracy':
                interval_list_of_storing_checkpoint_and_calculating_first_token_accuracy,
            'pre_trained_model_path': os.path.join(SAVE_MODEL_ROOT_DIR, f"task{i - 1}_fine_tuning"),
            "output_dir": os.path.join(SAVE_MODEL_ROOT_DIR, f"task{i}_fine_tuning"),
            'max_steps': 62_500,
            'learning_rate': 5e-06,
            'weight_decay': 0.01,
            'num_train_epochs': -1,
            'save_steps': save_steps,
            'first_token_accuracy_calculation_strategy': "STEP",
            'first_token_accuracy_calculation_interval': first_token_accuracy_calculation_interval,
            'remove_all_checkpoint_when_finish': False,
        }
    config_output_dir = os.path.join('./config', CONFIG_ID, BIOGRAPHY_TYPE)
    if not os.path.exists(config_output_dir):
        os.makedirs(config_output_dir)
    json.dump(result, open(os.path.join(config_output_dir, output_file_name), 'w'), indent=2)


def construct_recovery_fine_tuning_config():
    test_person_index_info_dict = {
        "train": [
            {
                "start": 0,
                "end": 50000
            }
        ],
        "task_0": [
            {
                "start": 50000,
                "end": 100000
            }
        ]
    }
    validation_person_index_info_dict = {
        'train': [
            {
                'start': 0,
                'end': 500
            }
        ],
        'task_0': [
            {
                'start': 50000,
                'end': 50500
            }
        ]
    }
    result = {
        'wandb': {
            "project": 'forgetting_recovery',  # PROJECT
            "continual_learning_exp_id": CONTINUAL_LEARNING_EXP_ID,
            "phase": "recovery",
            "run_name": "set_in_code (e.g. 0_0, 1_0, 1_1, ...)",
            "pre_trained_model_identifier": "set_in_code (e.g. task_0, task_1, ...)",
            "data_identifier": "set_in_code (e.g. task_0, task_1, ...)"
        },
        'shared': {"all_qa_data_path": ALL_QA_DATA_PATH},
        'run': {
            '1_0': {
                'training_person_index_info_list': [
                    {
                        'start': 0,
                        'end': 50_000
                    }
                ],
                "test_person_index_info_dict": test_person_index_info_dict.copy(),
                'validation_person_index_info_dict': validation_person_index_info_dict.copy(),
                "selected_step_interval_list_to_save_checkpoint": [],
                'additional_step_interval_list_to_calculate_first_token_accuracy': [],
                # do not set pre_trained_model_path
                # do not set output_dir
                'max_steps': 6250,
                "learning_rate": 5e-06,
                "weight_decay": 0.01,
                "num_train_epochs": -1,
                'save_steps': -1,
                "first_token_accuracy_calculation_strategy": "STEP",
                "first_token_accuracy_calculation_interval": 1000,
                'remove_all_checkpoint_when_finish': True,
            },
        },
    }
    for i in range(2, 6):
        # only recover the previous task
        test_person_index_info_dict.clear()  # clear it
        validation_person_index_info_dict.clear()  # clear it
        test_person_index_info_dict['train'] = [
            {
                'start': 100_000 + 20_000 * (i - 2),
                'end': 100_000 + 20_000 * (i - 2) + 10_000
            }
        ]
        test_person_index_info_dict[f'task_{i - 1}'] = [
            {
                'start': 100_000 + 20_000 * (i - 2) + 10_000,
                'end': 100_000 + 20_000 * (i - 2) + 20_000,
            }
        ]

        validation_person_index_info_dict['train'] = [
            {
                'start': 100_000 + 20_000 * (i - 2),
                'end': 100_000 + 20_000 * (i - 2) + 500
            }
        ]
        validation_person_index_info_dict[f'task_{i - 1}'] = [
            {
                'start': 100_000 + 20_000 * (i - 2) + 10_000,
                'end': 100_000 + 20_000 * (i - 2) + 10_000 + 500
            }
        ]

        result['run'][f'{i}_{i - 1}'] = {
            'training_person_index_info_list': [
                {
                    'start': 100_000 + 20_000 * (i - 2),
                    'end': 100_000 + 20_000 * (i - 2) + 10_000
                }
            ],
            "test_person_index_info_dict": test_person_index_info_dict.copy(),
            'validation_person_index_info_dict': validation_person_index_info_dict.copy(),
            "selected_step_interval_list_to_save_checkpoint": [],
            'additional_step_interval_list_to_calculate_first_token_accuracy': [],
            # do not set pre_trained_model_path
            # do not set output_dir
            'max_steps': 6250,
            "learning_rate": 5e-06,
            "weight_decay": 0.01,
            "num_train_epochs": -1,
            'save_steps': -1,
            "first_token_accuracy_calculation_strategy": "STEP",
            "first_token_accuracy_calculation_interval": 1000,
            'remove_all_checkpoint_when_finish': True,
        }
    json.dump(result, open(os.path.join('./config', CONFIG_ID, BIOGRAPHY_TYPE, 'recovery.json'), 'w'), indent=2)


if __name__ == '__main__':
    construct_fine_tuning_config('fine_tuning.json', 1000, -1)
    construct_recovery_fine_tuning_config()
