import json
import os
from typing import Dict

import wandb
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, GPTNeoXForCausalLM, TrainerCallback
from training.utility import DataArguments, set_seed, AdditionalTrainingArguments
from training.utility import train_and_save_model
from training.utility import FineTuningTestPersonIndexInfoDict, StepIntervalList
from data_module import construct_qa_fine_tuning_data_module, construct_qa_first_token_accuracy_data_module
from data_module import construct_qa_exact_match_data_module
from training.callback import FirstTokenAccuracyCallback, QAExactMatchAccuracyCallback
from training.callback import FirstTokenAccuracyCalculationStrategy
from training.callback import SaveSelectedStepCallback
import argparse


def person_index_info_dict_validation(
        validation_person_index_info_dict: FineTuningTestPersonIndexInfoDict,
        test_person_index_info_dict: FineTuningTestPersonIndexInfoDict
):
    for key in validation_person_index_info_dict.keys():
        assert key in test_person_index_info_dict
        validation_person_index_list = validation_person_index_info_dict[key]
        test_person_index_list = test_person_index_info_dict[key]
        for interval in validation_person_index_list:
            assert any([interval['start'] >= test_interval['start'] and interval['end'] <= test_interval['end'] for
                        test_interval in test_person_index_list])


def fine_tuning(args):
    # For full finetuning, we employ the AdamW optimizer with ε = 10−6. We use weight decays of 0.01 and 0.001,
    # and initial learning rates of 0.001, 0.0003, and 0.0001. There is no warmup,
    # and we implement cosine learning rate scheduling (reducing to 10% of the initial learning rate),
    # a batch size of 48, and a total of 50,000 training steps.
    # Given that we are presenting a negative result for full finetuning (as seen in Figure 2),
    # we display the best QA test accuracy among all the lr/wd parameter combinations.

    # region read config
    pre_training_config = json.load(open(args.config_path, 'r'))
    shared_config: Dict = pre_training_config['shared']  # config that is shared by all tasks
    run_config: Dict = pre_training_config['run'][args.run_config_dict_key]  # config that is different for each task
    wandb_config: Dict = pre_training_config['wandb']
    wandb_config['run_name'] = args.run_config_dict_key if args.wandb_run_name is None else args.wandb_run_name
    # for continual pre-training and fine-tuning
    wandb_config['pre_trained_model_identifier'] = f'task_{args.run_config_dict_key.split("_")[0]}'
    wandb_config['data_identifier'] = f'task_{args.run_config_dict_key.split("_")[1]}'
    # for recovery
    wandb_config['fine_tuned_model_identifier'] = f'task_{args.run_config_dict_key.split("_")[0]}'
    wandb_config['recovery_data_identifier'] = f'task_{args.run_config_dict_key.split("_")[1]}'
    del pre_training_config  # avoid misuse
    # endregion

    # region set trainer arguments
    # Ensure that only one value of pre_trained_model_path is specified
    if run_config.get('output_dir') is not None:
        assert args.output_dir is None, 'The value of output_dir is set in both command lines and configs'
    else:
        assert run_config.get('output_dir') is None, 'The value of output_dir is set in both command lines and configs'
        run_config['output_dir'] = args.output_dir

    # Ensure that only one value of pre_trained_model_path is specified
    # move the code upper so that the config can be logged to wandb
    if run_config.get('pre_trained_model_path') is not None:
        assert args.pre_trained_model_path is None, \
            'The value of pre_trained_model_path is set in both command lines and configs'
    else:
        assert run_config.get('pre_trained_model_path') is None, \
            'The value of pre_trained_model_path is set in both command lines and configs'
        run_config['pre_trained_model_path'] = args.pre_trained_model_path

    # validate the config
    if (run_config['remove_all_checkpoint_when_finish'] is True and
            len(run_config['selected_step_interval_list_to_save_checkpoint']) > 0):
        raise ValueError('remove_all_checkpoint_when_finish=True, '
                         'but selected_step_interval_list_to_save_checkpoint is not empty')
    remove_all_checkpoint_when_finish = run_config['remove_all_checkpoint_when_finish']

    # max_steps = 50_000  # paper 🔵
    max_steps = run_config['max_steps']  # default=-1, override num_train_epochs.
    num_train_epochs = run_config['num_train_epochs']  # default=3.0, total number of training epochs to perform.
    weight_decay = run_config['weight_decay']
    learning_rate = run_config['learning_rate']
    save_strategy = run_config['save_strategy'] if run_config.get('save_strategy') is not None else 'steps'
    output_dir = run_config['output_dir']
    # save_steps = max_steps // 10 if max_steps > 0 else 100  # set by myself 🟡
    if run_config.get('save_steps') is None or run_config['save_steps'] == -1:
        save_steps = max_steps // 10 if max_steps > 0 else 100
    else:
        save_steps = run_config['save_steps']

    training_args = TrainingArguments(
        output_dir=output_dir,  # set by myself 🟡
        optim='adamw_torch',  # default, paper 🔵
        per_device_train_batch_size=48,  # paper 🔵
        eval_strategy='no',  # default (no eval dataset)
        gradient_accumulation_steps=1,  # default
        max_steps=max_steps,  # set by myself 🟡
        # max_steps=100,  # debug 🔴
        num_train_epochs=num_train_epochs,  # set by myself 🟡
        weight_decay=weight_decay,  # set by myself 🟡
        adam_epsilon=1e-6,  # paper 🔵
        warmup_steps=0,  # paper 🔵
        lr_scheduler_type="cosine_with_min_lr",  # paper 🔵
        lr_scheduler_kwargs={"min_lr": learning_rate * 0.9},  # set by myself 🟡
        learning_rate=learning_rate,  # set by myself 🟡
        save_strategy=save_strategy,  # set by myself 🟡
        save_steps=save_steps,  # set by myself 🟡
        # save_steps=10,  # debug 🔴
        # save_total_limit=1,  # set by myself 🟡
        bf16=True,  # set by myself 🟡
        logging_steps=1,  # set by myself 🟡
        report_to=['wandb'],  # set by myself 🟡
        # deepspeed='./ds_configs/stage2.json',
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # endregion

    # region set wandb
    wandb.login(key=os.environ.get('WANDB_API'), relogin=True)  # NOTE: load WANDB_API from environment
    wandb.init(
        project=wandb_config['project'],
        name=wandb_config['run_name'],
        config={**wandb_config, 'all_config': {'shared': shared_config, 'run': run_config, 'wandb': wandb_config}},
    )
    # endregion

    # region construct tokenizer and model
    pre_trained_model_path = run_config['pre_trained_model_path']
    if os.path.isdir(os.path.join(pre_trained_model_path, 'final_model')):
        pre_trained_model_path = os.path.join(pre_trained_model_path, 'final_model')
    elif pre_trained_model_path.split('/')[-1].startswith('checkpoint-'):
        # For loading and evaluating saved ckpt
        # trigger during recovery
        assert os.path.isdir(pre_trained_model_path)
    else:
        # not expected to be triggered in my setting. disable it
        raise ValueError('Unexpected setting')

    max_seq_length = 32
    tokenizer = AutoTokenizer.from_pretrained(
        pre_trained_model_path,
        model_max_length=max_seq_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if 'gpt-neox' in pre_trained_model_path:
        model = GPTNeoXForCausalLM.from_pretrained(pre_trained_model_path)
    elif 'gpt2' in pre_trained_model_path:
        model = GPT2LMHeadModel.from_pretrained(pre_trained_model_path)
    else:
        raise ValueError(f"Unknown model type: {pre_trained_model_path}")
    # endregion

    # region construct data module and trainer
    data_args = DataArguments(all_qa_data_path=shared_config['all_qa_data_path'])
    additional_training_args = AdditionalTrainingArguments(
        first_token_accuracy_calculation_strategy=FirstTokenAccuracyCalculationStrategy[
            run_config['first_token_accuracy_calculation_strategy']],
        first_token_accuracy_calculation_interval=run_config['first_token_accuracy_calculation_interval'],
        fine_tuning_training_person_index_info_list=run_config['training_person_index_info_list'],
        fine_tuning_validation_person_index_info_dict=run_config['validation_person_index_info_dict'],
        fine_tuning_test_person_index_info_dict=run_config['test_person_index_info_dict'],
    )
    person_index_info_dict_validation(
        additional_training_args.fine_tuning_validation_person_index_info_dict,
        additional_training_args.fine_tuning_test_person_index_info_dict)

    qa_dataset = construct_qa_fine_tuning_data_module(
        tokenizer, data_args, additional_training_args, tokenizer.model_max_length)

    first_token_accuracy_dataset_info_list = construct_qa_first_token_accuracy_data_module(
        tokenizer, data_args, additional_training_args, tokenizer.model_max_length)
    first_token_accuracy_callbacks = [
        FirstTokenAccuracyCallback(
            d['dataset'],
            d['calculation_strategy'],
            d['calculation_interval'],
            log_prefix=d['log_prefix'],
            additional_step_interval_list=run_config['additional_step_interval_list_to_calculate_first_token_accuracy']
        )
        for d in first_token_accuracy_dataset_info_list
    ]
    qa_exact_match_dataset_info_list = construct_qa_exact_match_data_module(data_args, additional_training_args)
    qa_exact_match_callbacks = [
        QAExactMatchAccuracyCallback(d['dataset'], tokenizer, log_prefix=d['log_prefix'])
        for d in qa_exact_match_dataset_info_list
    ]
    callbacks: list[TrainerCallback | type[TrainerCallback]] = first_token_accuracy_callbacks + qa_exact_match_callbacks
    callbacks.append(SaveSelectedStepCallback(run_config['selected_step_interval_list_to_save_checkpoint']))
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=qa_dataset,
        callbacks=callbacks,
    )
    # endregion

    # region train and save model
    train_and_save_model(trainer, training_args, remove_all_checkpoint=remove_all_checkpoint_when_finish)
    # endregion


if __name__ == '__main__':
    my_env = os.environ.copy()
    my_env["PATH"] = '/dev_data/qsj/conda/envs/forgetting/bin/:' + my_env["PATH"]
    os.environ.update(my_env)
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument("--run_config_dict_key", type=str, required=True)

    # Let them can be specified in either the command line or the config
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--pre_trained_model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    fine_tuning(parser.parse_args())
