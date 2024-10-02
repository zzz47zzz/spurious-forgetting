import os
import pathlib
import random
from dataclasses import dataclass, field
import numpy as np
from transformers import Trainer, TrainingArguments
from enum import Enum, auto
import torch
from typing import List, Dict, TypedDict
import shutil


class AttentionMaskType(Enum):
    ALL_TRUE = auto()
    MASK_EOS = auto()
    MASK_PREVIOUS_BIOGRAPHY = auto()


class FirstTokenAccuracyCalculationStrategy(Enum):
    EPOCH = auto()
    STEP = auto()
    ONLY_END = auto()  # Only calculate at the end of the training. Ignore first_token_accuracy_calculation_interval


first_token_accuracy_calculation_interval_when_only_end = -1


class PersonIndexInterval(TypedDict):
    start: int
    end: int


PreTrainingPersonIndexInfoList = List[PersonIndexInterval]
FineTuningTrainPersonIndexInfoList = List[PersonIndexInterval]
FineTuningTestPersonIndexInfoDict = Dict[str, List[PersonIndexInterval]]
QARawData = Dict[str, Dict[str, Dict[str, str]]]

StepInterval = PersonIndexInterval
StepIntervalList = List[StepInterval]


@dataclass
class DataArguments:
    biography_data_path: str = field(default=None, metadata={"help": "Path to the biography data."})
    all_qa_data_path: str = field(default=None, metadata={"help": "Path to the all QA data."})


@dataclass
class AdditionalTrainingArguments:
    attention_mask_type: AttentionMaskType = field(
        default=None,
        metadata={"help": "Type of attention mask during pre-training."})
    first_token_accuracy_calculation_strategy: FirstTokenAccuracyCalculationStrategy = field(
        default=FirstTokenAccuracyCalculationStrategy.EPOCH,
        metadata={"help": "Strategy for calculating first token accuracy."}
    )
    first_token_accuracy_calculation_interval: int = field(
        default=1,
        metadata={"help": "Interval for calculating first token accuracy."}
    )
    # The following fields are used for pre-training
    pre_training_person_index_info_list: PreTrainingPersonIndexInfoList = field(
        default=None,
        metadata={"help": "List of biography ids for pre-training."}
    )
    # The following fields are used for fine-tuning. The keys of fine_tuning_validation_person_index_info_dict and
    # fine_tuning_test_person_index_info_dict are expected to be the same. The only different is that the index of
    # former dict is expected to be the subset of the latter. This will be checked when the fine-tuning begins.
    fine_tuning_training_person_index_info_list: FineTuningTrainPersonIndexInfoList = field(
        default=None,
        metadata={"help": "List of biography ids for fine-tuning training."}
    )
    fine_tuning_validation_person_index_info_dict: FineTuningTestPersonIndexInfoDict = field(
        default=None,
        metadata={"help": "Dict of biography ids for fine-tuning validation."
                          "The validation frequency is determined by first_token_accuracy_calculation_strategy and "
                          "first_token_accuracy_calculation_interval. It will NOT be used for early stopping."}
    )
    fine_tuning_test_person_index_info_dict: FineTuningTestPersonIndexInfoDict = field(
        default=None,
        metadata={"help": "Dict of biography ids for fine-tuning testing. "
                          "The testing will be carried out ONLY at the end of the fine-tuning process."}
    )


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train_and_save_model(trainer: Trainer, training_args: TrainingArguments, remove_all_checkpoint: bool):
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if not os.path.exists(os.path.join(training_args.output_dir, 'final_model')):
        os.makedirs(os.path.join(training_args.output_dir, 'final_model'))
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=os.path.join(training_args.output_dir, 'final_model'))
    if remove_all_checkpoint:
        for checkpoint_dir in pathlib.Path(training_args.output_dir).glob("checkpoint-*"):
            # remove all checkpoints when training is done
            shutil.rmtree(checkpoint_dir)


def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def construct_selected_person_index_set(
        person_index_info_list: PreTrainingPersonIndexInfoList | FineTuningTrainPersonIndexInfoList) -> set[int]:
    selected_person_index_set = set()
    for info_dict in person_index_info_list:
        for person_index in range(info_dict['start'], info_dict['end']):
            assert person_index not in selected_person_index_set, "The list contains overlap indexes."
            selected_person_index_set.add(person_index)
    return selected_person_index_set


construct_selected_step_set = construct_selected_person_index_set

attribute_list = ['birthday', 'birth_city', 'university', 'major', 'company_name', 'company_city']
