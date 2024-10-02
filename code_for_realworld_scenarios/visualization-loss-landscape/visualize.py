import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import pickle
import numpy as np
import wandb
import pandas as pd 
from pprint import pprint
from copy import deepcopy
from dataclasses import dataclass, field
import matplotlib
import matplotlib.pyplot as plt
import re
import torch
from numpy.linalg import svd


'''
Bash For Run this code

CUDA_VISIBLE_DEVICES=0 nohup python ./visualization-landscape/visualize.py --num_split 4 --split_id 0 >nohup_00.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python ./visualization-landscape/visualize.py --num_split 4 --split_id 1 >nohup_01.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python ./visualization-landscape/visualize.py --num_split 4 --split_id 2 >nohup_10.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python ./visualization-landscape/visualize.py --num_split 4 --split_id 3 >nohup_11.out 2>&1 &

'''


def compute_loss(eval_model, data_loader) -> float:

    total_loss = 0
    sample_cnt = 0
    eval_model.eval()
    with torch.no_grad():
        for lm_input in data_loader:
            batch_loss = eval_model(**{'input_ids':lm_input['input_ids_with_ans'], 
                                        'attention_mask':lm_input['attention_mask_with_ans'],
                                        'labels':lm_input['labels_with_ans']}).loss.item()
            total_loss += batch_loss*(lm_input['input_ids_with_ans'].shape[0])
            sample_cnt += lm_input['input_ids_with_ans'].shape[0]

    total_loss = total_loss/sample_cnt

    return total_loss

import pickle
import sys
sys.path.append('/dev_data/zjh/LLM_CL/')
from pathlib import Path
cwd = Path('/dev_data/zjh/LLM_CL/')
import os
os.chdir(cwd)
from main_CL import *
from utils.evaluation import evaluate_sent_level_acc_with_generation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_split",type=int, default=4, required=False, help="Number of total splits")
parser.add_argument("--split_id",type=int, default=2, required=False, help="Split ID, e.g., 0,1,2,3")
params = parser.parse_args()
num_split = params.num_split
split_id = params.split_id


# Exp: SEQ (2024.08.29, Start from model finetuned on Task0, then directly finetune on Task1)
# Direction 1
# ft_ckpt_xbase_id, ft_ckpt_xaxis_id = 'checkpoint-100', 'checkpoint-150' 
# ckpt_xbase = f'/dev_data/zjh/physics-of-forgetting-in-llm/physics_of_forgetting/model/gpt-neox/processed_0720_v0730/config_v0806/multi5_permute_fullname/task_0/fine_tuning/1_1_first_200steps_eval_every_step/{ft_ckpt_xbase_id}'  
# ckpt_xaxis = f'/dev_data/zjh/physics-of-forgetting-in-llm/physics_of_forgetting/model/gpt-neox/processed_0720_v0730/config_v0806/multi5_permute_fullname/task_0/fine_tuning/1_1_first_200steps_eval_every_step/{ft_ckpt_xaxis_id}'  

# # Direction 2
# ft_ckpt_ybase_id, ft_ckpt_yaxis_id = 'checkpoint-150', 'checkpoint-62500'
# ckpt_ybase = f'/dev_data/zjh/physics-of-forgetting-in-llm/physics_of_forgetting/model/gpt-neox/processed_0720_v0730/config_v0806/multi5_permute_fullname/task_0/fine_tuning/1_1_first_200steps_eval_every_step/{ft_ckpt_ybase_id}'  
# ckpt_yaxis = f'/dev_data/zjh/physics-of-forgetting-in-llm/physics_of_forgetting/model/gpt-neox/processed_0720_v0730/config_v0806/multi5_permute_fullname/task_0/fine_tuning/1_1/{ft_ckpt_yaxis_id}'  


# Exp: REPLAY020 (2024.09.09, Start from model finetuned on Task0, then directly finetune on Task1 + 20% Task 0 (data replay))
# # Direction 1
# ft_ckpt_xbase_id, ft_ckpt_xaxis_id = 'task_0_epoch_last', 'task_1_epoch_0_step_199' 
# ckpt_xbase = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-09-15-23-53/checkpoint_llm_{ft_ckpt_xbase_id}'  
# ckpt_xaxis = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-09-15-23-53/checkpoint_llm_{ft_ckpt_xaxis_id}'  

# # Direction 2
# ft_ckpt_ybase_id, ft_ckpt_yaxis_id = 'task_1_epoch_0_step_199', 'task_1_epoch_23'
# ckpt_ybase = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-09-15-23-53/checkpoint_llm_{ft_ckpt_ybase_id}'  
# ckpt_yaxis = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-08-23-10-46/checkpoint_llm_{ft_ckpt_yaxis_id}' 


# Exp: REPLAY020-ReAlign (2024.09.12, the same as REPLAY020, but we only focus on the first 0-199 steps (Undo Alignment) and 199-249 steps (ReAlignment))
# Direction 1
ft_ckpt_xbase_id, ft_ckpt_xaxis_id = 'task_1_epoch_0_step_99', 'task_1_epoch_0_step_179' 
ckpt_xbase = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-09-15-23-53/checkpoint_llm_{ft_ckpt_xbase_id}'  
ckpt_xaxis = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-09-15-23-53/checkpoint_llm_{ft_ckpt_xaxis_id}'  

# Direction 2
ft_ckpt_ybase_id, ft_ckpt_yaxis_id = 'task_1_epoch_0_step_179', 'task_1_epoch_0_step_239'
ckpt_ybase = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-09-15-23-53/checkpoint_llm_{ft_ckpt_ybase_id}'  
ckpt_yaxis = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-09-15-23-53/checkpoint_llm_{ft_ckpt_yaxis_id}' 


# Exp: REPLAY050 (2024.09.10, Start from model finetuned on Task0, then directly finetune on Task1 + 50% Task 0 (data replay))
# # Direction 1
# ft_ckpt_xbase_id, ft_ckpt_xaxis_id = 'task_0_epoch_last', 'task_1_epoch_0_step_199' 
# ckpt_xbase = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-10-18-43-58/checkpoint_llm_{ft_ckpt_xbase_id}'  
# ckpt_xaxis = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-10-18-43-58/checkpoint_llm_{ft_ckpt_xaxis_id}'  

# # Direction 2
# ft_ckpt_ybase_id, ft_ckpt_yaxis_id = 'task_1_epoch_0_step_199', 'task_1_epoch_23'
# ckpt_ybase = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-10-18-43-58/checkpoint_llm_{ft_ckpt_ybase_id}'  
# ckpt_yaxis = f'/dev_data/zjh/LLM_CL/experiments/SOTA-pythia-160m-bio-finetune_task0-saveckpt-REPLAY_0/2024-09-09-17-41-48/checkpoint_llm_{ft_ckpt_yaxis_id}' 



backbone = 'EleutherAI/pythia-160m-deduped'

tokenizer = AutoTokenizer.from_pretrained(backbone)

config_ckpt_xbase = AutoConfig.from_pretrained(ckpt_xbase)
config_ckpt_xaxis = AutoConfig.from_pretrained(ckpt_xaxis)
config_ckpt_ybase = AutoConfig.from_pretrained(ckpt_ybase)
config_ckpt_yaxis = AutoConfig.from_pretrained(ckpt_yaxis)

model_xbase = AutoModelForCausalLM.from_pretrained(ckpt_xbase, config=config_ckpt_xbase) 
model_xaxis = AutoModelForCausalLM.from_pretrained(ckpt_xaxis, config=config_ckpt_xaxis) 
model_ybase = AutoModelForCausalLM.from_pretrained(ckpt_ybase, config=config_ckpt_ybase) 
model_yaxis = AutoModelForCausalLM.from_pretrained(ckpt_yaxis, config=config_ckpt_yaxis) 

# 61 * 61 models
n_model = 61

params = get_params(default_cfg_path='./config/CIT/biography_qa_task6_test_only/SEQ.yaml',is_run_in_ipynb=True)
params.__setattr__('backbone_cache_path',ckpt_xbase)
params.__setattr__('load_llm_ckpt',True)
params.__setattr__('batch_size',128)

# Initialize Accelerator
accelerator = Accelerator()
# Dataset
CL_dataset = get_dataset(params)
model = get_model(params, CL_dataset, accelerator)  
# Result 
save_dir = os.path.abspath('./visualization-landscape')
log_file    = f'log_split_{split_id}.txt'

for x_ratio in np.array_split(np.linspace(-1,7,41),num_split)[split_id]:

    for y_ratio in np.linspace(-1,12,n_model):

        for (n1,p1), (n2,p2), (n3,p3), (n4,p4), (n5,p5) in zip(model_xbase.named_parameters(), 
                                                            model_xaxis.named_parameters(),
                                                            model_ybase.named_parameters(), 
                                                            model_yaxis.named_parameters(),
                                                            model.model.named_parameters()):
    
            assert n1 == n2 and n2 == n3 and n3 == n4 and n4 == n5
            p5.data = p1.data + (p2.data - p1.data) * x_ratio + (p4.data - p3.data) * y_ratio
        model.model.cuda()

        loss_0 = compute_loss(model.model, model.train_loader_list[0])
        loss_1 = compute_loss(model.model, model.train_loader_list[1])

        print(f'Ratio ({x_ratio},{y_ratio}); Task 0 Loss = {loss_0}; Task 1 Loss = {loss_1}')
        with open(os.path.join(save_dir,log_file),'a') as f:
            f.write(f'{x_ratio},{y_ratio},{loss_0},{loss_1}\n')