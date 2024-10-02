from transformers import AutoTokenizer, GPTNeoXForCausalLM
from residual_stream_shift_analysis.data_module import collate_fn_for_fine_tuning_qa_dataset
from training.utility import FineTuningTrainPersonIndexInfoList, QARawData, attribute_list
from training.data_module import QADataset, filter_qa_data_with_token_info
import os
import json
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm, trange
from typing import Dict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def construct_residual_stream(
        model_path: str,
        qa_data_path: str,
        person_index_info_list: FineTuningTrainPersonIndexInfoList,
        save_root_dir: str,
        batch_size: int):
    raw_data: QARawData = json.load(open(qa_data_path))
    filtered_data = filter_qa_data_with_token_info(raw_data, person_index_info_list)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=32, padding_side="right", use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    model = GPTNeoXForCausalLM.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    qa_dataset = QADataset(tokenizer, filtered_data, tokenizer.model_max_length)
    data_loader = DataLoader(
        qa_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_for_fine_tuning_qa_dataset)
    residual_stream_of_last_token_dict = {}
    for batch, query_end_index in tqdm(data_loader):
        hidden_state_select_helper = torch.arange(0, tokenizer.model_max_length * query_end_index.shape[0],
                                                  tokenizer.model_max_length)
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
            query_end_index = query_end_index.cuda()
            hidden_state_select_helper = hidden_state_select_helper.cuda()
        batch['output_hidden_states'] = True
        with torch.no_grad():
            output = model(**batch)
        hidden_states = output['hidden_states']
        query_end_index = query_end_index + hidden_state_select_helper
        for i, hidden_state in enumerate(hidden_states):
            hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1])
            hidden_state_selected = hidden_state[query_end_index]
            hidden_state_selected = hidden_state_selected.cpu()  # remove from GPU to save memory
            residual_stream_of_last_token_dict.setdefault(i, []).append(hidden_state_selected)  # wrong type checking
    for key, value in residual_stream_of_last_token_dict.items():
        residual_stream_of_last_token_dict[key] = torch.concatenate(value, dim=0)
        save_dir = os.path.join(save_root_dir, str(key))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(residual_stream_of_last_token_dict[key], os.path.join(save_dir, 'residual_stram.pt'))


def get_projection_value(
        before_residual_stream_root_dir: str,
        after_residual_stream_root_dir: str,
        direction_x_type,
) -> Dict[int, Dict[str, torch.Tensor]]:
    assert len(os.listdir(before_residual_stream_root_dir)) == len(os.listdir(after_residual_stream_root_dir))
    layer_num = len(os.listdir(before_residual_stream_root_dir))
    result = {}
    for layer in trange(layer_num, desc='Calculating projection value'):
        before_residual_stream = torch.load(
            os.path.join(before_residual_stream_root_dir, str(layer), 'residual_stram.pt'))
        after_residual_stream = torch.load(
            os.path.join(after_residual_stream_root_dir, str(layer), 'residual_stram.pt'))
        # get the direction of the main principal component, which will be the direction of y-axis
        before_residual_stream_mean = before_residual_stream.mean(dim=0)
        s, u, v = torch.pca_lowrank(before_residual_stream, center=True)
        direction_y = v[:, 0]
        direction_y = direction_y / direction_y.norm()  # shape: (hidden_size,)
        # get the direction of the mean of the shift, which will be the direction of x-axis
        match direction_x_type:
            case 'mean_shift':
                direction_x = after_residual_stream.mean(dim=0) - before_residual_stream.mean(dim=0)
            case 'first_principal_component':
                direction_x = v[:, 1]
            case _:
                raise ValueError('Invalid x_axis')
        direction_x = direction_x / direction_x.norm()  # shape: (hidden_size,)
        # calculate the projection value
        before_residual_stream_centered = before_residual_stream - before_residual_stream_mean
        before_residual_stram_projection_x = before_residual_stream_centered @ direction_x
        before_residual_stram_projection_y = before_residual_stream_centered @ direction_y
        after_residual_stream_centered = after_residual_stream - before_residual_stream_mean
        after_residual_stram_projection_x = after_residual_stream_centered @ direction_x
        after_residual_stram_projection_y = after_residual_stream_centered @ direction_y
        result[layer] = {
            'before_residual_stram_projection_x': before_residual_stram_projection_x,
            'before_residual_stram_projection_y': before_residual_stram_projection_y,
            'after_residual_stram_projection_x': after_residual_stram_projection_x,
            'after_residual_stram_projection_y': after_residual_stram_projection_y,
        }
    return result


def construct_many_residual_stream():
    project_root = '/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting'
    model_path_after_pre_training_on_task0 = os.path.join(project_root, 'model/gpt-neox/processed_0720_v0730/'
                                                                        'config_v0806/multi5_permute_fullname/task_0/'
                                                                        'final_model')
    model_path_after_fine_tuning_on_task0 = os.path.join(project_root, 'model/gpt-neox/processed_0720_v0730/'
                                                                       'config_v0806/multi5_permute_fullname/task_0/'
                                                                       'fine_tuning/0_0/final_model')
    qa_data_path = os.path.join(project_root, 'data/processed_0720_v0730/qa/all.json')
    sample_person_index_info_list_used_for_task_0_fine_tuning_training = [{'start': 0, 'end': 5_000}]
    sample_person_index_info_list_used_for_task_0_fine_tuning_testing = [{'start': 50_000, 'end': 55_000}]
    sample_person_index_info_list_used_for_task_0_fine_tuning_training_and_testing = (
            sample_person_index_info_list_used_for_task_0_fine_tuning_training +
            sample_person_index_info_list_used_for_task_0_fine_tuning_testing
    )
    save_root_dir = './residual_stream_shift_analysis/tensor_warehouse'
    for model_path, person_index_info_list, save_root_dir in [
        (model_path_after_pre_training_on_task0,
         sample_person_index_info_list_used_for_task_0_fine_tuning_training,
         os.path.join(save_root_dir, 'after_pre_training_on_task0__range_0_to_5000')),
        (model_path_after_pre_training_on_task0,
         sample_person_index_info_list_used_for_task_0_fine_tuning_testing,
         os.path.join(save_root_dir, 'after_pre_training_on_task0__range_50000_to_55000')),
        (model_path_after_pre_training_on_task0,
         sample_person_index_info_list_used_for_task_0_fine_tuning_training_and_testing,
         os.path.join(save_root_dir, 'after_pre_training_on_task0__range_0_to_5000_50000_to_55000')),
        (model_path_after_fine_tuning_on_task0,
         sample_person_index_info_list_used_for_task_0_fine_tuning_training,
         os.path.join(save_root_dir, 'after_fine_tuning_on_task0__range_0_to_5000')),
        (model_path_after_fine_tuning_on_task0,
         sample_person_index_info_list_used_for_task_0_fine_tuning_testing,
         os.path.join(save_root_dir, 'after_fine_tuning_on_task0__range_50000_to_55000')),
        (model_path_after_fine_tuning_on_task0,
         sample_person_index_info_list_used_for_task_0_fine_tuning_training_and_testing,
         os.path.join(save_root_dir, 'after_fine_tuning_on_task0__range_0_to_5000_50000_to_55000')),
    ]:
        construct_residual_stream(
            model_path,
            qa_data_path,
            person_index_info_list,
            save_root_dir,
            512
        )


def draw_graph(direction_x_type):
    residual_stream_output_root_dir = './residual_stream_shift_analysis/tensor_warehouse/v0824'
    for root_dir_pair in [
        (os.path.join(residual_stream_output_root_dir, 'after_pre_training_on_task0__range_0_to_5000'),
         os.path.join(residual_stream_output_root_dir, 'after_fine_tuning_on_task0__range_0_to_5000')),
        (os.path.join(residual_stream_output_root_dir, 'after_pre_training_on_task0__range_50000_to_55000'),
         os.path.join(residual_stream_output_root_dir, 'after_fine_tuning_on_task0__range_50000_to_55000')),
        (os.path.join(residual_stream_output_root_dir, 'after_pre_training_on_task0__range_0_to_5000_50000_to_55000'),
         os.path.join(residual_stream_output_root_dir, 'after_fine_tuning_on_task0__range_0_to_5000_50000_to_55000')),
    ]:
        point_info_dict = get_projection_value(root_dir_pair[0], root_dir_pair[1], direction_x_type)
        for layer, point_info in point_info_dict.items():
            # layer = 12
            # point_info = point_info_dict[layer]
            sample_num = len(point_info['before_residual_stram_projection_x'])
            assert sample_num % 6 == 0
            before_residual_stream_df = pd.DataFrame({
                'x': point_info['before_residual_stram_projection_x'],
                'y': point_info['before_residual_stram_projection_y'],
                'attribute': attribute_list * (sample_num // 6),
                'type': ['before'] * sample_num
            })
            after_residual_stream_df = pd.DataFrame({
                'x': point_info['after_residual_stram_projection_x'],
                'y': point_info['after_residual_stram_projection_y'],
                'attribute': attribute_list * (sample_num // 6),
                'type': ['after'] * sample_num
            })
            point_num = 60
            assert point_num % 6 == 0
            before_residual_stream_df = before_residual_stream_df[:point_num]
            after_residual_stream_df = after_residual_stream_df[:point_num]
            sns.set(style='darkgrid')
            sns.set_theme(rc={
                "figure.dpi": 200,
                "figure.figsize": (18, 8)}
            )
            ax = sns.scatterplot(data=pd.concat([before_residual_stream_df, after_residual_stream_df]), x='x', y='y',
                                 style='type', hue='attribute', palette='PuOr')
            for (x1, y1, _, _), (x2, y2, _, _) in zip(before_residual_stream_df.values,
                                                      after_residual_stream_df.values):
                plt.plot([x1, x2], [y1, y2], color='gray', linestyle='--', linewidth=0.5)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.title(f'Layer {layer}')
            # uncomment the following line to save the figure
            # plt.savefig(
            #     f'./residual_stream_shift_analysis/figure/'
            #     f'x_axis_second_principle_component__y_axis_first_principle_component/{layer}.png')
            # plt.savefig(
            #     f'./residual_stream_shift_analysis/figure/'
            #     f'x_axis_mean_shift__y_axis_first_principle_component/{layer}.png')
            plt.show()
        break


if __name__ == '__main__':
    # construct_many_residual_stream()
    draw_graph('first_principal_component')
    draw_graph('mean_shift')
