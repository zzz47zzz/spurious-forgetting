from matplotlib.pyplot import figure
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from residual_stream_shift_analysis.data_module import ResidualStreamQADataset
from training.utility import FineTuningTrainPersonIndexInfoList, QARawData, attribute_list
from training.data_module import filter_qa_data_with_token_info
import os
import json
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm, trange
from typing import Dict, List, Tuple
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from enum import StrEnum, auto


class FigureType(StrEnum):
    X_AXIS_BEFORE_SECOND_PRINCIPLE_COMPONENT_Y_AXIS_BEFORE_FIRST_PRINCIPLE_COMPONENT = auto()
    X_AXIS_AFTER_FIRST_PRINCIPLE_COMPONENT_Y_AXIS_BEFORE_FIRST_PRINCIPLE_COMPONENT = auto()
    X_AXIS_AFTER_SECOND_PRINCIPLE_COMPONENT_Y_AXIS_AFTER_FIRST_PRINCIPLE_COMPONENT = auto()
    X_AXIS_MEAN_SHIFT_Y_AXIS_BEFORE_FIRST_PRINCIPLE_COMPONENT = auto()
    X_AXIS_MEAN_SHIFT_Y_AXIS_AFTER_FIRST_PRINCIPLE_COMPONENT = auto()


def construct_residual_stream(
        model_path: str,
        qa_data_path: str,
        person_index_info_list: FineTuningTrainPersonIndexInfoList,
        save_root_dir: str,
        batch_size: int) -> None:
    raw_data: QARawData = json.load(open(qa_data_path))
    filtered_data = filter_qa_data_with_token_info(raw_data, person_index_info_list)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=32, padding_side="right", use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    model = GPTNeoXForCausalLM.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    dataset = ResidualStreamQADataset(tokenizer, filtered_data, tokenizer.model_max_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    result_dict: Dict[int, Dict[str, List[torch.Tensor]]] = {}
    for batch in tqdm(data_loader, 'Calculating the residual stream'):
        hidden_state_select_helper = torch.arange(0, tokenizer.model_max_length * batch['input_ids'].shape[0],
                                                  tokenizer.model_max_length)
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
            hidden_state_select_helper = hidden_state_select_helper.cuda()
        batch['output_hidden_states'] = True
        query_end_indexes = batch.pop('query_end_indexes').reshape(-1)
        attributes = batch.pop('attributes').reshape(-1)
        with torch.no_grad():
            output = model(**batch)
        hidden_states = output['hidden_states']
        query_end_indexes = query_end_indexes + hidden_state_select_helper
        hidden_state_to_last_query_token_hidden_state_dict = {}
        for i, hidden_state in enumerate(hidden_states):
            hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1])
            hidden_state_selected = hidden_state[query_end_indexes]
            hidden_state_to_last_query_token_hidden_state_dict[i] = hidden_state_selected
        for hidden_state_index in hidden_state_to_last_query_token_hidden_state_dict:
            hidden_state = hidden_state_to_last_query_token_hidden_state_dict[hidden_state_index]
            result_dict.setdefault(hidden_state_index, {})
            for attribute_id in range(len(attribute_list)):
                index = (attributes == attribute_id).nonzero()
                index = index.reshape(-1)
                hidden_state_selected = hidden_state[index]
                hidden_state_selected = hidden_state_selected.cpu()  # remove from GPU to save memory
                result_dict[hidden_state_index].setdefault(attribute_list[attribute_id], []).append(
                    hidden_state_selected)
    for hidden_state_index in tqdm(result_dict, 'Writing result to disk'):
        for attribute in result_dict[hidden_state_index]:
            result_to_write = torch.cat(result_dict[hidden_state_index][attribute], dim=0)
            save_dir = os.path.join(save_root_dir, f'hidden_state_{hidden_state_index}', attribute)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(result_to_write, os.path.join(save_dir, 'residual_stram.pt'))


def get_projection_value(
        residual_stream_dir_of_hidden_state: str,
        x_direction: torch.Tensor,  # expect shape: (d, 1)
        y_direction: torch.Tensor,  # expect shape: (d, 1)
        x_mean: torch.Tensor = None,
        y_mean: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    if not (len(x_direction.shape) == 2 and x_direction.shape[1] == 1):
        x_direction = x_direction.reshape(-1, 1)
    if not (len(y_direction.shape) == 2 and y_direction.shape[1] == 1):
        y_direction = y_direction.reshape(-1, 1)
    if x_mean is None:
        x_mean = torch.zeros(x_direction.shape[0])
    if y_mean is None:
        y_mean = torch.zeros(y_direction.shape[0])
    x_direction = x_direction / torch.norm(x_direction)
    y_direction = y_direction / torch.norm(y_direction)
    result = {}
    for attribute in attribute_list:
        # shape: (n, d)
        residual_stream = torch.load(os.path.join(residual_stream_dir_of_hidden_state, attribute, 'residual_stram.pt'))
        projection_x = (residual_stream - x_mean) @ x_direction
        projection_y = (residual_stream - y_mean) @ y_direction
        result[attribute] = torch.cat([projection_x, projection_y], dim=1)
    return result


def perform_svd_decomposition_of_hidden_state(
        residual_stream_dir_of_hidden_state: str,
        center: bool) -> Dict[str, torch.Tensor]:
    residual_stream_list: List[torch.Tensor] = []
    for attribute in attribute_list:
        residual_stream = torch.load(os.path.join(residual_stream_dir_of_hidden_state, attribute, 'residual_stram.pt'))
        residual_stream_list.append(residual_stream)
    residual_stream = torch.cat(residual_stream_list, dim=0)
    s, u, v = torch.pca_lowrank(residual_stream, center=center)
    return {
        's': s, 'u': u, 'v': v,
        'mean': residual_stream.mean(dim=0) if center else None
    }


def calculate_mean_shift_of_residual_stream(
        residual_stream_dir_of_before_hidden_state: str,
        residual_stream_dir_of_after_hidden_state: str) -> torch.Tensor:
    before_residual_stream_list: List[torch.Tensor] = []
    after_residual_stream_list: List[torch.Tensor] = []
    for attribute in attribute_list:
        before_residual_stream = torch.load(
            os.path.join(residual_stream_dir_of_before_hidden_state, attribute, 'residual_stram.pt'))
        before_residual_stream_list.append(before_residual_stream)
        after_residual_stream = torch.load(
            os.path.join(residual_stream_dir_of_after_hidden_state, attribute, 'residual_stram.pt'))
        after_residual_stream_list.append(after_residual_stream)
    before_residual_stream = torch.cat(before_residual_stream_list, dim=0)
    before_mean = before_residual_stream.mean(dim=0)
    after_residual_stream = torch.cat(after_residual_stream_list, dim=0)
    after_mean = after_residual_stream.mean(dim=0)
    return after_mean - before_mean


def get_information_for_plotting(
        before_projection_result: Dict[str, torch.Tensor],
        after_projection_result: Dict[str, torch.Tensor],
        person_num_of_each_attribute: int = None
) -> Tuple[pd.DataFrame, List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]]:
    # 1. attribute is one of the attribute_list
    # 2. label can be 'before' or 'after'
    # 3. the regression result is calculated by all the points in before_projection_result and after_projection_result,
    #    instead of the selected points that are used for plotting
    df_for_plotting = None
    point_pair_list = []
    for attribute in attribute_list:
        before_projection = before_projection_result[attribute]
        after_projection = after_projection_result[attribute]
        if person_num_of_each_attribute is not None:
            before_projection = before_projection[:person_num_of_each_attribute]
            after_projection = after_projection[:person_num_of_each_attribute]
        before_df = pd.DataFrame(before_projection, columns=['x', 'y'])
        before_df['attribute'] = attribute
        before_df['label'] = 'before'
        after_df = pd.DataFrame(after_projection, columns=['x', 'y'])
        after_df['attribute'] = attribute
        after_df['label'] = 'after'
        # region handle df_for_plotting
        if df_for_plotting is None:
            df_for_plotting = pd.concat([before_df, after_df], ignore_index=True)
        else:
            df_for_plotting = pd.concat([df_for_plotting, before_df, after_df], ignore_index=True)
        # endregion
        # region handle point_pair_list
        for i in range(len(before_projection)):
            point_pair_list.append(((before_projection[i, 0], before_projection[i, 1]),
                                    (after_projection[i, 0], after_projection[i, 1])))
        # endregion
    return df_for_plotting, point_pair_list


def draw_figure_by_hidden_state_index_to_plotting_info_dict(
        hidden_state_index_to_plotting_info_dict,
        x_min: np.float64,
        x_max: np.float64,
        y_min: np.float64,
        y_max: np.float64,
        figure_title_prefix: str,
        save_root_dir: str = None
):
    x_axis_min = x_min - 0.1 * (x_max - x_min)
    x_axis_max = x_max + 0.1 * (x_max - x_min)
    y_axis_min = y_min - 0.1 * (y_max - y_min)
    y_axis_max = y_max + 0.1 * (y_max - y_min)
    sns.set_theme()
    mpl.rcParams['figure.dpi'] = 800
    hidden_state_count = len(hidden_state_index_to_plotting_info_dict)
    combined_fig, combined_ax = plt.subplots(hidden_state_count, 1, figsize=(6, 6 * hidden_state_count))
    for ax_index, hidden_state_index in tqdm(enumerate(hidden_state_index_to_plotting_info_dict),
                                             desc='Drawing the figure',
                                             total=hidden_state_count):
        df_for_plotting = hidden_state_index_to_plotting_info_dict[hidden_state_index]['df_for_plotting']
        point_pair_list = hidden_state_index_to_plotting_info_dict[hidden_state_index]['point_pair_list']
        # region set figure style
        sns.set_theme()
        # https://stackoverflow.com/questions/51937381/increase-dpi-of-plt-show
        mpl.rcParams['figure.dpi'] = 800
        # endregion
        # region paint the figure
        fig, ax_of_single_plot = plt.subplots()
        for ax in [combined_ax[ax_index], ax_of_single_plot]:
            for point_pair in point_pair_list:
                ax.plot([point_pair[0][0], point_pair[1][0]], [point_pair[0][1], point_pair[1][1]],
                        color='gray', linestyle='--', linewidth=0.5)
            sns.scatterplot(data=df_for_plotting, x='x', y='y', hue='attribute', style='label', palette='gist_stern',
                            ax=ax)
            # endregion
            # region set figure style
            box = ax.get_position()
            # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_xlim([x_axis_min, x_axis_max])
            ax.set_ylim([y_axis_min, y_axis_max])
            ax.set_title(f'{figure_title_prefix}{hidden_state_index}')
            ax.set_ylim([y_axis_min, y_axis_max])
            ax.set_title(f'{figure_title_prefix}{hidden_state_index}')
        # endregion
        if save_root_dir is not None:
            if not os.path.exists(save_root_dir):
                os.makedirs(save_root_dir)
            save_path = os.path.join(save_root_dir, f'{figure_title_prefix}{hidden_state_index}.png')
            fig.savefig(save_path)
        else:
            fig.show()
        plt.close(fig)  # close the figure to save memory
    if save_root_dir is not None:
        combined_fig.savefig(os.path.join(save_root_dir, 'combined.png'))
    else:
        print('The combined figure is not shown, because it is too large')
    plt.close(combined_fig)


def draw_figure_of_selected_hidden_state(
        residual_stream_root_dir_of_before_hidden_state: str,
        residual_stream_root_dir_of_after_hidden_state: str,
        figure_type: FigureType,
        hidden_state_range: List[int],
        center: bool,  # indicate whether to center the hidden state before SVD decomposition,
        person_num_of_each_attribute: int,
        figure_title_prefix: str,
        save_root_dir: str = None  # if None, show the figure, otherwise save the figure to the path
) -> None:
    # region calculate the projection value
    hidden_state_index_to_plotting_info_dict = {}
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    slope_information_dict = {'hidden_state_index_first': {}, 'attribute_key_first': {}}
    for hidden_state_index in tqdm(hidden_state_range, 'Calculating the projection value and slope'):
        before_dir = os.path.join(residual_stream_root_dir_of_before_hidden_state, f'hidden_state_{hidden_state_index}')
        after_dir = os.path.join(residual_stream_root_dir_of_after_hidden_state, f'hidden_state_{hidden_state_index}')

        # region calculate x_direction, y_direction, x_mean, y_mean
        match figure_type:
            case FigureType.X_AXIS_BEFORE_SECOND_PRINCIPLE_COMPONENT_Y_AXIS_BEFORE_FIRST_PRINCIPLE_COMPONENT:
                before_svd_result = perform_svd_decomposition_of_hidden_state(before_dir, center)
                x_direction = before_svd_result['v'][:, 1]
                y_direction = before_svd_result['v'][:, 0]
                x_mean = before_svd_result['mean']
                y_mean = before_svd_result['mean']
            case FigureType.X_AXIS_AFTER_FIRST_PRINCIPLE_COMPONENT_Y_AXIS_BEFORE_FIRST_PRINCIPLE_COMPONENT:
                before_svd_result = perform_svd_decomposition_of_hidden_state(before_dir, center)
                after_svd_result = perform_svd_decomposition_of_hidden_state(after_dir, center)
                x_direction = after_svd_result['v'][:, 0]
                y_direction = before_svd_result['v'][:, 0]
                x_mean = before_svd_result['mean']
                y_mean = after_svd_result['mean']
            case FigureType.X_AXIS_AFTER_SECOND_PRINCIPLE_COMPONENT_Y_AXIS_AFTER_FIRST_PRINCIPLE_COMPONENT:
                after_svd_result = perform_svd_decomposition_of_hidden_state(after_dir, center)
                x_direction = after_svd_result['v'][:, 1]
                y_direction = after_svd_result['v'][:, 0]
                x_mean = after_svd_result['mean']
                y_mean = after_svd_result['mean']
            case FigureType.X_AXIS_MEAN_SHIFT_Y_AXIS_BEFORE_FIRST_PRINCIPLE_COMPONENT:
                before_svd_result = perform_svd_decomposition_of_hidden_state(before_dir, center)
                x_direction = calculate_mean_shift_of_residual_stream(before_dir, after_dir)
                y_direction = before_svd_result['v'][:, 0]
                x_mean = before_svd_result['mean']
                y_mean = before_svd_result['mean']
            case FigureType.X_AXIS_MEAN_SHIFT_Y_AXIS_AFTER_FIRST_PRINCIPLE_COMPONENT:
                after_svd_result = perform_svd_decomposition_of_hidden_state(after_dir, center)
                x_direction = calculate_mean_shift_of_residual_stream(before_dir, after_dir)
                y_direction = after_svd_result['v'][:, 0]
                x_mean = after_svd_result['mean']
                y_mean = after_svd_result['mean']
            case _:
                raise NotImplementedError(f'Unknown figure type: {figure_type}')
        # endregion
        before_projection_result = get_projection_value(before_dir,
                                                        x_direction,
                                                        y_direction,
                                                        x_mean,
                                                        y_mean)
        after_projection_result = get_projection_value(after_dir,
                                                       x_direction,
                                                       y_direction,
                                                       x_mean,
                                                       y_mean)
        # region get information for plotting
        df_for_plotting, point_pair_list = get_information_for_plotting(
            before_projection_result, after_projection_result, person_num_of_each_attribute)
        x_min = min(x_min, df_for_plotting['x'].min())
        x_max = max(x_max, df_for_plotting['x'].max())
        y_min = min(y_min, df_for_plotting['y'].min())
        y_max = max(y_max, df_for_plotting['y'].max())
        hidden_state_index_to_plotting_info_dict[hidden_state_index] = {
            'df_for_plotting': df_for_plotting,
            'point_pair_list': point_pair_list
        }
        # endregion
        # region calculate the slope information
        slope_information_dict['hidden_state_index_first'][hidden_state_index] = {}
        for key in attribute_list + ['all']:
            # the if-else statement is ugly, but I don't want to refactor it
            if key != 'all':
                before_projection = before_projection_result[key]
                after_projection = after_projection_result[key]
            else:
                before_projection = torch.cat(
                    [before_projection_result[attribute] for attribute in attribute_list], dim=0)
                after_projection = torch.cat(
                    [after_projection_result[attribute] for attribute in attribute_list], dim=0)
            before_projection_mean = before_projection.mean(dim=0)
            after_projection_mean = after_projection.mean(dim=0)
            slope = ((after_projection_mean[1] - before_projection_mean[1]) /
                     (after_projection_mean[0] - before_projection_mean[0]))

            slope_information_dict['hidden_state_index_first'][hidden_state_index][key] = {
                'slope': slope.item(),
                'before_mean': {'x': before_projection_mean[0].item(), 'y': before_projection_mean[1].item()},
                'after_mean': {'x': after_projection_mean[0].item(), 'y': after_projection_mean[1].item()}
            }

            slope_information_dict['attribute_key_first'].setdefault(key, {})
            slope_information_dict['attribute_key_first'][key][hidden_state_index] = {
                'slope': slope.item(),
                'before_mean': {'x': before_projection_mean[0].item(), 'y': before_projection_mean[1].item()},
                'after_mean': {'x': after_projection_mean[0].item(), 'y': after_projection_mean[1].item()}
            }
        # endregion
    # endregion

    draw_figure_by_hidden_state_index_to_plotting_info_dict(
        hidden_state_index_to_plotting_info_dict, x_min, x_max, y_min, y_max,
        figure_title_prefix, save_root_dir
    )
    if save_root_dir is not None:
        json.dump(slope_information_dict, open(os.path.join(save_root_dir, 'slope_information.json'), 'w'), indent=4)
    else:
        print(slope_information_dict)


def calculate_angle_between_the_principal_component_of_residual_stream(
        residual_stream_root_dir_of_before_hidden_state: str,
        residual_stream_root_dir_of_after_hidden_state: str,
        hidden_state_range: List[int],
        center: bool,
        principal_component_index: int,
        save_root_dir: str = None
):
    result = {}
    for hidden_state_index in tqdm(hidden_state_range, 'Calculating the angle between the first principal component'):
        before_dir = os.path.join(residual_stream_root_dir_of_before_hidden_state, f'hidden_state_{hidden_state_index}')
        after_dir = os.path.join(residual_stream_root_dir_of_after_hidden_state, f'hidden_state_{hidden_state_index}')
        # calculate the first principal component of the hidden state
        before_svd_result = perform_svd_decomposition_of_hidden_state(before_dir, center)
        after_svd_result = perform_svd_decomposition_of_hidden_state(after_dir, center)
        # calculate angle
        before_principal_component = before_svd_result['v'][:, principal_component_index]
        after_principal_component = after_svd_result['v'][:, principal_component_index]
        angle = torch.acos(torch.dot(before_principal_component, after_principal_component) /
                           (torch.norm(before_principal_component) * torch.norm(after_principal_component)))
        angle = angle * 180 / torch.pi
        result[f'hidden_state_{hidden_state_index}'] = angle.item()
    if save_root_dir is not None:
        if not os.path.exists(save_root_dir):
            os.makedirs(save_root_dir)
        save_path = os.path.join(save_root_dir,
                                 f'angle_between_the_principal_component_{principal_component_index}.json')
        json.dump(result, open(save_path, 'w'), indent=4)
    else:
        print(result)


def preliminary_exp():
    def construct_many_residual_stream_for_preliminary_exp() -> None:
        project_root = '/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting'
        model_path_after_pre_training_on_task0 = os.path.join(
            project_root, 'model/gpt-neox/processed_0720_v0730/config_v0806/multi5_permute_fullname/task_0/final_model')
        model_path_after_fine_tuning_on_task0 = os.path.join(
            project_root, 'model/gpt-neox/processed_0720_v0730/config_v0806/multi5_permute_fullname/'
                          'task_0/fine_tuning/0_0/final_model')
        qa_data_path = os.path.join(project_root, 'data/processed_0720_v0730/qa/all.json')
        sample_person_index_info_list_used_for_task_0_fine_tuning_training = [{'start': 0, 'end': 5_000}]
        save_root_dir = './residual_stream_shift_analysis/tensor_warehouse/v0826/preliminary_exp'
        for model_path, person_index_info_list, save_root_dir in [
            (model_path_after_pre_training_on_task0,
             sample_person_index_info_list_used_for_task_0_fine_tuning_training,
             os.path.join(save_root_dir, 'after_pre_training_on_task0__range_0_to_5000')),
            (model_path_after_fine_tuning_on_task0,
             sample_person_index_info_list_used_for_task_0_fine_tuning_training,
             os.path.join(save_root_dir, 'after_fine_tuning_on_task0__range_0_to_5000')),
        ]:
            construct_residual_stream(
                model_path,
                qa_data_path,
                person_index_info_list,
                save_root_dir,
                512
            )

    construct_many_residual_stream_for_preliminary_exp()
    result_dir = f'./residual_stream_shift_analysis/analysis_result/v0826/preliminary_exp'
    before_root_dir = (f'./residual_stream_shift_analysis/tensor_warehouse/v0826/preliminary_exp/'
                       f'after_pre_training_on_task0__range_0_to_5000')
    after_root_dir = (f'./residual_stream_shift_analysis/tensor_warehouse/v0826/preliminary_exp/'
                      f'after_fine_tuning_on_task0__range_0_to_5000')
    for center in [True, False]:
        save_dir = os.path.join(result_dir,
                                'before_task0_after_pre_training__after_task0_after_fine_tuning__range_0_to_5000')
        if center:
            save_dir = os.path.join(save_dir, 'center')
        else:
            save_dir = os.path.join(save_dir, 'no_center')
        calculate_angle_between_the_principal_component_of_residual_stream(
            before_root_dir,
            after_root_dir,
            list(range(13)),
            center,
            0,
            save_root_dir=save_dir
        )
        for figure_type in list(FigureType):
            figure_type = FigureType(figure_type)  # this line of code is useless, it is used to suppress the warning
            draw_figure_of_selected_hidden_state(
                before_root_dir,
                after_root_dir,
                figure_type,
                list(range(13)),
                center,
                10,
                f'Hidden_State_',
                save_root_dir=os.path.join(save_dir, figure_type.name.lower())
            )


def exp_0827():
    # model_from_b7_server is downloaded from b7_server, the location is:
    # /dev_data/zjh/physics-of-forgetting-in-llm/physics_of_forgetting/model/gpt-neox/processed_0720_v0730/config_v0806/multi5_permute_fullname/task_0
    # data in analysis_result is generated in 20240829, using the code below
    # new model is trained on 20240903
    project_root = '/dev_data/cxd/nlp/physics-of-forgetting-in-llm/physics_of_forgetting'
    pre_trained_final_model_path = os.path.join(
        project_root, 'model_from_b7_server/multi5_permute_fullname/task_0/final_model')
    fine_tuning_model_root_dir = os.path.join(
        project_root, 'model_from_b7_server/multi5_permute_fullname/task_0/fine_tuning')
    qa_data_path = os.path.join(project_root, 'data/processed_0720_v0730/qa/all.json')
    save_root_dir = os.path.join(project_root, 'residual_stream_shift_analysis/tensor_warehouse/v0826/exp_0827')
    residual_stream_info_list = [
        {
            'before_model_path': pre_trained_final_model_path,
            'after_model_path': os.path.join(
                fine_tuning_model_root_dir, '0_0/final_model'),
            'person_index_info_list': [{'start': 0, 'end': 5_000}],
            'before_save_root_dir': os.path.join(
                save_root_dir, 'after_pre_training_on_task0__range_0_to_5000'),
            'after_save_root_dir': os.path.join(
                save_root_dir, 'after_fine_tuning_on_task0_for_625000_step__range_0_to_5000'),
        },  # learn task0
        {
            'before_model_path': os.path.join(
                fine_tuning_model_root_dir, '1_1_first_200steps_eval_every_step/checkpoint-100'),
            'after_model_path': os.path.join(
                fine_tuning_model_root_dir, '1_1_first_200steps_eval_every_step/checkpoint-150'),
            'person_index_info_list': [{'start': 0, 'end': 5_000}],
            'before_save_root_dir': os.path.join(
                save_root_dir, 'after_fine_tuning_on_task1_for_100_step__range_0_to_5000'),
            'after_save_root_dir': os.path.join(
                save_root_dir, 'after_fine_tuning_on_task1_for_150_step__range_0_to_5000'),
        },  # forget task 0 (step100 to step150)
        {
            'before_model_path': os.path.join(
                fine_tuning_model_root_dir, '1_1_first_200steps_eval_every_step/checkpoint-200'),
            'after_model_path': os.path.join(
                fine_tuning_model_root_dir, '1_1/final_model'),
            'person_index_info_list': [{'start': 0, 'end': 5_000}],
            'before_save_root_dir': os.path.join(
                save_root_dir, 'after_fine_tuning_on_task1_for_200_step__range_0_to_5000'),
            'after_save_root_dir': os.path.join(
                save_root_dir, 'after_fine_tuning_on_task1_for_625000_step__range_0_to_5000'),
        },  # learn task1 after forgetting task0 (step200 to step62.5k), what happen to task0?
        {
            'before_model_path': os.path.join(
                fine_tuning_model_root_dir, '1_1_first_200steps_eval_every_step/checkpoint-200'),
            'after_model_path': os.path.join(
                fine_tuning_model_root_dir, '1_1/final_model'),
            'person_index_info_list': [{'start': 100_000, 'end': 105_000}],
            'before_save_root_dir': os.path.join(
                save_root_dir, 'after_fine_tuning_on_task1_for_200_step__range_100000_to_105000'),
            'after_save_root_dir': os.path.join(
                save_root_dir, 'after_fine_tuning_on_task1_for_625000_step__range_100000_to_105000'),
        },  # learn task1 after forgetting task0 (step200 to step62.5k), what happen to task1?
        {
            'before_model_path': os.path.join(
                fine_tuning_model_root_dir, '0_0/final_model'),
            'after_model_path': os.path.join(
                fine_tuning_model_root_dir, '1_1/final_model'),
            'person_index_info_list': [{'start': 0, 'end': 5_000}],
            'before_save_root_dir': os.path.join(
                save_root_dir, 'after_fine_tuning_on_task0_for_625000_step__range_0_to_5000'),
            'after_save_root_dir': os.path.join(
                save_root_dir, 'after_fine_tuning_on_task1_for_625000_step__range_0_to_5000'),
        },  # learn task1 after learning task0 (step0 to step62.5k), what happen to task0?
        {
            'before_model_path': os.path.join(
                fine_tuning_model_root_dir, '0_0/final_model'),
            'after_model_path': os.path.join(
                fine_tuning_model_root_dir, '1_1/final_model'),
            'person_index_info_list': [{'start': 100_000, 'end': 105_000}],
            'before_save_root_dir': os.path.join(
                save_root_dir, 'after_fine_tuning_on_task0_for_625000_step__range_100000_to_105000'),
            'after_save_root_dir': os.path.join(
                save_root_dir, 'after_fine_tuning_on_task1_for_625000_step__range_100000_to_105000'),
        },  # learn task1 after learning task0 (step0 to step62.5k), what happen to task1?
    ]
    # region construct residual stream
    for pair_info in residual_stream_info_list:
        for model_path, person_index_info_list, save_root_dir in [
            (pair_info['before_model_path'], pair_info['person_index_info_list'], pair_info['before_save_root_dir']),
            (pair_info['after_model_path'], pair_info['person_index_info_list'], pair_info['after_save_root_dir']),
        ]:
            if not os.path.exists(save_root_dir):
                construct_residual_stream(
                    model_path,
                    qa_data_path,
                    person_index_info_list,
                    save_root_dir,
                    512
                )
            else:
                print(f'{save_root_dir} already exists, skip the construction process')
    # endregion
    # region draw figure
    save_root_dir = os.path.join(project_root, './residual_stream_shift_analysis/analysis_result/v0826/exp_0827')
    for pair_info in residual_stream_info_list:
        before_residual_stream_save_dir_name = os.path.basename(pair_info['before_save_root_dir'])
        after_residual_stream_save_dir_name = os.path.basename(pair_info['after_save_root_dir'])
        before_residual_stream_label, before_range = before_residual_stream_save_dir_name.split('__range_')
        after_residual_stream_label, after_range = after_residual_stream_save_dir_name.split('__range_')
        assert before_range == after_range
        figure_save_root_dir = os.path.join(
            save_root_dir,
            f'before_{before_residual_stream_label}__after_{after_residual_stream_label}__range_{before_range}')
        for center in [True, False]:
            center_str = 'center' if center else 'no_center'
            calculate_angle_between_the_principal_component_of_residual_stream(
                pair_info['before_save_root_dir'],
                pair_info['after_save_root_dir'],
                list(range(13)),
                center,
                0,
                save_root_dir=os.path.join(figure_save_root_dir, center_str)
            )
            for figure_type in list(FigureType):
                figure_type = FigureType(figure_type)
                draw_figure_of_selected_hidden_state(
                    pair_info['before_save_root_dir'],
                    pair_info['after_save_root_dir'],
                    figure_type,
                    list(range(13)),
                    center,
                    10,
                    f'Hidden_State_',
                    save_root_dir=os.path.join(figure_save_root_dir, center_str, figure_type.name.lower())
                )
    # endregion


if __name__ == '__main__':
    exp_0827()
