from sympy.polys.polyconfig import query

from training.data_module import QADataset, filter_qa_data_with_token_info, IGNORE_TOKEN_ID
from training.utility import FineTuningTrainPersonIndexInfoList, QARawData, attribute_list
from torch.utils.data import DataLoader
import json
from transformers import AutoTokenizer
import os
import torch
from tqdm import tqdm


def collate_fn_for_fine_tuning_qa_dataset(batch):
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    not_label = (labels == IGNORE_TOKEN_ID)
    query_mask = not_label & attention_mask
    query_length = query_mask.sum(dim=1)
    query_end_index = query_length - 1
    return {
        'attention_mask': attention_mask,
        'input_ids': input_ids,
        # do not return labels, since it is useless
    }, query_end_index


class ResidualStreamQADataset(QADataset):
    def __init__(self, tokenizer, raw_data, model_max_length):
        super().__init__(tokenizer, raw_data, model_max_length)
        for person_index in tqdm(raw_data, 'Checking the order of attribute'):
            _attribute_list = raw_data[person_index].keys()
            assert attribute_list == list(_attribute_list)
        self.attribute_list = torch.arange(len(attribute_list)).repeat(len(raw_data)).split(1, dim=0)
        all_attention_mask = torch.stack(self.attention_mask_list)
        all_label = torch.stack(self.label_list)
        not_all_label = (all_label == IGNORE_TOKEN_ID)
        all_query_mask = not_all_label & all_attention_mask
        all_query_length = all_query_mask.sum(dim=1)
        all_query_end_index = all_query_length - 1
        self.query_end_index_list = list(torch.split(all_query_end_index, 1, dim=0))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['query_end_indexes'] = self.query_end_index_list[index]
        item['attributes'] = self.attribute_list[index]
        item.pop('labels')
        return item

    def __len__(self):
        return super().__len__()


def qa_dataset_data_loader_validation():
    project_root = '/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting'
    model_path = os.path.join(project_root, 'model/gpt-neox/v_0720/multi5_permute_fullname/task_0/final_model')
    data_path = os.path.join(project_root, 'data/processed_0720_v0730/qa/all.json')
    person_index_info_list: FineTuningTrainPersonIndexInfoList = [{'start': 0, 'end': 500}]
    raw_data: QARawData = json.load(open(data_path))
    filtered_data = filter_qa_data_with_token_info(raw_data, person_index_info_list)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=32, padding_side="right", use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    qa_dataset = QADataset(tokenizer, filtered_data, tokenizer.model_max_length)
    data_loader = DataLoader(qa_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_for_fine_tuning_qa_dataset)
    batch = next(iter(data_loader))
    print(batch)


def residual_stram_qa_dataset_validation():
    project_root = '/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting'
    model_path = os.path.join(project_root, 'model/gpt-neox/v_0720/multi5_permute_fullname/task_0/final_model')
    data_path = os.path.join(project_root, 'data/processed_0720_v0730/qa/all.json')
    person_index_info_list: FineTuningTrainPersonIndexInfoList = [{'start': 0, 'end': 500}]
    raw_data: QARawData = json.load(open(data_path))
    filtered_data = filter_qa_data_with_token_info(raw_data, person_index_info_list)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=32, padding_side="right", use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    residual_stram_qa_dataset = ResidualStreamQADataset(tokenizer, filtered_data, tokenizer.model_max_length)
    item = residual_stram_qa_dataset[0]
    print(item)


if __name__ == '__main__':
    residual_stram_qa_dataset_validation()
