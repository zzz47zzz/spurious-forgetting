import json
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import pipeline, set_seed, GPTNeoXForCausalLM
from training.data_module import QADataset, filter_qa_data_with_token_info
from training.utility import FineTuningTrainPersonIndexInfoList, QARawData
from torch.utils.data import DataLoader


def get_model_logits():
    model_path = "/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting/model/gpt2/origin"

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2Model.from_pretrained(model_path)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)


def generate_text_by_pipeline():
    # model_path = "/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting/model/gpt-neox/v_0720/multi5_permute_fullname/task_0/final_model"
    model_path = "/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting/model/gpt-neox/v_0720/multi5_permute_fullname/task_0_no_use_parallel_residual/final_model"
    generator = pipeline(task='text-generation', model=model_path)
    set_seed(42)

    def biography_generator(text): return generator(text, do_sample=False, max_length=512)

    print()


def generate_text_by_model():
    model_path = "/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting/model/gpt-neox/v_0720/multi5_permute_fullname/task_0/final_model"
    model = GPTNeoXForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    tokenizer.pad_token = tokenizer.unk_token
    model.eval()
    sentence_list = ["What is the birth date of Wesley Alexander Emley?\nAnswer:",
                     "Which university did Wesley Alexander Emley study?\nAnswer:"]
    input_ids = tokenizer(sentence_list, return_tensors='pt', padding=True)['input_ids']
    attention_mask = tokenizer(sentence_list, return_tensors='pt', padding=True)['attention_mask']
    generate_result = model.generate(input_ids, max_length=512, do_sample=False, attention_mask=attention_mask)
    print(tokenizer.batch_decode(generate_result))
    print()


def check_pretrain_model():
    model_path = "/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting/model/gpt-neox/v_0720/multi5_permute_fullname/task_0/final_model"
    template_path = "/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting/template/processed_0720_v0730/1_birthday.json"
    pattern = '<<BIRTHDAY>>'
    person_name_list = ['Curtis Chase Emley', 'Martha Claudia Deyarmond']
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right', model_max_length=512)
    generator = pipeline(task='text-generation', model=model_path)
    set_seed(42)

    def biography_generator(text):
        return generator(text, do_sample=False, max_length=512, stop_strings=['.'], tokenizer=tokenizer)

    person_name_to_result_count_dict = {}
    for person_name in person_name_list:
        template_dict = json.load(open(template_path, 'r'))
        result_count_dict = {}
        for k, v in template_dict.items():
            assert pattern == '<<BIRTHDAY>>'
            prompt = ' ' + v.replace('<<PERSON_NAME>>', person_name)
            prompt = prompt[: -len(' <<BIRTHDAY>>.')]
            generation = biography_generator(prompt)
            pure_generation = generation[0]['generated_text'][len(prompt):]
            print(pure_generation)
            result_count_dict.setdefault(pure_generation, 0)
            result_count_dict[pure_generation] += 1
        print('=============================')
        person_name_to_result_count_dict[person_name] = result_count_dict
    print()


def probing_exploration():
    person_index_info_list: FineTuningTrainPersonIndexInfoList = [
        {'start': 0, 'end': 500}
    ]
    model_path = "/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting/model/gpt-neox/v_0720/multi5_permute_fullname/task_0/final_model"
    data_path = '/dev_data/cxd/physics-of-forgetting-in-llm/physics_of_forgetting/data/processed_0720_v0730/qa/all.json'
    raw_data: QARawData = json.load(open(data_path))
    filtered_data = filter_qa_data_with_token_info(raw_data, person_index_info_list)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=32, padding_side="right", use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    model = GPTNeoXForCausalLM.from_pretrained(model_path)
    qa_dataset = QADataset(tokenizer, filtered_data, tokenizer.model_max_length)
    data_loader = DataLoader(qa_dataset, batch_size=8, shuffle=False)
    batch = next(iter(data_loader))
    batch['output_hidden_states'] = True
    output = model(**batch)
    print()


if __name__ == '__main__':
    get_model_logits()
