from transformers import AutoTokenizer
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from training.utility import attribute_list
import json
from typing import Dict
from tqdm import tqdm, trange

TOTAL = 100
BATCH_SIZE = 48


def sequence_forward(model: GPTNeoXForCausalLM,
                     tokenizer: GPTNeoXTokenizerFast,
                     qa_data_dict: Dict[str, Dict[str, Dict[str, str]]],
                     log_prefix: str):
    log = log_prefix
    for attribute in attribute_list:
        correct, total = 0, 0
        for person_index in tqdm(qa_data_dict.keys()):
            total += 1
            qa_info = qa_data_dict[person_index][attribute]
            prompt, answer = qa_info['prompt'], qa_info['answer']
            tokenizer_result = tokenizer(prompt, return_tensors='pt')
            input_ids, attention_mask = tokenizer_result['input_ids'], tokenizer_result['attention_mask']
            generate_result = model.generate(input_ids, max_length=512, do_sample=False, attention_mask=attention_mask)
            generate_result = tokenizer.batch_decode(generate_result)[0]
            if not generate_result.endswith(tokenizer.eos_token):
                continue
            generate_answer = generate_result[len(prompt):-len(tokenizer.eos_token)]
            if generate_answer == answer:
                correct += 1
            if total == TOTAL:
                break
        log += f"Attribute: {attribute:>12}, correct: {correct:>5}, total: {total:>5}, accuracy: {correct / total:>5}\n"
    print(log)


def batch_forward(model: GPTNeoXForCausalLM,
                  tokenizer: GPTNeoXTokenizerFast,
                  qa_data_dict: Dict[str, Dict[str, Dict[str, str]]],
                  log_prefix: str):
    log = log_prefix
    for attribute in attribute_list:
        correct, total = 0, 0
        prompt_list, answer_list = [], []
        for person_index in qa_data_dict.keys():
            qa_info = qa_data_dict[person_index][attribute]
            prompt_list.append(qa_info['prompt'])
            answer_list.append(qa_info['answer'])
            total += 1
            if total == TOTAL:
                break
        for person_index in trange(0, TOTAL, BATCH_SIZE):
            batch_prompt_list = prompt_list[person_index: person_index + BATCH_SIZE]
            batch_answer_list = answer_list[person_index: person_index + BATCH_SIZE]
            tokenizer_result = tokenizer(batch_prompt_list, return_tensors='pt', padding=True)
            input_ids, attention_mask = tokenizer_result['input_ids'], tokenizer_result['attention_mask']
            generate_result = model.generate(input_ids, max_length=512, do_sample=False, attention_mask=attention_mask)
            generate_result = tokenizer.batch_decode(generate_result)
            for i in range(len(batch_prompt_list)):
                left_padding_count = (attention_mask[i].shape[0] - attention_mask[i].sum()).item()
                if not generate_result[i].endswith(tokenizer.eos_token):
                    continue
                generate_answer = generate_result[i][
                                  left_padding_count * len(tokenizer.eos_token) + len(batch_prompt_list[i]):
                                  -len(tokenizer.eos_token)]
                if generate_answer.startswith(batch_answer_list[i]):
                    correct += 1
        log += f"Attribute: {attribute:>12}, correct: {correct:>5}, total: {total:>5}, accuracy: {correct / total:>5}\n"
    print(log)


def main():
    model_path = "./model/gpt-neox/single_fullname/full_fine_tuning_save_step_1250/lr_5e-06_wd_0.01/final_model"
    train_qa_data_dict = json.load(open("./data/processed_0703/qa/train.json", 'r'))
    test_qa_data_dict = json.load(open("./data/processed_0703/qa/test.json", 'r'))
    model: GPTNeoXForCausalLM = GPTNeoXForCausalLM.from_pretrained(model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    sequence_forward(model, tokenizer, train_qa_data_dict, 'sequence_forward - train:\n')
    sequence_forward(model, tokenizer, test_qa_data_dict, 'sequence_forward - test:\n')
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    tokenizer.pad_token = tokenizer.unk_token
    batch_forward(model, tokenizer, train_qa_data_dict, 'batch_forward - train:\n')
    batch_forward(model, tokenizer, test_qa_data_dict, 'batch_forward - test:\n')


if __name__ == '__main__':
    main()

"""
BATCH_SIZE = 48

==============================================================================

sequence_forward - train:
Attribute:     birthday, correct:   100, total:   100, accuracy:   1.0
Attribute:   birth_city, correct:   100, total:   100, accuracy:   1.0
Attribute:   university, correct:   100, total:   100, accuracy:   1.0
Attribute:        major, correct:   100, total:   100, accuracy:   1.0
Attribute: company_name, correct:   100, total:   100, accuracy:   1.0
Attribute: company_city, correct:    11, total:   100, accuracy:  0.11
sequence_forward - test:
Attribute:     birthday, correct:    54, total:   100, accuracy:  0.54
Attribute:   birth_city, correct:    33, total:   100, accuracy:  0.33
Attribute:   university, correct:    58, total:   100, accuracy:  0.58
Attribute:        major, correct:    39, total:   100, accuracy:  0.39
Attribute: company_name, correct:    38, total:   100, accuracy:  0.38
Attribute: company_city, correct:     7, total:   100, accuracy:  0.07

batch_forward - train:
Attribute:     birthday, correct:   100, total:   100, accuracy:   1.0
Attribute:   birth_city, correct:   100, total:   100, accuracy:   1.0
Attribute:   university, correct:   100, total:   100, accuracy:   1.0
Attribute:        major, correct:   100, total:   100, accuracy:   1.0
Attribute: company_name, correct:   100, total:   100, accuracy:   1.0
Attribute: company_city, correct:    11, total:   100, accuracy:  0.11
batch_forward - test:
Attribute:     birthday, correct:    54, total:   100, accuracy:  0.54
Attribute:   birth_city, correct:    33, total:   100, accuracy:  0.33
Attribute:   university, correct:    58, total:   100, accuracy:  0.58
Attribute:        major, correct:    40, total:   100, accuracy:   0.4
Attribute: company_name, correct:    38, total:   100, accuracy:  0.38
Attribute: company_city, correct:     7, total:   100, accuracy:  0.07

==============================================================================

sequence_forward - train:
Attribute:     birthday, correct:    61, total:   100, accuracy:  0.61
Attribute:   birth_city, correct:    34, total:   100, accuracy:  0.34
Attribute:   university, correct:    53, total:   100, accuracy:  0.53
Attribute:        major, correct:    44, total:   100, accuracy:  0.44
Attribute: company_name, correct:    37, total:   100, accuracy:  0.37
Attribute: company_city, correct:    11, total:   100, accuracy:  0.11
sequence_forward - test:
Attribute:     birthday, correct:    42, total:   100, accuracy:  0.42
Attribute:   birth_city, correct:    25, total:   100, accuracy:  0.25
Attribute:   university, correct:    46, total:   100, accuracy:  0.46
Attribute:        major, correct:    38, total:   100, accuracy:  0.38
Attribute: company_name, correct:    31, total:   100, accuracy:  0.31
Attribute: company_city, correct:     8, total:   100, accuracy:  0.08

batch_forward - train:
Attribute:     birthday, correct:    61, total:   100, accuracy:  0.61
Attribute:   birth_city, correct:    34, total:   100, accuracy:  0.34
Attribute:   university, correct:    54, total:   100, accuracy:  0.54
Attribute:        major, correct:    45, total:   100, accuracy:  0.45
Attribute: company_name, correct:    37, total:   100, accuracy:  0.37
Attribute: company_city, correct:    11, total:   100, accuracy:  0.11
batch_forward - test:
Attribute:     birthday, correct:    42, total:   100, accuracy:  0.42
Attribute:   birth_city, correct:    25, total:   100, accuracy:  0.25
Attribute:   university, correct:    46, total:   100, accuracy:  0.46
Attribute:        major, correct:    39, total:   100, accuracy:  0.39
Attribute: company_name, correct:    31, total:   100, accuracy:  0.31
Attribute: company_city, correct:     8, total:   100, accuracy:  0.08

==============================================================================
"""