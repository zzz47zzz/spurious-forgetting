import os
from typing import Dict, Optional

import pandas as pd
import random
import numpy as np
import json
from training.utility import set_seed, attribute_list
from transformers import AutoTokenizer
from collections import OrderedDict
from tqdm import trange, tqdm

ATTRIBUTE_OUTPUT_ROOT_DIR = './data/processed_0720_v0730/attribute'
BIOGRAPHY_OUTPUT_ROOT_DIR = './data/processed_0720_v0730/biography'
TEMPLATE_OUTPUT_ROOT_DIR = './template/processed_0720_v0730/'
QA_OUTPUT_ROOT_DIR = './data/processed_0720_v0730/qa/'


class DataProcessor:
    @staticmethod
    def process_name_gender(first_name_count: int, middle_name_count: int, last_name_count: int, fullname_count: int,
                            previous_person_name_gender_file_path: Optional[str] = None):
        set_seed()
        # generate first name, middle name
        name_gender_dataset = pd.read_csv('./data/raw/name/name_gender_dataset.csv')
        name_list = []
        for row in range(len(name_gender_dataset)):
            name = name_gender_dataset.iloc[row]['Name']
            if not name.isalpha():
                continue
            if name in name_list:
                continue
            name_list.append(name)
            if len(set(name_list)) == first_name_count + middle_name_count:
                break
        assert len(name_list) == first_name_count + middle_name_count
        first_name_dict = {}
        middle_name_dict = {}
        for i in range(0, first_name_count):
            first_name_dict[i] = name_list[i]
        for i in range(first_name_count, first_name_count + middle_name_count):
            middle_name_dict[i - first_name_count] = name_list[i]
        assert len(set(first_name_dict.values())) == first_name_count
        assert len(set(middle_name_dict.values())) == middle_name_count
        # generate last name
        last_name_file = open('./data/raw/name/us_last_name.txt', 'r')
        all_lst_name = [line.strip() for line in last_name_file.readlines()]
        random_index = random.sample(range(0, len(all_lst_name)), last_name_count)
        last_name_dict = {}
        for i, index in enumerate(random_index):
            assert all_lst_name[index].isalpha()
            last_name_dict[i] = all_lst_name[index]
        assert len(set(last_name_dict.values())) == last_name_count
        # generate full name (need to be optimized in the future)
        fullname_index_set = set()
        while len(fullname_index_set) < fullname_count:
            first_name_index = random.randint(0, first_name_count - 1)
            middle_name_index = random.randint(0, middle_name_count - 1)
            last_name_index = random.randint(0, last_name_count - 1)
            fullname_index_set.add((first_name_index, middle_name_index, last_name_index))
        fullname_gender_dict = {}
        for i, (first_name_index, middle_name_index, last_name_index) in enumerate(fullname_index_set):
            # if previous_person_name_gender_file_path is set, gender will be reassigned
            fullname_gender_dict[i] = {
                'first_name_index': first_name_index,
                'middle_name_index': middle_name_index,
                'last_name_index': last_name_index,
                'fullname': f"{first_name_dict[first_name_index]} "
                            f"{middle_name_dict[middle_name_index]} "
                            f"{last_name_dict[last_name_index]}",
                'gender': 'male' if random.random() < 0.5 else 'female',
            }
        # region handle previous data
        if previous_person_name_gender_file_path is not None:
            print('Person name reordering...')
            previous_full_name_gender_dict = json.load(open(previous_person_name_gender_file_path, 'r'))
            reordered_full_name_gender_dict = {}
            for key in tqdm(fullname_gender_dict, desc='Phase 1'):
                del fullname_gender_dict[key]['gender']  # remove the gender information
            full_name_gender_dict_reverse = {str(value): key for key, value in fullname_gender_dict.items()}
            for key, value in tqdm(previous_full_name_gender_dict.items(), desc='Phase 2'):
                name_info = {
                    'first_name_index': value['first_name_index'],
                    'middle_name_index': value['middle_name_index'],
                    'last_name_index': value['last_name_index'],
                    'fullname': value['fullname'],
                    # without gender information
                }
                reordered_full_name_gender_dict[key] = value
                full_name_gender_dict_reverse.pop(str(name_info))
            for full_name_gender_dict_key in tqdm(full_name_gender_dict_reverse.values(), desc='Phase 3'):
                name_gender_info = fullname_gender_dict[full_name_gender_dict_key]
                name_gender_info['gender'] = 'male' if random.random() < 0.5 else 'female'
                reordered_full_name_gender_dict[len(reordered_full_name_gender_dict)] = name_gender_info
            assert len(fullname_gender_dict) == len(reordered_full_name_gender_dict)
            fullname_gender_dict = reordered_full_name_gender_dict
        # endregion
        # write data
        if not os.path.exists(os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'selected_person_name_component')):
            os.mkdir(os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'selected_person_name_component'))
        json.dump(first_name_dict,
                  open(os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'selected_person_name_component/first_name.json'), 'w'),
                  indent=4)
        json.dump(middle_name_dict,
                  open(os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'selected_person_name_component/middle_name.json'), 'w'),
                  indent=4)
        json.dump(last_name_dict,
                  open(os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'selected_person_name_component/last_name.json'), 'w'),
                  indent=4)
        json.dump(fullname_gender_dict,
                  open(os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'person_name_gender.json'), 'w'),
                  indent=4)

    @staticmethod
    def process_birthday(begin_year: int, end_year: int):
        month_index_to_name = {
            1: 'January',
            2: 'February',
            3: 'March',
            4: 'April',
            5: 'May',
            6: 'June',
            7: 'July',
            8: 'August',
            9: 'September',
            10: 'October',
            11: 'November',
            12: 'December'
        }
        birthday_dict = {}
        for year in range(begin_year, end_year):
            for month in range(1, 13):
                for day in range(1, 29):
                    birthday_dict[len(birthday_dict)] = {
                        'year': year,
                        'month': month,
                        'day': day,
                        'birthday': f'{month_index_to_name[month]} {day}, {year}'
                    }
        json.dump(birthday_dict, open(os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'birthday.json'), 'w'), indent=4)

    @staticmethod
    def process_city(count: int):
        city_dataset = pd.read_csv('./data/raw/city/List_of_United_States_cities_by_population_2.csv')
        city_dict = {}
        for i in range(1, count + 1):
            city_dict[len(city_dict)] = {
                'city': city_dataset.iloc[i]['City'],
                'state': city_dataset.iloc[i]['ST'],
                'fullname': f"{city_dataset.iloc[i]['City']}, {city_dataset.iloc[i]['ST']}"
            }
        assert len(set([city['fullname'] for city in city_dict.values()])) == count
        json.dump(city_dict, open(os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'city.json'), 'w'), indent=4)

    @staticmethod
    def process_university(count: int):
        university_dataset_list = [
            pd.read_csv('./data/raw/university/List_of_research_universities_in_the_United_States_1.csv'),
            pd.read_csv('./data/raw/university/List_of_research_universities_in_the_United_States_2.csv'),
            pd.read_csv('./data/raw/university/List_of_research_universities_in_the_United_States_3.csv'),
        ]
        university_dict = {}
        for dataset in university_dataset_list:
            for i in range(0, len(dataset)):
                university = dataset.iloc[i]['Institution']
                if not university.replace(
                        ' ', '').replace(',', '').replace('&', '').replace('\u2013', '').replace('-', '').isalpha():
                    continue
                university_dict[len(university_dict)] = university
                if len(university_dict) >= count:
                    break
            if len(university_dict) >= count:
                break
        assert len(set(university_dict.values())) == count
        json.dump(university_dict, open(os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'university.json'), 'w'), indent=4)

    @staticmethod
    def process_major(count: int):
        major_dataset = pd.read_csv('./data/raw/major/major.tsv', sep='\t')
        major_dict = {}
        for i in range(len(major_dataset)):
            major = major_dataset.iloc[i]['Program Name']
            if not major.replace(' ', '').isalpha():
                continue
            major_dict[len(major_dict)] = major
            if len(major_dict) >= count:
                break
        assert len(set(major_dict.values())) == count
        json.dump(major_dict, open(os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'major.json'), 'w'), indent=4)

    @staticmethod
    def process_company(count: int):
        company_dataset = pd.read_csv('./data/raw/company/Fortune 500 2017 - Fortune 500.csv')
        company_dict = {}
        for row in range(len(company_dataset)):
            company = company_dataset.iloc[row]['Title']
            city = company_dataset.iloc[row]['Hqcity']
            state = company_dataset.iloc[row]['Hqstate']
            company_dict[len(company_dict)] = {
                'name': company,
                'city': city,
                'state': state,
                'city_fullname': f'{city}, {state}'
            }
            if len(company_dict) >= count:
                break
        assert len(set([company['name'] for company in company_dict.values()])) == count
        json.dump(company_dict, open(os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'company.json'), 'w'), indent=4)


class TemplateProcessor:
    @staticmethod
    def process_helper(template_path: str, output_path: str, attribute_pattern: str, template_count: int):
        birthday_template = open(template_path, 'r').read()
        template_list = birthday_template.split('\n')
        template_dict = {}
        for i in range(len(template_list)):
            template = template_list[i]
            template = template.split('. ')[1].strip()
            assert template.startswith('<<PERSON_NAME>>')
            assert template.endswith('.')
            assert attribute_pattern in template
            if template in template_dict.values():
                continue
            template_dict[len(template_dict)] = template
            if len(template_dict) >= template_count:
                break
        assert len(template_dict) == template_count
        json.dump(template_dict, open(output_path, 'w'), indent=4)

    @staticmethod
    def process(template_count: int):
        if not os.path.exists(TEMPLATE_OUTPUT_ROOT_DIR):
            os.mkdir(TEMPLATE_OUTPUT_ROOT_DIR)
        for params in [
            ('./template/raw/1_birthday.txt', os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '1_birthday.json'),
             '<<BIRTHDAY>>'),
            ('./template/raw/2_birth_city.txt', os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '2_birth_city.json'),
             '<<BIRTH_CITY>>'),
            ('./template/raw/3_college.txt', os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '3_college.json'),
             '<<COLLEGE>>'),
            ('./template/raw/4_major.txt', os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '4_major.json'),
             '<<MAJOR>>'),
            ('./template/raw/5_company_name.txt', os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '5_company_name.json'),
             '<<COMPANY_NAME>>'),
            ('./template/raw/6_company_city.txt', os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '6_company_city.json'),
             '<<COMPANY_CITY>>'),
        ]:
            params = params + (template_count,)
            TemplateProcessor.process_helper(*params)
        university_template_dict = json.load(open(os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '3_college.json'), 'r'))
        for k in university_template_dict:
            assert '<<COLLEGE>>' in university_template_dict[k]
            university_template_dict[k] = university_template_dict[k].replace('<<COLLEGE>>', '<<UNIVERSITY>>')
        os.remove(os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '3_college.json'))
        json.dump(university_template_dict,
                  open(os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '3_university.json'), 'w'),
                  indent=4)


class DatasetConstructor:
    def __init__(self,
                 birthday_file_name,
                 city_file_name,
                 company_file_name,
                 major_file_name,
                 person_name_gender_file_name,
                 university_file_name,
                 birthday_template_file_name,
                 birth_city_template_file_name,
                 university_template_file_name,
                 major_template_file_name,
                 company_name_template_file_name,
                 company_city_template_file_name,
                 birthday_qa_template_file_name,
                 birth_city_qa_template_file_name,
                 university_qa_template_file_name,
                 major_qa_template_file_name,
                 company_name_qa_template_file_name,
                 company_city_qa_template_file_name,
                 tokenizer,
                 ):
        self.birthday_dict = json.load(open(birthday_file_name, 'r'))
        self.city_dict = json.load(open(city_file_name, 'r'))
        self.company_dict = json.load(open(company_file_name, 'r'))
        self.major_dict = json.load(open(major_file_name, 'r'))
        self.person_name_gender_dict = json.load(open(person_name_gender_file_name, 'r'))
        self.university_dict = json.load(open(university_file_name, 'r'))
        self.birthday_template_dict = json.load(open(birthday_template_file_name, 'r'))
        self.birth_city_template_dict = json.load(open(birth_city_template_file_name, 'r'))
        self.university_template_dict = json.load(open(university_template_file_name, 'r'))
        self.major_template_dict = json.load(open(major_template_file_name, 'r'))
        self.company_name_template_dict = json.load(open(company_name_template_file_name, 'r'))
        self.company_city_template_dict = json.load(open(company_city_template_file_name, 'r'))
        self.birthday_qa_template = open(birthday_qa_template_file_name, 'r').read()
        self.birth_city_qa_template = open(birth_city_qa_template_file_name, 'r').read()
        self.university_qa_template = open(university_qa_template_file_name, 'r').read()
        self.major_qa_template = open(major_qa_template_file_name, 'r').read()
        self.company_name_qa_template = open(company_name_qa_template_file_name, 'r').read()
        self.company_city_qa_template = open(company_city_qa_template_file_name, 'r').read()
        self.index_file_name = os.path.join(BIOGRAPHY_OUTPUT_ROOT_DIR, 'index.json')
        if os.path.exists(self.index_file_name):
            self.index_file = json.load(open(self.index_file_name, 'r'))
        else:
            self.index_file = {}
        self.tokenizer = tokenizer

    def construct_biography_index(self, previous_index_file_path=None):
        set_seed()
        biography_index_dict = {}
        for i in trange(len(self.person_name_gender_dict)):
            birthday_index = random.randint(0, len(self.birthday_dict) - 1)
            birth_city_index = random.randint(0, len(self.city_dict) - 1)
            university_index = random.randint(0, len(self.university_dict) - 1)
            major_index = random.randint(0, len(self.major_dict) - 1)
            company_index = random.randint(0, len(self.company_dict) - 1)
            distinct_biography_entries_count = 5  # the maximum number of distinct biography entries for each person
            birthday_template_index_list = random.sample(range(0, len(self.birthday_template_dict)),
                                                         distinct_biography_entries_count)
            birth_city_template_index_list = random.sample(range(0, len(self.birth_city_template_dict)),
                                                           distinct_biography_entries_count)
            university_template_index_list = random.sample(range(0, len(self.university_template_dict)),
                                                           distinct_biography_entries_count)
            major_template_index_list = random.sample(range(0, len(self.major_template_dict)),
                                                      distinct_biography_entries_count)
            company_name_template_index_list = random.sample(range(0, len(self.company_name_template_dict)),
                                                             distinct_biography_entries_count)
            company_city_template_index_list = random.sample(range(0, len(self.company_city_template_dict)),
                                                             distinct_biography_entries_count)
            template_order_list = []
            for j in range(distinct_biography_entries_count):
                order = random.sample(attribute_list, 6)
                while order in template_order_list:
                    order = random.sample(attribute_list, 6)
                template_order_list.append(order)
            template_index_dict = {}
            for j in range(distinct_biography_entries_count):
                template_index_dict[str(j)] = {
                    'birthday': str(birthday_template_index_list[j]),
                    'birth_city': str(birth_city_template_index_list[j]),
                    'university': str(university_template_index_list[j]),
                    'major': str(major_template_index_list[j]),
                    'company_name': str(company_name_template_index_list[j]),
                    'company_city': str(company_city_template_index_list[j]),
                    'order': template_order_list[j],
                }
            biography_index_dict[str(i)] = {
                'person_name': str(i),
                'birthday': str(birthday_index),
                'birth_city': str(birth_city_index),
                'university': str(university_index),
                'major': str(major_index),
                'company': str(company_index),
                'template_index_dict': template_index_dict,
            }
        # region check whether the previous index is a subset of the current index
        if previous_index_file_path is not None:
            previous_index_dict = json.load(open(previous_index_file_path, 'r'))
            for key in previous_index_dict:
                assert previous_index_dict[key] == biography_index_dict[key]
        # endregion
        if os.path.exists(self.index_file_name):
            old_index_dict = json.load(open(self.index_file_name, 'r'))
            assert old_index_dict == biography_index_dict
        else:
            json.dump(biography_index_dict, open(self.index_file_name, 'w'), indent=4)
        self.index_file = biography_index_dict

    def get_person_info(self, person_index: str):
        return {
            'person_name': self.person_name_gender_dict[self.index_file[person_index]['person_name']]['fullname'],
            'birthday': self.birthday_dict[self.index_file[person_index]['birthday']]['birthday'],
            'birth_city': self.city_dict[self.index_file[person_index]['birth_city']]['fullname'],
            'university': self.university_dict[self.index_file[person_index]['university']],
            'major': self.major_dict[self.index_file[person_index]['major']],
            'company_name': self.company_dict[self.index_file[person_index]['company']]['name'],
            'company_city': self.company_dict[self.index_file[person_index]['company']]['city_fullname'],
        }

    def construct_biography_entry(self, person_index: str, template_index: str, permute: bool, fullname: bool):
        template_dict = {
            'birthday': self.birthday_template_dict[
                self.index_file[person_index]['template_index_dict'][template_index]['birthday']],
            'birth_city': self.birth_city_template_dict[
                self.index_file[person_index]['template_index_dict'][template_index]['birth_city']],
            'university': self.university_template_dict[
                self.index_file[person_index]['template_index_dict'][template_index]['university']],
            'major': self.major_template_dict[
                self.index_file[person_index]['template_index_dict'][template_index]['major']],
            'company_name': self.company_name_template_dict[
                self.index_file[person_index]['template_index_dict'][template_index]['company_name']],
            'company_city': self.company_city_template_dict[
                self.index_file[person_index]['template_index_dict'][template_index]['company_city']],
        }
        person_info = self.get_person_info(person_index)
        gender = self.person_name_gender_dict[self.index_file[person_index]['person_name']]['gender']
        if gender == 'female':
            for key in template_dict:
                template_dict[key] = template_dict[key].replace('his', 'her').replace('him', 'her')
        current_biography = ' '
        token_info = {}
        permuted_attribute_list = []
        if permute:
            order = self.index_file[person_index]['template_index_dict'][template_index]['order']
            for attribute in order:
                permuted_attribute_list.append(attribute)
        else:
            permuted_attribute_list = attribute_list
        for attribute_name in permuted_attribute_list:
            template: str = template_dict[attribute_name]
            assert template.startswith('<<PERSON_NAME>>')
            if fullname or attribute_name == permuted_attribute_list[0]:
                template = template.replace('<<PERSON_NAME>>', person_info['person_name'])
            else:
                if gender == 'female':
                    if template.startswith("<<PERSON_NAME>>'s"):
                        template = template.replace("<<PERSON_NAME>>'s", 'Her')
                    else:
                        assert template.startswith("<<PERSON_NAME>> ")
                        template = template.replace("<<PERSON_NAME>>", 'She')
                else:
                    if template.startswith("<<PERSON_NAME>>'s"):
                        template = template.replace("<<PERSON_NAME>>'s", 'His')
                    else:
                        assert template.startswith("<<PERSON_NAME>> ")
                        template = template.replace("<<PERSON_NAME>>", 'He')
            current_biography = current_biography + template + ' '
            pattern = '<<' + attribute_name.upper() + '>>'
            prefix, _ = current_biography.split(' ' + pattern)
            prefix_token_list = self.tokenizer(prefix)['input_ids']
            first_attribute_token_position = len(prefix_token_list)
            first_attribute_token = self.tokenizer(' ' + person_info[attribute_name])['input_ids'][0]
            current_biography = current_biography.replace(pattern, person_info[attribute_name])
            # check whether the tokenization result is correct
            current_biography_token_list = self.tokenizer(current_biography)['input_ids']
            assert current_biography_token_list[:first_attribute_token_position] == prefix_token_list
            assert current_biography_token_list[first_attribute_token_position] == first_attribute_token
            # write token info
            token_info[attribute_name] = {
                'first_token_position': first_attribute_token_position,
                'first_token': first_attribute_token,
            }
        current_biography = current_biography[:-1]  # remove the last space
        return {
            'biography': current_biography,
            'token_info': token_info,
            'tokenizer': self.tokenizer.__class__.__name__,
        }

    def construct_biography(self, permute: bool, fullname: bool, multi: int, output_file_name: str):
        # single is multi1
        result_dict = {}
        for person_index in trange(len(self.index_file), desc=output_file_name):
            for template_index in range(multi):
                biography = self.construct_biography_entry(str(person_index), str(template_index), permute, fullname)
                result_dict[str(person_index) + '_' + str(template_index)] = biography
        output_path = os.path.join(BIOGRAPHY_OUTPUT_ROOT_DIR, output_file_name)
        if os.path.exists(output_path):
            old_result_dict = json.load(open(output_path, 'r'))
            assert old_result_dict == result_dict
        else:
            json.dump(result_dict, open(output_path, 'w'), indent=4)

    def construct_qa(self, train_ratio: float = None):
        if train_ratio is None:
            train_qa_count = len(self.index_file)
        else:
            train_qa_count = int(len(self.index_file) * train_ratio)
        train_dataset = {}
        test_dataset = {}
        for person_index in tqdm(self.index_file):
            qa_dict = {}
            for pattern, template in [
                ('<<BIRTHDAY>>', self.birthday_qa_template),
                ('<<BIRTH_CITY>>', self.birth_city_qa_template),
                ('<<UNIVERSITY>>', self.university_qa_template),
                ('<<MAJOR>>', self.major_qa_template),
                ('<<COMPANY_NAME>>', self.company_name_qa_template),
                ('<<COMPANY_CITY>>', self.company_city_qa_template),
            ]:
                person_info = self.get_person_info(person_index)
                assert template[-1] == '>', 'the answer should not end with .'
                answer = ' ' + pattern
                prompt = template[: -len(answer)]
                info_key = pattern[2: -2].lower()
                answer = answer.replace(pattern, person_info[info_key])
                assert answer[0] == ' ', 'the answer should start with space'
                prompt = prompt.replace('<<PERSON_NAME>>', person_info['person_name'])
                assert prompt[-1] == ':'
                prompt_token = self.tokenizer(prompt)['input_ids']
                answer_token = self.tokenizer(answer)['input_ids']
                sequence_token = self.tokenizer(prompt + answer)['input_ids']
                assert sequence_token == prompt_token + answer_token
                qa_dict[info_key] = {
                    'prompt': prompt,
                    'answer': answer
                }
            if int(person_index) < train_qa_count:
                train_dataset[int(person_index)] = qa_dict
            else:
                test_dataset[int(person_index)] = qa_dict
        if not os.path.exists(QA_OUTPUT_ROOT_DIR):
            os.mkdir(QA_OUTPUT_ROOT_DIR)
        if train_ratio is None:
            json.dump(train_dataset, open(os.path.join(QA_OUTPUT_ROOT_DIR, 'all.json'), 'w'), indent=4)
        else:
            json.dump(train_dataset, open(os.path.join(QA_OUTPUT_ROOT_DIR, 'train.json'), 'w'), indent=4)
            json.dump(test_dataset, open(os.path.join(QA_OUTPUT_ROOT_DIR, 'test.json'), 'w'), indent=4)


def main():
    # previous_person_name_gender_file_path = './data/processed_0720/attribute/person_name_gender.json'
    # DataProcessor.process_name_gender(400, 400, 1000, 200000,
    #                                   previous_person_name_gender_file_path=previous_person_name_gender_file_path)
    # DataProcessor.process_birthday(1900, 2100)
    # DataProcessor.process_city(200)
    # DataProcessor.process_university(300)
    # DataProcessor.process_major(100)
    # DataProcessor.process_company(263)
    # TemplateProcessor.process(50)
    tokenizer = AutoTokenizer.from_pretrained(
        "./model/gpt-neox/pythia-160m",
        model_max_length=512,  # unimportant config
        padding_side="right",
        use_fast=True,
    )
    constructor = DatasetConstructor(
        birthday_file_name=os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'birthday.json'),
        city_file_name=os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'city.json'),
        company_file_name=os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'company.json'),
        major_file_name=os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'major.json'),
        person_name_gender_file_name=os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'person_name_gender.json'),
        university_file_name=os.path.join(ATTRIBUTE_OUTPUT_ROOT_DIR, 'university.json'),
        birthday_template_file_name=os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '1_birthday.json'),
        birth_city_template_file_name=os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '2_birth_city.json'),
        university_template_file_name=os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '3_university.json'),
        major_template_file_name=os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '4_major.json'),
        company_name_template_file_name=os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '5_company_name.json'),
        company_city_template_file_name=os.path.join(TEMPLATE_OUTPUT_ROOT_DIR, '6_company_city.json'),
        birthday_qa_template_file_name='./template/qa_template/1_birthday.txt',
        birth_city_qa_template_file_name='./template/qa_template/2_birth_city.txt',
        university_qa_template_file_name='./template/qa_template/3_university.txt',
        major_qa_template_file_name='./template/qa_template/4_major.txt',
        company_name_qa_template_file_name='./template/qa_template/5_company_name.txt',
        company_city_qa_template_file_name='./template/qa_template/6_company_city.txt',
        tokenizer=tokenizer,
    )
    # constructor.construct_biography_index(previous_index_file_path='./data/processed_0720/biography/index.json')
    # constructor.construct_biography(False, False, 1, 'single.json')
    # constructor.construct_biography(False, True, 1, 'single_fullname.json')
    # constructor.construct_biography(True, False, 1, 'single_permute.json')
    # constructor.construct_biography(True, True, 1, 'single_permute_fullname.json')
    # constructor.construct_biography(False, False, 5, 'multi5.json')
    # constructor.construct_biography(False, True, 5, 'multi5_fullname.json')
    # constructor.construct_biography(True, False, 5, 'multi5_permute.json')
    # constructor.construct_biography(True, True, 5, 'multi5_permute_fullname.json')
    # constructor.construct_qa(0.5)  # for processed_0720
    constructor.construct_qa()  # for processed_0720_v0730
    pass


if __name__ == '__main__':
    main()
