import os
from typing import Dict

import pandas as pd
import random
import numpy as np
import json
from training.utility import set_seed
from transformers import AutoTokenizer
from collections import OrderedDict
from tqdm import trange, tqdm


class DataProcessor:
    @staticmethod
    def process_name_gender(first_name_count: int, middle_name_count: int, last_name_count: int, full_name_count: int):
        set_seed()
        # generate first name, middle name
        name_gender_dataset = pd.read_csv('./data/raw/name/name_gender_dataset.csv')
        name_set = set()
        for row in range(len(name_gender_dataset)):
            if not name_gender_dataset.iloc[row]['Name'].isalpha():
                continue
            name_set.add(name_gender_dataset.iloc[row]['Name'])
            if len(name_set) == first_name_count + middle_name_count:
                break
        assert len(name_set) == first_name_count + middle_name_count
        name_list = list(name_set)
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
        # generate full name
        full_name_index_set = set()
        while len(full_name_index_set) < full_name_count:
            first_name_index = random.randint(0, first_name_count - 1)
            middle_name_index = random.randint(0, middle_name_count - 1)
            last_name_index = random.randint(0, last_name_count - 1)
            full_name_index_set.add((first_name_index, middle_name_index, last_name_index))
        full_name_gender_dict = {}
        for i, (first_name_index, middle_name_index, last_name_index) in enumerate(full_name_index_set):
            full_name_gender_dict[i] = {
                'first_name_index': first_name_index,
                'middle_name_index': middle_name_index,
                'last_name_index': last_name_index,
                'full_name': f"{first_name_dict[first_name_index]} "
                             f"{middle_name_dict[middle_name_index]} "
                             f"{last_name_dict[last_name_index]}",
                'gender': 'male' if random.random() < 0.5 else 'female',
            }
        # write data
        if not os.path.exists('./data/processed_0703/attribute/selected_person_name_component'):
            os.mkdir('./data/processed_0703/attribute/selected_person_name_component')
        json.dump(first_name_dict,
                  open('./data/processed_0703/attribute/selected_person_name_component/first_name.json', 'w'),
                  indent=4)
        json.dump(middle_name_dict,
                  open('./data/processed_0703/attribute/selected_person_name_component/middle_name.json', 'w'),
                  indent=4)
        json.dump(last_name_dict,
                  open('./data/processed_0703/attribute/selected_person_name_component/last_name.json', 'w'),
                  indent=4)
        json.dump(full_name_gender_dict,
                  open('./data/processed_0703/attribute/person_name_gender.json', 'w'),
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
        json.dump(birthday_dict, open('./data/processed_0703/attribute/birthday.json', 'w'), indent=4)

    @staticmethod
    def process_city(count: int):
        city_dataset = pd.read_csv('./data/raw/city/List_of_United_States_cities_by_population_2.csv')
        city_dict = {}
        for i in range(1, count + 1):
            city_dict[len(city_dict)] = {
                'city': city_dataset.iloc[i]['City'],
                'state': city_dataset.iloc[i]['ST'],
                'full_name': f"{city_dataset.iloc[i]['City']}, {city_dataset.iloc[i]['ST']}"
            }
        assert len(set([city['full_name'] for city in city_dict.values()])) == count
        json.dump(city_dict, open('./data/processed_0703/attribute/city.json', 'w'), indent=4)

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
        json.dump(university_dict, open('./data/processed_0703/attribute/university.json', 'w'), indent=4)

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
        json.dump(major_dict, open('./data/processed_0703/attribute/major.json', 'w'), indent=4)

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
        json.dump(company_dict, open('./data/processed_0703/attribute/company.json', 'w'), indent=4)


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
        if not os.path.exists('./template/processed_0703'):
            os.mkdir('./template/processed_0703')
        for params in [
            ('./template/raw/1_birthday.txt', './template/processed_0703/1_birthday.json', '<<BIRTHDAY>>'),
            ('./template/raw/2_birth_city.txt', './template/processed_0703/2_birth_city.json', '<<BIRTH_CITY>>'),
            ('./template/raw/3_college.txt', './template/processed_0703/3_college.json', '<<COLLEGE>>'),
            ('./template/raw/4_major.txt', './template/processed_0703/4_major.json', '<<MAJOR>>'),
            ('./template/raw/5_company_name.txt', './template/processed_0703/5_company_name.json', '<<COMPANY_NAME>>'),
            ('./template/raw/6_company_city.txt', './template/processed_0703/6_company_city.json', '<<COMPANY_CITY>>'),
        ]:
            params = params + (template_count,)
            TemplateProcessor.process_helper(*params)
        university_template_dict = json.load(open('./template/processed_0703/3_college.json', 'r'))
        for k in university_template_dict:
            assert '<<COLLEGE>>' in university_template_dict[k]
            university_template_dict[k] = university_template_dict[k].replace('<<COLLEGE>>', '<<UNIVERSITY>>')
        os.remove('./template/processed_0703/3_college.json')
        json.dump(university_template_dict, open('./template/processed_0703/3_university.json', 'w'), indent=4)


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
        self.index_file_name = './data/processed_0703/biography/index.json'
        if os.path.exists(self.index_file_name):
            self.index_file = json.load(open(self.index_file_name, 'r'))
        else:
            self.index_file = None
        self.pattern_to_info_key = {
            '<<PERSON_NAME>>': 'person_name',
            '<<BIRTHDAY>>': 'birthday',
            '<<BIRTH_CITY>>': 'birth_city',
            '<<UNIVERSITY>>': 'university',
            '<<MAJOR>>': 'major',
            '<<COMPANY_NAME>>': 'company_name',
            '<<COMPANY_CITY>>': 'company_city',
        }
        self.info_key_to_pattern = {v: k for k, v in self.pattern_to_info_key.items()}
        self.tokenizer = tokenizer

    def construct_biography_index(self):
        set_seed()
        biography_index_dict = {}
        for i in range(len(self.person_name_gender_dict)):
            birthday_index = random.randint(0, len(self.birthday_dict) - 1)
            birth_city_index = random.randint(0, len(self.city_dict) - 1)
            university_index = random.randint(0, len(self.university_dict) - 1)
            major_index = random.randint(0, len(self.major_dict) - 1)
            company_index = random.randint(0, len(self.company_dict) - 1)
            birthday_template_index = random.randint(0, len(self.birthday_template_dict) - 1)
            birth_city_template_index = random.randint(0, len(self.birth_city_template_dict) - 1)
            university_template_index = random.randint(0, len(self.university_template_dict) - 1)
            major_template_index = random.randint(0, len(self.major_template_dict) - 1)
            company_name_template_index = random.randint(0, len(self.company_name_template_dict) - 1)
            company_city_template_index = random.randint(0, len(self.company_city_template_dict) - 1)
            biography_index_dict[str(i)] = {
                'person_name_index': i,
                'birthday_index': birthday_index,
                'birth_city_index': birth_city_index,
                'university_index': university_index,
                'major_index': major_index,
                'company_index': company_index,
                'birthday_template_index': birthday_template_index,
                'birth_city_template_index': birth_city_template_index,
                'university_template_index': university_template_index,
                'major_template_index': major_template_index,
                'company_name_template_index': company_name_template_index,
                'company_city_template_index': company_city_template_index,
            }
        if os.path.exists(self.index_file_name):
            old_index_dict = json.load(open(self.index_file_name, 'r'))
            assert old_index_dict == biography_index_dict
        else:
            json.dump(biography_index_dict, open(self.index_file_name, 'w'), indent=4)
        self.index_file = biography_index_dict

    def extract_person_info(self, person_index: str) -> Dict:
        return {
            'person_name':
                self.person_name_gender_dict[str(self.index_file[person_index]['person_name_index'])]['full_name'],
            'birthday': self.birthday_dict[str(self.index_file[person_index]['birthday_index'])]['birthday'],
            'birth_city': self.city_dict[str(self.index_file[person_index]['birth_city_index'])]['full_name'],
            'university': self.university_dict[str(self.index_file[person_index]['university_index'])],
            'major': self.major_dict[str(self.index_file[person_index]['major_index'])],
            'company_name': self.company_dict[str(self.index_file[person_index]['company_index'])]['name'],
            'company_city': self.company_dict[str(self.index_file[person_index]['company_index'])]['city_fullname'],
        }

    def extract_biography_template(self, person_index: str) -> OrderedDict:
        return OrderedDict({
            'birthday': self.birthday_template_dict[str(self.index_file[person_index]['birthday_template_index'])],
            'birth_city': self.birth_city_template_dict[
                str(self.index_file[person_index]['birth_city_template_index'])],
            'university': self.university_template_dict[
                str(self.index_file[person_index]['university_template_index'])],
            'major': self.major_template_dict[str(self.index_file[person_index]['major_template_index'])],
            'company_name':
                self.company_name_template_dict[str(self.index_file[person_index]['company_name_template_index'])],
            'company_city':
                self.company_city_template_dict[str(self.index_file[person_index]['company_city_template_index'])],
        })

    @staticmethod
    def gender_to_female(sentence: str) -> str:
        return sentence.replace('his', 'her').replace('him', 'her')

    # def construct_biography_fullname(self):
    #     result_dict = {}
    #     for k in range(len(self.index_file)):
    #         k = str(k)
    #         biography = ' '.join(self.extract_biography_template(k).values())
    #         gender = self.person_name_gender_dict[str(self.index_file[k]['person_name_index'])]['gender']
    #         if gender == 'female':
    #             biography = self.gender_to_female(biography)
    #         person_info = self.extract_person_info(k)
    #         for pattern in self.pattern_to_info_key:
    #             assert pattern in biography
    #             biography = biography.replace(pattern, person_info[self.pattern_to_info_key[pattern]])
    #         person_name = self.person_name_gender_dict[str(self.index_file[k]['person_name_index'])]['full_name']
    #         biography = biography.replace('<<PERSON_NAME>>', person_name)
    #         result_dict[k] = biography
    #     json.dump(result_dict, open('./data/processed_0703/biography/fullname.json', 'w'), indent=4)

    def construct_biography_single_fullname(self):
        result_dict = {}
        for k in trange(len(self.index_file)):
            k = str(k)
            template_dict = self.extract_biography_template(k)
            person_name_index = str(self.index_file[k]['person_name_index'])
            gender = self.person_name_gender_dict[person_name_index]['gender']
            if gender == 'female':
                for key, template in template_dict.items():
                    template_dict[key] = self.gender_to_female(template)
            person_name = self.person_name_gender_dict[person_name_index]['full_name']
            person_info = self.extract_person_info(k)
            current_biography = ' '  # add a space to the beginning
            token_info = {}
            for attribute_name in template_dict:
                template = template_dict[attribute_name]
                current_biography += template.replace('<<PERSON_NAME>>', person_name) + ' '
                prefix, _ = current_biography.split(' ' + self.info_key_to_pattern[attribute_name])
                prefix_token_list = self.tokenizer(prefix)['input_ids']
                first_attribute_token_position = len(prefix_token_list)
                first_attribute_token = self.tokenizer(' ' + person_info[attribute_name])['input_ids'][0]
                current_biography = current_biography.replace(self.info_key_to_pattern[attribute_name],
                                                              person_info[attribute_name])
                # check whether the tokenization result is correct
                current_biography_token_list = self.tokenizer(current_biography)['input_ids']
                assert current_biography_token_list[:first_attribute_token_position] == prefix_token_list
                assert current_biography_token_list[first_attribute_token_position] == first_attribute_token
                token_info[attribute_name] = {
                    'first_token_position': first_attribute_token_position,
                    'first_token': first_attribute_token,
                }
            current_biography = current_biography[:-1]  # remove the last space
            result_dict[k] = {
                'biography': current_biography,
                'token_info': token_info,
                'tokenizer': self.tokenizer.__class__.__name__,
            }
        json.dump(result_dict,
                  open('./data/processed_0703/biography/single_fullname.json', 'w'),
                  indent=4)

    def construct_qa(self, train_ratio: float):
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
                person_info = self.extract_person_info(person_index)
                assert template[-1] == '>', 'the answer should not end with .'
                answer = ' ' + pattern
                prompt = template[: -len(answer)]
                answer = answer.replace(pattern,
                                        person_info[self.pattern_to_info_key[pattern]])
                assert answer[0] == ' ', 'the answer should start with space'
                prompt = prompt.replace('<<PERSON_NAME>>',
                                        person_info[self.pattern_to_info_key['<<PERSON_NAME>>']])
                assert prompt[-1] == ':'
                prompt_token = self.tokenizer(prompt)['input_ids']
                answer_token = self.tokenizer(answer)['input_ids']
                sequence_token = self.tokenizer(prompt + answer)['input_ids']
                assert sequence_token == prompt_token + answer_token
                qa_dict[self.pattern_to_info_key[pattern]] = {
                    'prompt': prompt,
                    'answer': answer
                }
            if int(person_index) < train_qa_count:
                train_dataset[int(person_index)] = qa_dict
            else:
                test_dataset[int(person_index)] = qa_dict
        if not os.path.exists('./data/processed_0703/qa'):
            os.mkdir('./data/processed_0703/qa')
        json.dump(train_dataset, open('./data/processed_0703/qa/train.json', 'w'), indent=4)
        json.dump(test_dataset, open('./data/processed_0703/qa/test.json', 'w'), indent=4)


def main():
    DataProcessor.process_name_gender(400, 400, 1000, 100000)
    # DataProcessor.process_birthday(1900, 2100)
    DataProcessor.process_city(200)
    # DataProcessor.process_university(300)
    # DataProcessor.process_major(100)
    # DataProcessor.process_company(263)
    # TemplateProcessor.process(50)
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "./model/gpt-neox/pythia-70m",
    #     model_max_length=512,  # unimportant config
    #     padding_side="right",
    #     use_fast=True,
    # )
    # constructor = DatasetConstructor(
    #     birthday_file_name='./data/processed_0703/attribute/birthday.json',
    #     city_file_name='./data/processed_0703/attribute/city.json',
    #     company_file_name='./data/processed_0703/attribute/company.json',
    #     major_file_name='./data/processed_0703/attribute/major.json',
    #     person_name_gender_file_name='./data/processed_0703/attribute/person_name_gender.json',
    #     university_file_name='./data/processed_0703/attribute/university.json',
    #     birthday_template_file_name='./template/processed_0703/1_birthday.json',
    #     birth_city_template_file_name='./template/processed_0703/2_birth_city.json',
    #     university_template_file_name='./template/processed_0703/3_university.json',
    #     major_template_file_name='./template/processed_0703/4_major.json',
    #     company_name_template_file_name='./template/processed_0703/5_company_name.json',
    #     company_city_template_file_name='./template/processed_0703/6_company_city.json',
    #     birthday_qa_template_file_name='./template/qa_template/1_birthday.txt',
    #     birth_city_qa_template_file_name='./template/qa_template/2_birth_city.txt',
    #     university_qa_template_file_name='./template/qa_template/3_university.txt',
    #     major_qa_template_file_name='./template/qa_template/4_major.txt',
    #     company_name_qa_template_file_name='./template/qa_template/5_company_name.txt',
    #     company_city_qa_template_file_name='./template/qa_template/6_company_city.txt',
    #     tokenizer=tokenizer,
    # )
    # # constructor.construct_biography_index()
    # constructor.construct_biography_single_fullname()
    # constructor.construct_qa(0.5)


if __name__ == '__main__':
    main()
