# 240701

一共有 $N=100000$ 个个体.

- First, middle, and last names are drawn from pools of **400**, **400**, and **1000** English names respectively. We apply rejection sampling to ensure all N individuals have unique full names. 
  - First name / Middle name: [first names/us.txt at NameDatabases (github.com)](https://github.com/smashew/NameDatabases/blob/master/NamesDatabases/first names/us.txt) $\rightarrow$ `us_first_name.txt`
  - Last name: [surnames/us.txt at NameDatabases (github.com)](https://github.com/smashew/NameDatabases/blob/master/NamesDatabases/surnames/us.txt) $\rightarrow$ `us_last_name.txt`
  - 如何决定性别? 
    - 随机指定
    - [Gender by Name - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/591/gender+by+name) $\rightarrow$ `name_gender_dataset.csv`
- Birth years range from 1900 to 2099, months are selected from the 12 months, and days are chosen between 1 and 28. 
  - 直接通过代码生成.
- Birth cities are selected from **200** US cities, with their respective state abbreviations, such as Princeton, NJ and Cambridge, MA. 
  - 根据人口筛选城市: [List of United States cities by population - Wikipedia](https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population) $\rightarrow$ `List_of_United_States_cities_by_population_2.csv` (通过[link](https://wikitable2csv.ggor.de/)得到)
- Universities are drawn from a list of **300** US institutions. Some may have similar prefixes, like University of California, Berkeley/Irvine/Davis/etc. 
  - [List of research universities in the United States - Wikipedia](https://en.wikipedia.org/wiki/List_of_research_universities_in_the_United_States) $\rightarrow$ `List_of_research_universities_in_the_United_States_1/2/3.csv` (通过[link](https://wikitable2csv.ggor.de/)得到, )
- Majors are selected from **100** common college disciplines, including Computer Science, Physics, and Music.
  - [Most Popular College Majors: A Complete Ranking (scholarshipinstitute.org)](https://scholarshipinstitute.org/top-ranked-college-majors/) $\rightarrow$ 爬虫
- Employers are chosen from a list of **263** companies, featuring names like Meta Platforms, Microsoft, and Google.
  - ~~[List of largest companies in the United States by revenue - Wikipedia](https://en.wikipedia.org/wiki/List_of_largest_companies_in_the_United_States_by_revenue) (110个公司)~~
  - [Fortune 500 2017 (github.com)](https://github.com/iestynlee/DataAnalysis/blob/main/Fortune 500 2017 - Fortune 500.csv) (500个公司) $\rightarrow$ `Fortune 500 2017 - Fortune 500.csv`

# 240702

论文样例:

**Anya Briar Forger** was born on **October 2, 1996**. She spent her early years in **Princeton, NJ**. She received mentorship and guidance from faculty members at **Massachusetts Institute of Technology**. She completed her education with a focus on **Communications**. She had a professional role at **Meta Platforms**. She was employed in **Menlo Park, CA**.

---

**Carlos Jameson Stokes** has his annual celebration on **November 12, 2088**. He celebrates his birth in **San Francisco, CA**. He graduated from **Oklahoma State University**. He explored the theoretical aspects of **Information Systems**. He contributed his expertise to **United Airlines Holdings**. He acquired industry knowledge while working in **Chicago, IL**.

---

**Alondra Bennett Rooney** celebrates their life journey every year on **April 1, 1909**. They owe their roots to **Durham, NC**. They benefited from the resources and facilities provided by **University of South Alabama**. They developed a strong foundation in **Data Science**. They had a job at **The Southern Company**. They were involved in the industry of **Atlanta, GA**.

---

**Aidan Alexa Dennis**’s birth is celebrated annually on **July 17, 1968**. She calls **Palmdale, CA** her birthplace. She specialized in her field of study at **Stevens Institute of Technology**. She completed a rigorous program in **International Business**. She had employment prospects at **Johnson & Johnson**. She gained work experience in **New Brunswick, NJ**.

---

- 句子的开头可以是`<<PERSON_NAME>>`, 也可以是`<<PERSON_NAME>>'s`.
- 没有在prompt中保证属性, 如`<<BIRTHDAY>>`, 必须要在句子的最后, 但是大部分结果中属性在句子的最后面.
- 原文中主语可以是they, she, he.

birthday prompt:

```
Below you will be given a sentence. Try to paraphrase it in a different way while preserving its meaning.

The sentence needed to be paraphased is: 
<<PERSON_NAME>> was born on <<BIRTHDAY>>.

You should make sure that:
1. You don't need to fill the missing value of the sentence. Keep the template in the generation result.
2. The sentence should always begin with the name person, i.e., <<PERSON_NAME>>.
3. You can only use his to refer <<PERSON_NAME>> if necessary.
4. All paraphrases must be different.

Here are some examples:
<<PERSON_NAME>> has his annual celebration on <<BIRTHDAY>>.
<<PERSON_NAME>> celebrates his life journey every year on <<BIRTHDAY>>.
<<PERSON_NAME>>’s birth is celebrated annually on <<BIRTHDAY>>.

List 70 paraphrases of the sentence.
```

birth city prompt:

```
Below you will be given a sentence. Try to paraphrase it in a different way while preserving its meaning. 

The sentence needed to be paraphased is: 
<<PERSON_NAME>> was born in <<BIRTH_CITY>>.

You should make sure that:
1. You don't need to fill the missing value of the sentence. Keep the template in the generation result.
2. The sentence should always begin with the name person, i.e., <<PERSON_NAME>>.
3. You can only use his to refer <<PERSON_NAME>> if necessary.
4. All paraphrases must be different.

Here are some examples:
<<PERSON_NAME>> spent his early years in <<BIRTH_CITY>>.
<<PERSON_NAME>> celebrates his birth in <<BIRTH_CITY>>.
<<PERSON_NAME>> owes his roots to <<BIRTH_CITY>>.
<<PERSON_NAME>> calls <<BIRTH_CITY>> his birthplace.

List 70 paraphrases of the sentence.
```

college prompt:

```
Below you will be given a sentence. Try to paraphrase it in a different way while preserving its meaning. 

The sentence needed to be paraphased is: 
<<PERSON_NAME>> graduated from <<COLLEGE>>.

You should make sure that:
1. You don't need to fill the missing value of the sentence. Keep the template in the generation result.
2. The sentence should always begin with the name person, i.e., <<PERSON_NAME>>.
3. You can only use "his" to refer <<PERSON_NAME>> if necessary. Do NOT use "her" or "them".
4. All paraphrases must be different.

Here are some examples:
<<PERSON_NAME>> received mentorship and guidance from faculty members at <<COLLEGE>>.
<<PERSON_NAME>> finished his education at <<COLLEGE>>.
<<PERSON_NAME>> specialized in her field of study at <<COLLEGE>>.

List 70 paraphrases of the sentence.
```

major prompt

```
Below you will be given a sentence. Try to paraphrase it in a different way while preserving its meaning. 

The sentence needed to be paraphased is: 
<<PERSON_NAME>> majored in <<MAJOR>>.

You should make sure that:
1. You don't need to fill the missing value of the sentence. Keep the template in the generation result.
2. The sentence should always begin with the name person, i.e., <<PERSON_NAME>>.
3. You can only use his to refer <<PERSON_NAME>> if necessary. Do NOT use "her" or "them".
4. All paraphrases must be different.

Here are some examples:
<<PERSON_NAME>> explored the theoretical aspects of <<MAJOR>>.
<<PERSON_NAME>> developed a strong foundation in <MAJOR>>.
<<PERSON_NAME>> completed a rigorous program in <MAJOR>>.

List 70 paraphrases of the sentence.
```

company name prompt

```
Below you will be given a sentence. Try to paraphrase it in a different way while preserving its meaning. 

The sentence needed to be paraphased is: 
<<PERSON_NAME>> worked in <<COMPANY_NAME>>.

You should make sure that:
1. You don't need to fill the missing value of the sentence. Keep the template in the generation result.
2. The sentence should always begin with the name person, i.e., <<PERSON_NAME>>.
3. You can only use his to refer <<PERSON_NAME>> if necessary.
4. All paraphrases must be different.

Here are some examples:
<<PERSON_NAME>> had a job at <<COMPANY_NAME>>.
<<PERSON_NAME>> had employment prospects at <<COMPANY_NAME>>.
<<PERSON_NAME>> had a professional role at <<COMPANY_NAME>>.
<<PERSON_NAME>> contributed his expertise to <<COMPANY_NAME>>.

List 70 paraphrases of the sentence.
```

company city prompt

```
Below you will be given a sentence. Try to paraphrase it in a different way while preserving its meaning. 

The sentence needed to be paraphased is: 
<<PERSON_NAME>> was employed in <<COMPANY_CITY>>.

You should make sure that:
1. You don't need to fill the missing value of the sentence. Keep the template in the generation result.
2. The sentence should always begin with the name person, i.e., <<PERSON_NAME>>.
3. You can only use his to refer <<PERSON_NAME>> if necessary.
4. All paraphrases must be different.

Here are some examples:
<<PERSON_NAME>> acquired industry knowledge while working in <<COMPANY_CITY>>.
<<PERSON_NAME>> were involved in the industry of <<COMPANY_CITY>>.
<<PERSON_NAME>> gained work experience in <<COMPANY_CITY>>.

List 70 paraphrases of the sentence.
```

## 为什么不用gpt-4

gpt-4难以根据指令生成需要的结果

gpt-4的结果: 

```
Sure, here are 70 paraphrases of the given template sentence.

1. <<PERSON_NAME>> came into the world on <<BIRTHDAY>>.
2. <<PERSON_NAME>> made their debut in the world on <<BIRTHDAY>>.
3. <<PERSON_NAME>> first drew breath on <<BIRTHDAY>>.
4. The day <<BIRTHDAY>> marks <<PERSON_NAME>>'s entry into the world.
5. <<BIRTHDAY>> was the day <<PERSON_NAME>> was born.
6. The world welcomed <<PERSON_NAME>> on <<BIRTHDAY>>.
7. The birth of <<PERSON_NAME>> happened on <<BIRTHDAY>>.
8. The start of <<PERSON_NAME>>'s life was on <<BIRTHDAY>>.
9. <<PERSON_NAME>> first opened their eyes to the world on <<BIRTHDAY>>.
10. <<PERSON_NAME>>'s life journey began on <<BIRTHDAY>>.
11. The advent of <<PERSON_NAME>> was on <<BIRTHDAY>>.
12. On <<BIRTHDAY>>, <<PERSON_NAME>> was born.
13. <<PERSON_NAME>>'s existence commenced on <<BIRTHDAY>>.
14. On <<BIRTHDAY>>, the world gained <<
```

# 240703

## 生成结果:

"Tina Tom Fan's special day is September 26, 2073. He started his life in Greensboro, NC. He earned a degree from University of Oklahoma. He pursued studies in Management Science. He had duties at Humana. He pursued his career in Louisville, KY."

# 240705

训练集中的数据个数似乎与训练时间有关? 所有实验均是80000steps.

一共80000 steps, 176个batch, 455个epoch. 大约要训练17-18小时.

其中一条训练数据: 

```
"Wesley Alexander Emley celebrates his special day on May 28, 1952. Wesley Alexander Emley's first breath was taken in Elk Grove, CA. Wesley Alexander Emley left Kansas State University with a degree. Wesley Alexander Emley completed his major in EMT and Paramedic. Wesley Alexander Emley had duties at HP. Wesley Alexander Emley spent his working hours in Palo Alto, CA.<|endoftext|>...<|endoftext|>Genevieve Rita Ribron's entry into the world"
```

```
input_ids: 
tensor([   54, 49048, 10009,  2295,  1636, 24538,   465,  2041,  1110,   319,
         1737,  2579,    11, 26352,    13, 28634, 10009,  2295,  1636,   338,
          717,  8033,   373,  2077,   287, 40151, 24144,    11,  7257,    13,
        28634, 10009,  2295,  1636,  1364,  9470,  1812,  2059,   351,   257,
         4922,    13, 28634, 10009,  2295,  1636,  5668,   465,  1688,   287,
          412, 13752,   290,  2547,  2434,   291,    13, 28634, 10009,  2295,
         1636,   550, 10741,   379,  6574,    13, 28634, 10009,  2295,  1636,
         3377,   465,  1762,  2250,   287, 44878, 34317,    11,  7257,    13,
        50256,    32, 35670,    64, 19068,  1001,   361,  1625,   656,  6224,
          ...,
        50256, 39358,    85, 12311, 42605, 23133,  1313,   338,  5726,   656,
          262,   995])
target:
tensor([   54, 49048, 10009,  2295,  1636, 24538,   465,  2041,  1110,   319,
         1737,  2579,    11, 26352,    13, 28634, 10009,  2295,  1636,   338,
          717,  8033,   373,  2077,   287, 40151, 24144,    11,  7257,    13,
        28634, 10009,  2295,  1636,  1364,  9470,  1812,  2059,   351,   257,
         4922,    13, 28634, 10009,  2295,  1636,  5668,   465,  1688,   287,
          412, 13752,   290,  2547,  2434,   291,    13, 28634, 10009,  2295,
         1636,   550, 10741,   379,  6574,    13, 28634, 10009,  2295,  1636,
         3377,   465,  1762,  2250,   287, 44878, 34317,    11,  7257,    13,
        50256,  -100, 35670,    64, 19068,  1001,   361,  1625,   656,  6224,
          ...,
        50256,  -100,    85, 12311, 42605, 23133,  1313,   338,  5726,   656,
          262,   995])
attention_mask:
tensor([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          ...,
        False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
         True,  True])
```


