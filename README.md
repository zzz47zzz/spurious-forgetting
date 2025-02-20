# [ICLR 2025] Spurious Forgetting in Continual Learning of Language Models

[![ICLR 2025](https://img.shields.io/badge/ICLR2025-Spurious_Forgetting-1c8139.svg)](https://openreview.net/forum?id=ScI7IlKGdI)

## 📢 Updates 

- **February 2025** – Code cleaned and additional instructions for generating the Biography dataset added. The processed Biography dataset is available for download [here](https://drive.google.com/file/d/1TEIN1qBwN8d1E6AbyyLdPQfx_ZLKD_0K/view?usp=sharing).
- **February 2025** – Paper interpretations now available on [PaperWeekly](https://mp.weixin.qq.com/s/d7QkZGBE1IKnrEfcqDH4ng) and [知乎](https://zhuanlan.zhihu.com/p/23021161842).
- **January 2025** – The code for our ICLR 2025 paper, **"Spurious Forgetting in Continual Learning of Language Models"**, is now publicly available! 🚀 Explore the code and our findings.

Welcome to the repository for our ICLR 2025 paper, **"Spurious Forgetting in Continual Learning of Language Models"**. This repository is organized into two main sections, each focusing on different experiments and use cases. 

![Illustration](./assets/introduction.png)

---

## **1. Experiments on the Biography Dataset**

### 👀 Overview
This section contains experiments utilizing the **Biography Dataset**, a synthetic dataset designed to simulate a controlled continual learning environment for language models.

### 📂 Directory: `code_for_biography_dataset`

#### **Generating the Biography Dataset**

To generate the Biography dataset, follow these steps:

1. Navigate to the `physics_of_forgetting` directory:
    ```bash
    cd ./code_for_biography_dataset/physics_of_forgetting/
    ```

2. Set the `PYTHONPATH` and run the preprocessing script:
    ```bash
    export PYTHONPATH=.
    python ./data/preprocess.py
    ```

   After running the script for about **15 minutes**, the preprocessed data will be saved in the following locations:
   - **Pretraining data**: `./data/processed_final/biography/multi5_permute_fullname.json` (~1.17GB)
   - **Fine-tuning QA data**: `./data/processed_final/qa/all.json` (~183MB)

   **Note**: The pretraining data (`multi5_permute_fullname.json`) contains five instances of each person in the dataset, with the attributes shuffled to simulate a dynamic learning environment. For further details, refer to [this paper](https://arxiv.org/abs/2309.14316).

#### **Data Structure Overview**

The preprocessed data is structured as follows:

- **Pretraining Data** (`multi5_permute_fullname.json`): Each person in the dataset has five entries, with shuffled attributes. Here is an example structure for a single person (Person 0):

    ```json
    {
        "0_0": {
            "biography": "Lucy Damian Moscicki held a job in Palo Alto, CA. Lucy Damian Moscicki's life journey started in Elk Grove, CA. Lucy Damian Moscicki specialized in EMT and Paramedic. Lucy Damian Moscicki completed his degree requirements at Kansas State University. Lucy Damian Moscicki celebrates his special day on May 28, 1952. Lucy Damian Moscicki contributed his skills to HP.",
            "token_info": {
                "company_city": {
                    "first_token_position": 10,
                    "first_token": 5226
                },
                "birth_city": {
                    "first_token_position": 27,
                    "first_token": 3599
                },
                "major": {
                    "first_token_position": 41,
                    "first_token": 33566
                },
                "university": {
                    "first_token_position": 58,
                    "first_token": 15391
                },
                "birthday": {
                    "first_token_position": 73,
                    "first_token": 2552
                },
                "company_name": {
                    "first_token_position": 88,
                    "first_token": 19517
                }
            },
            "tokenizer": "GPTNeoXTokenizerFast"
        }
    }
    ```

- **QA Data** (`all.json`): This contains question-answer pairs about each person's biography. Here's an example for Person 0:

    ```json
    {
        "0": {
            "birthday": {
                "prompt": "What is the birth date of Lucy Damian Moscicki?\nAnswer:",
                "answer": " May 28, 1952"
            },
            "birth_city": {
                "prompt": "What is the birth city of Lucy Damian Moscicki?\nAnswer:",
                "answer": " Elk Grove, CA"
            },
            "university": {
                "prompt": "Which university did Lucy Damian Moscicki study?\nAnswer:",
                "answer": " Kansas State University"
            },
            "major": {
                "prompt": "What major did Lucy Damian Moscicki study?\nAnswer:",
                "answer": " EMT and Paramedic"
            },
            "company_name": {
                "prompt": "Which company did Lucy Damian Moscicki work for?\nAnswer:",
                "answer": " HP"
            },
            "company_city": {
                "prompt": "Where did Lucy Damian Moscicki work?\nAnswer:",
                "answer": " Palo Alto, CA"
            }
        }
    }
    ```

#### **Alternative Dataset Download**

Alternatively, you can download the final preprocessed dataset directly from [Google Drive](https://drive.google.com/file/d/1TEIN1qBwN8d1E6AbyyLdPQfx_ZLKD_0K/view?usp=sharing), which contains the exact data used for all experiments in our paper. This data is identical to what you would generate by running the preprocessing script except the order of person name. You can copy the pretraining and QA files from our directory `processed_0720_v0730` to your directory `processed_final`.

#### **Pretraining and Fine-tuning Setup**

We generate data for a total of **200K persons**. When running the pretraining or fine-tuning experiments, use the configuration files in the `./config` folder to specify which set of persons to use for each phase:

- For **pretraining** on persons 0-100K, use the configuration file: `./config/v0731/single/pre_training.json`.
- For **fine-tuning** on the first 50K persons and testing on the next 50K, use the configuration file: `./config/v0731/single/fine_tuning.json`.


#### **Experiments**
1. **Pretraining**:
   - Train a model on 100K individuals to establish a foundational knowledge base.
2. **Continual Finetuning**:
   - Incrementally finetune the model on 20K individuals.
   - Extended Settings:
     - Include more tasks.
     - Vary the number of individuals.
     - Explore diverse task types.
3. **Recovery Experiments**:
   - Investigate the model’s ability to recover performance on previously seen tasks.

#### **Visualizations**
- **Feature Perspective**:
  - Analyze residual stream shifts in the visualization directory:
    ```bash
    ./code_for_biography_dataset/physics_of_forgetting/residual_stream_shift_analysis
    ```

---

## **2. Experiments on Real-World Scenarios**

### 👀 Overview
This section extends the research to real-world scenarios, integrating methods and datasets that reflect practical continual learning challenges. 

### 📂 Directory: `code_for_realworld_scenarios`

#### **Repository Details**
This section builds upon [this incremental learning repository](https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm). For detailed instructions on dataset preprocessing and usage, refer to the README within this directory:
```bash
./code_for_realworld_scenarios/README.md
```

#### **Experiments**
1. **Continual Finetuning on Biography Dataset**:
   - Methods: EWC, LAMOL, Task Vector, Gradient Projection, SEQ, REPLAY, Freeze.
2. **Safety Alignment**:
   - Methods: Freeze, SEQ.
3. **Continual Instruction Tuning**:
   - Methods: Freeze, SEQ.
4. **Continual Knowledge Editing**:
   - Methods: Freeze, SEQ.
5. **Instance Incremental Learning**:
   - Methods: Freeze, SEQ.

#### **Visualizations**
- **Task Vector**:
  - Explore tradeoffs using:
    ```bash
    ./code_for_realworld_scenarios/visualization-tradeoff
    ```
- **Continual Learning Methods**:
  - Visualize EWC, LAMOL, and Gradient Projection results:
    ```bash
    ./code_for_realworld_scenarios/visualization_continual_learning_methods
    ```
- **Weight Update Perspective**:
  - Examine orthogonal weight updates:
    ```bash
    ./code_for_realworld_scenarios/visualization-orthogonal-weight-update
    ```
- **Loss Landscape Perspective**:
  - Analyze the model’s loss landscape:
    ```bash
    ./code_for_realworld_scenarios/visualization-loss-landscape
    ```

---

### **Cite Our Work**
If you find this repository useful, please consider citing our research:
```bibtex
@inproceedings{
    zheng2025spurious,
    title={Spurious Forgetting in Continual Learning of Language Models},
    author={Junhao Zheng and Xidi Cai and Shengjie Qiu and Qianli Ma},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=ScI7IlKGdI}
}
```

### **🚀 Star the Repository**
Help us grow by starring 🌟 this repository on GitHub! 💖

---

Thank you for your interest in our work. We look forward to your feedback and collaboration! ✨

If you have questions about this repository, please feel free to contact me at [junhaozheng47@outlook.com](mailto:junhaozheng47@outlook.com).
