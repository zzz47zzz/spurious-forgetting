# A Codebase for Continual Learning of Large Language Models

## Contents
- [Introduction](#Introduction)
- [Supported List](#Supported%20List)
- [Usage](#Usage)

## Introduction
This is a repository for Continual Learning of Large Language Models. 
- It supports both generative and discriminative models in [transformers](https://huggingface.co/docs/transformers/index).
- It supports using [accelerate](https://huggingface.co/docs/accelerate/index) for distributed data parrallel and model parallel.
- It supports using [wandb](https://wandb.ai/site) for logging.


## Supported List

### Scenario
- [x] Safety Alignment (AOA alignment)
- [x] Continual Instruction Tuning
- [x] Continual Knowledge Editing
- [x] Instance-Incremental Learning


### Methods

#### General (Text/Intent) Classification
- [x] SEQ
- [x] [LAMOL_KD (arXiv)](https://arxiv.org/abs/2312.07887)
- [x] [Replay](https://arxiv.org/abs/1902.10486)
- [x] [PEFT (including, LoRA, PromptTuning)](https://huggingface.co/docs/peft/index)
- [x] [LAMOL (ICLR 2020)](https://openreview.net/forum?id=Skgxcn4YDS)
- [x] [L2KD (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.233/)
- [x] [PCLL (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.766/)
- [x] [ProgPrompt (ICLR 2023)](https://openreview.net/forum?id=UJTgQBc91_)
- [x] [LFPT5 (ICLR 2022)](https://openreview.net/forum?id=7mozamSFNt4)
- [x] [AdapterCL (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.590/)
- [x] [EWC (PNAS 2017)](https://www.pnas.org/doi/abs/10.1073/pnas.1611835114)

### Datasets


#### Safety Alignment
- [x] AOA alignment

#### Continual Instruction Tuning
- [x] TRACE benchmark

#### Continual Knowledge Editing
- [x] ZsRE

#### Instance Incremental Learning
- [x] Concept-1K

#### Intent Classification
- [x] Topic3datasets (agnews, dbpedia, yahoo)


## Usage

### Overview
```
.
├── main_CL.py                      # This this the python file to be executed for running all experiments
├── utils                               # This folder contains all basic files for incremental learning 
│   ├── backbone.py                     # This file loads backbone models from the transformers library
│   ├── buffer.py                       # This file defines the replay buffer
│   ├── classifier.py                   # This file loads Linear/CosineLinear classifiers
│   ├── wrapmodel.py                    # This file wrap the model for using DeepSpeed with accelerate
│   ├── dataformat_preprocess.py        # This file preprocess the raw datasets to the continual learning dataset
│   ├── dataloader.py                   # This file prepare the input for languge models
│   ├── dataset.py                      # This file defines the format for different datasets for continual learning
│   ├── download_backbones.py           # This file downloads models in advance to avoid network problem.
│   ├── evaluation.py                   # This file defines the evaluation process for various tasks
│   ├── factory.py                      # This file loads the various models from the ./models folder
│   ├── logger.py                       # This file defines the logger
│   ├── metric.py                       # This file defines the evaluation metric for continual learning
│   ├── optimizer.py                    # This file defines the optimizer for different models
│   ├── prompt.py                       # This file defines the prompt used for different tasks
│   ├── probing.py                      # This file computes the probing performance
│   └── config.py                       # This file defines general parameters and settings for the experiments
├── config                          # This folder contains the hyper-parameters for each methods in each datasets
├── dataset                         # This folder contains datasets for continual learning
├── models                          # This folder contains models for continual learning
└── experiments                     # This folder contains log data for each run                 
```

### Quick Start

#### Step 1: prepare the environment
```
pip install -r requirement.txt
```

#### Step 2: prepare the dataset
Check the *support_dataset_list* in *utils/dataformat_preprocess.py* and select the dataset you want for experiment.

Then, download the raw dataset to the folder *dataset/{dataset-name}*.
For example, download the concept_1k to the folder *dataset/biography_qa*.
The raw datasets can be downloaded in the Supplementray Material (in the OpenReview link of this paper).
The downloaded raw datasets will have the following folders.
```
./
├── biography_qa
├── concept_1k
├── safety_alignment
├── TRACE
└── zsre
```

Next, exceute the *preprocess_dataset.sh*.
It will automatically preprocess 4 default datasets for reproducing results ('biography_qa','TRACE','zsre','concept_1k','safety_alignment') and create new folders in *datasets/{dataset-for-continual-learning-name}* automatically (e.g.,*biography_qa_task2*).
After preprossing all dataset to the format for continual learning, the dataset folder will look like this.
```
./
├── biography_qa
├── biography_qa_task2
├── concept_1k
├── concept_1k_task10
├── safety_alignment
├── safety_alignment_llama2_task2
├── TRACE
├── TRACE_task8
├── zsre
└── zsre_task10
```


If you do not need to customize the datasets, you can skip to Step 3.

To customize the datasets, you can run *utils/dataformat_preprocess.py* with your own parameters (e.g., random seeds, num of tasks).
This process will create a new target folder *dataset/{dataset-for-continual-learning-name}*.
In the target folder, two json files *continual_data.json* and *continual_config.json* will be saved.
For example, you can prepare biography_qa by runing
```
python utils/dataformat_preprocess.py --dataset biography_qa --seed 1
``` 

The program will create target folders *dataset/biography_qa_task2*.

We note that fixing the random seed enables that exctaly the same datasets can be generated on different devices.
Finally, the post-precessed dataset *biography_qa_task2* is ready for continual learning!

#### Step 3: select the yaml file for hyper-parameters
The yaml file contains the hyper-parameters for each method.
For example, the hyper-parameter of SEQ can be defined in *config/biography_qa_task2/SEQ.yaml*
```
il_mode: CIT

dataset: biography_qa_task2
classification_type: sentence-level

classifier: None

method: SEQ

training_epochs: 25
training_epochs_first_task: 10

lr: 5e-6 
batch_size: 48

info_per_steps: 25

prompt_type: auto
```

### Step 4: execute the main_CL.py
For example, you can run SEQ method on clinc150_task15 dataset with bert-base-cased using the following command:

```
python main_CL.py --exp_prefix {your-experiment-name} --cfg './config/biography_qa_task2/SEQ.yaml' --backbone pythia-160m-bio-pretrained-100k --load_llm_ckpt True --backbone_cache_path {your-model-pretrained-on-100K-individuals}
```

If you want to use wandb for logging (see [here](https://docs.wandb.ai/tutorials/pytorch) for more help):
```
python main_CL.py --is_wandb True --wandb_project {your-project-name} --wandb_entity {your-entity-name} --exp_prefix {your-experiment-name} --cfg './config/biography_qa_task2/SEQ.yaml' --backbone pythia-160m-bio-pretrained-100k --load_llm_ckpt True --backbone_cache_path {your-model-pretrained-on-100K-individuals}
```

If you want to use accelerate for data/model parallel (see [here](https://huggingface.co/docs/accelerate/quicktour) for more help):
```
accelerate launch --config_file {your-accelerate-config-file} main_CL.py --is_wandb True --wandb_project {your-project-name} --wandb_entity {your-entity-name} --exp_prefix {your-experiment-name} --cfg './config/biography_qa_task2/SEQ.yaml' --backbone pythia-160m-bio-pretrained-100k --load_llm_ckpt True --backbone_cache_path {your-model-pretrained-on-100K-individuals}
```

Please refer to *utils/config.py* for more general paramters and *models/{model-name}.py* for more model-specific parameters.
