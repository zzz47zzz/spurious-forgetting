# Code for "Spurious Forgetting in Continual Learning of Language Models"


## 1. Experiments on the Biography dataset

**`code_for_biography_dataset`**: This directory contains the code to reproduce experiments using the synthetic Biography dataset. You can generate the dataset by running `./code_for_biography_dataset/physics_of_forgetting/data_preparation/preprocess_0720.py`. 

The experiments include:
   - Pretraining on 100K individuals
   - Continual finetuning on 20K individuals (Extended Setting: More tasks, varying individual counts, and diverse task types)
   - Recovery experiments on old tasks

It also contains the visualization for the following experiments:
   - Feature Perspective: `./code_for_biography_dataset/physics_of_forgetting/residual_stream_shift_analysis` 

## 2. Experiments on the real-world scenarios

**`code_for_realworld_scenarios`**: This folder includes code for experiments conducted on both the synthetic Biography dataset and real-world scenarios. The raw dataset is available in the `Supplementary Material`. For details on dataset preprocessing and overall repository usage, please refer to `./code_for_realworld_scenarios/README.md`. 

The experiments include:
   - Continual Finetuning on the Biography Dataset (Methods: EWC, LAMOL, Task Vector, Gradient Projection, SEQ, REPLAY, Freeze)
   - Safety Alignment (Methods: Freeze, SEQ)
   - Continual Instruction Tuning (Methods: Freeze, SEQ)
   - Continual Knowledge Editing (Methods: Freeze, SEQ)
   - Instance Incremental Learning (Methods: Freeze, SEQ)

It also contains the visualization notebook for the following experiments:
   - Task Vector: `./code_for_realworld_scenarios/visualization-tradeoff`
   - EWC, LAMOL, Gradient Projection: `./code_for_realworld_scenarios/visualization_continual_learning_methods`
   - Weight Update Perspective: `./code_for_realworld_scenarios/visualization-orthogonal-weight-update`.
   - Loss Landscape Perspective: `./code_for_realworld_scenarios/visualization-loss-landscape`