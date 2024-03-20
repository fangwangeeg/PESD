# PESD
This is the implementation of the paper titled ['Cross-Dataset EEG Emotion Recognition based on Pre-trained Vision Transformer Considering Emotional Sensitivity Diversity'] using PyTorch (Version 1.11.0).

This repository contains the source code of our paper, using the following datasets:


- [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html): 15 subjects participated in experiments with videos as emotion stimuli (three emotions: positive/negative/neutral) and EEG was recorded with 62 channels at a sampling rate of 1000Hz.

- [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html): 15 subjects participated experiments with videos as emotion stimuli (four emotions: happy/sad/neutral/fear).  EEG was recorded with 62 channels at a sampling frequency of 1000Hz.

- [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/): 32 subjects participated experiments with music videos as emotion stimuli (emotional states across arousal, valence, liking, and dominance, utilizing a scale from 1 to 9 for each video).  EEG was recorded with 32 channels at a sampling frequency of 512Hz.

- [FACED](https://doi.org/10.7303/syn50614194): 123 subjects participated experiments with videos as emotion stimuli (nine emotional categories: amusement, inspiration, joy, tenderness; anger, fear, disgust, sadness, and neutral emotion). EEG was recorded with 32 channels at a sampling frequency of 1000Hz.


## Prerequisites
Please follow the steps below in order to be able to train our models:

1 - Download the four datasets and perform pre-processing and feature extraction using the code available in the [./PESD/feature_extract/]. The code while strictly following the official dataset description and the original publication in which the dataset was published.

2 - Save the extracted features and labels into folders for each dataset (e.g., '/SEED_PSD/', 'SEED_IV_PSD', 'DEAP_PSD', and 'FACED_PSD'). 

3 - Pretrained Models: We provide the pre-trained models [./PESD/pretrained_models] from the four datasets. Each pretrained model is trained on the subject with the highest emotional sensitivity in their dataset. 
 
4 - Run the 'main.py' script to implement the PESD model and the benchmarks outlined in the paper.
