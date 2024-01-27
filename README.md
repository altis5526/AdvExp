# Where are Biases? Adversarial Debiasing with Spurious Feature Visualization (AdvExp)
## Introduction
This is an official repository of MMM2024 regular paper: "Where are Biases? Adversarial Debiasing with Spurious Feature Visualization
" by Chi-Yu Chen, Pu Ching, Pei-Hsin Huang, and Min-Chun Hu
## Abstract
To avoid deep learning models utilizing shortcuts in a training dataset, many debiasing models have been developed to encourage models learning from accurate correlations. Some research constructs robust models via adversarial training. Although this series of methods shows promising debiasing performance, we do not know precisely what spurious features have been discarded during adversarial training. To address its lack of explainability especially in scenarios with low error tolerance, we design AdvExp, which not only visualizes the underlying spurious feature behind adversarial training but also maintains good debiasing performance with the assistance of a robust optimization algorithm. We show promising performance of AdvExp on BiasCheXpert, a subsampled dataset from CheXpert, and uncover potential regions in radiographs recognized by deep neural networks as gender or race-related features.
## Requirements
```
pip install -r requirements.txt
```
## Dataset
### BiasCheXpert
BiasCheXpert is a dataset subsampled from CheXpert-v1.0, a public dataset containing chest radiographs of 65240 patients, to artificially enhance bias in our experiments. The original dataset of CheXpert can be downloaded on [Website](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2). In our original experiment, we actually used the downsampled CheXpert version officially released by STANFORD ML GROUP, but the link was not publicily released. We provided the csv files in our repo which helps you to create BiasCheXpert from CheXpert.
## Usage
```
python main.py
```
