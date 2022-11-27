# Language Identification

## Introduction

This is a project for language identification.
This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on
the [Language Identification](https://huggingface.co/datasets/papluca/language-identification#additional-information)
dataset.

We follow the instruction
of [xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection).
Besides the naive classification
in [xlm-roberta-base-language-detection](https://huggingface.co/papluca/xlm-roberta-base-language-detection), we add
balance-training and contrastive learning:

* Balance Training: all samples are fully shuffled to guarantee that there are equal number of samples for each language
  in each batch.
* Contrastive Learning: In each batch, we grouped the samples, each language had only one sample in each group, and each
  group contained all the languages in the dataset. For each anchor, the other samples in the same group can be
  considered negative samples. We randomly choose a sample from another group of the same language as the anchor as the
  positive sample.

## Performance

We test our model on
the [Language Identification](https://huggingface.co/datasets/papluca/language-identification#additional-information)
dataset, the overall accuracy on test set is 99.64%, the overall accuracy on dev set is 99.76%.

A more detailed evaluation on test set is provided by the following table.

Lang    | Precision | Recall | F1    | support | 
--------|-----------|--------|-------|---------|
ar    | 1.0       | 0.998  | 0.999 | 500     |
bg  | 0.996     | 0.998  | 0.997 | 500     |
de  | 1.0       | 1.0    | 1.0   | 500     |
el   | 1.0       | 1.0    | 1.0   | 500     |
en | 0.998     | 1.0    | 0.999 | 500     |
es   | 0.996     | 1.0    | 0.998 | 500     |
fr | 1.0       | 1.0    | 1.0   | 500     |
hi | 0.992     | 0.98   | 0.986 | 500     |
it    | 1.30      | 0.7    | 1.9   | 500     |
ja   | 1.0       | 1.0    | 1.0   | 500     |
nl | 1.0       | 0.996  | 0.998 | 500     |
pl   | 0.996     | 1.0    | 0.996 | 500     |
pt | 1.0       | 0.992  | 0.996 | 500     |
ru | 0.994     | 0.998  | 0.996 | 500     |
sw    | 0.980     | 0.998  | 0.989 | 500     |
th   | 1.0       | 1.0    | 1.0   | 500     |
tr   | 0.996     | 1.0    | 0.999 | 500     |
ur | 0.982     | 0.976  | 0.979 | 500     |
vi | 1.0       | 1.0    | 1.0   | 500     |
zh    | 1.0       | 1.0    | 1.0   | 500     |

We also test our model on the [XNLI-15way](https://github.com/facebookresearch/XNLI) dataset to evaluate the
generalization, which contains a 15way parallel corpus of 10,000 sentences.
The overall accuracy is 98.79%. A more detailed evaluation is provided by the following table.

Lang    | Precision | Recall | F1    | support | 
--------|-----------|--------|-------|---------|
ar    | 0.999     | 0.999  | 0.999 | 10,000     |
bg  | 0.998     | 0.997  | 0.997 | 10,000     |
de  | 0.999     | 0.994  | 0.997 | 10,000     |
el   | 1.0       | 1.0    | 1.0   | 10,000     |
en | 0.994     | 0.933  | 0.963 | 10,000     |
es   | 0.999     | 0.962  | 0.98  | 10,000     |
fr | 0.999     | 0.995  | 0.997 | 10,000     |
hi | 0.986     | 0.982  | 0.984 | 10,000     |
ru | 0.997     | 0.998  | 0.998 | 10,000     |
sw    | 0.972     | 0.995  | 0.983 | 10,000     |
th   | 1.0       | 0.999  | 0.997 | 10,000     |
tr   | 0.996     | 0.998  | 0.997 | 10,000     |
ur | 0.983     | 0.97   | 0.977 | 10,000     |
vi | 1.0       | 0.998  | 0.999 | 10,000     |
zh    | 1.0       | 0.999  | 1.0   | 10,000     |

## Download

We released our best checkpoint and log. You can download
from [here](https://drive.google.com/drive/folders/1R6Nxx5lu6JkFL_ktWeTi0llW_0cZ1Xel?usp=sharing).

## Training procedure

Before training and testing, you need to download [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) and [Language Identification](https://huggingface.co/datasets/papluca/language-identification#additional-information).

For training, you can run

```
python run_xlm_LangIden.py --do_train --example_file_train path_to_train_file --example_file_dev path_to_dev_file --model_name_or_path path_to_xlm-roberta-base --output_dir output_dir

```

For testing on the [Language Identification](https://huggingface.co/datasets/papluca/language-identification#additional-information), you can run

```
python run_xlm_LangIden.py --do_test --eval_model_dir path_to_ckpt  --example_file_test path_to_test --model_name_or_path path_to_xlm-roberta-base

```
For testing on the [XNLI-15way](https://github.com/facebookresearch/XNLI), you can run

```
python run_xlm_LangIden_XNLI_test.py --do_test --eval_model_dir path_to_ckpt --example_file_test path_to_XNLI-15way --model_name_or_path path_to_xlm-roberta-base

```

The evaluation result will be saved in the eval_model_dir.

## Training hyperparameters

The following hyperparameters were used during training:

* learning_rate: 2e-05
* train_batch_size: 120
* num gpus: 2
* seed: 42
* optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
* lr_scheduler_type: linear
* num_epochs: 4

Model is trained on two A100-40G.

## Framework versions

* Pytorch 1.7.1+cu110
* Transformers 4.18.0


