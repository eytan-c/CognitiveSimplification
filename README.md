# Cognitive Simplification
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

This is the respective repository of Paper: [Cognitive Simplification Operations Improve Text Simplification](). It includes the code used in the paper, as well as access to the FestAbility dataset, also accessible on [huggingface datasets](https://huggingface.co/datasets/eytanc/FestAbilityTranscripts).

There are three main sections for this repository:
1. Annotate a new dataset with Cognitive Simplification Operations, as detailed in the paper.
2. The training and evaluation scripts used in the paper, for reproduction purposes.
3. The dataset comparison analysis presented in Section 8 of the paper.

 # Creating New Annotated Datasets
 The following pipline can be used to create datasets annotated with the Simplification Operations Scheme proposed in the paper.
 
 The pipline is as follows - all in the [annotate_dataset](annotate_dataset) folder:
 1. Run [make_base_ds_csv.py](annotate_dataset/make_base_ds_csv.py) to create a dataset in csv format with Word-Level operation annotations (see SARI metric, ) and TER score for each Simplification Instance (SI).
 2. Run [classify_datasets.py](annotate_dataset/classify_datasets.py) to create a CSV with additional metrics used for the final decision on the existence of a particular operation. We provide this step to save the dataset in a neutral state, so that future methods of classifying operations can be applied to the same data.
 3. Run [ops_classify_decisions.py](annotate_dataset/ops_classify_decisions.py) to create the final dataset format in JSON or text form, including decision on which operation appear in each SI, to be uploaded into a huggingface datasets format.
 4. The files in the [create_hf_datasets](create_hf_datasets) folder show examples of how to convert JSON files to a huggingface datasets format.
 
 The existing dataset will also be uploaded to huggingface datasets. 
 
 # Training and Evaluating Pretrained LMs on Simplification Datasets
To fine-tune a pretrained T5 or BART model on a chosen simplification dataset run [run_training.py](run_training.py), interacting with different runtime variables via cmd line arguments. Currently only supports `manual` or `wiki-auto` training data (as provided in this repository).
 
To evaluate a saved finetuned model on the ASSET dataset (Alva-Manchego et al. 2020) or the Cognitive Simplification dataset presented in the paper, run [run_eval.py](run_eval.py) with the respective cmd line arguments.

All possible arguments are accessed via `--help`.

# Running a dataset analysis on new annotated datasets
 By running [create_dataset_analysis.py](create_dataset_analysis.py), you will create files in the correct format to be analyzed by [compare_datasets.py](compare_datasets.py). 
 The input for [create_dataset_analysis.py](create_dataset_analysis.py) is expected to be in the format as outputed by the annotation pipeline.
 
 By default, [compare_datasets.py](compare_datasets.py) compares all existing data in the [data/csvs/dataset_analysis](data/csvs/dataset_analysis) folder. This can be modified via cmd line arguments.

