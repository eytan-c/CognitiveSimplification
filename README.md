# CognitiveSimplification
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


 # Creating New Annotated Datasets
 The following pipline can be used to create datasets annotated with the Simplification Operations Scheme proposed in the paper.
 
 The pipline is as follows:
 1. Run [make_base_ds_csv.py](make_base_ds_csv.py) to create a dataset in csv format with Word-Level operation annotations (see SARI metric, ) and TER score for each Simplification Instance (SI).
 2. Run [classify_datasets.py](classify_datasets.py) to create a CSV with additional metrics used for the final decision on the existence of a particular operation. We provide this step to save the dataset in a neutral state, so that future methods of classifying operations can be applied to the same data.
 3. Run [ops_classify_decisions.py](ops_classify_decisions.py) to create the final dataset format in JSON or text form, including decision on which operation appear in each SI, to be uploaded into a huggingface datasets format.
 4. The files in the [create_hf_datasets](create_hf_datasets) folder show examples of how to convert JSON files to a huggingface datasets format.
 
 The existing dataset will also be uploaded to huggingface datasets. 
 
 # Training and Evaluating Pretrained LMs on Simplification Datasets
 By running [run_training.py](run_training.py), one can fine-tune a pretrained T5 or BART model on a chosen simplification dataset, provided via a cmd line arguments. Currently only supports `manual` or `wiki-auto`.
 
 By running [run_eval.py](run_eval.py), one can evaluate a saved finetuned model on the ASSET dataset (Alva-Manchego et al. 2020).
