"""
Functions for heuristic decision on simplification actions.
Also, run "classification" on dataset and save to txt files + jsons
"""
import argparse

import tqdm
from typing import NamedTuple, Union, TextIO, Union
import json
import pandas as pd
import numpy as np
import pathlib


def decide_proximation(row: Union[NamedTuple, pd.Series]):
    """
    Decide on the proximation (PROX) action.
    Heuristic - if the row has one of point-of-view change, verb tense change, or passive->active change.
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :return: Boolean for resolution of heuristic
    """
    return sum([row.pov_change, row.verb_tense_change, row.passive_active_sub]) > 0


def decide_rephrasing(row: Union[NamedTuple, pd.Series]):
    """
    Decide on the rephrasing (REPHRASE) action.
    Heuristic - if the row has one of the rephrasing actions (synonyms, word to phrase, phrase to word,
                or phrase to phrase).
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :return: Boolean for resolution of heuristic
    """
    return sum([row.simple_synonym, row.word2phrase, row.phrase2word, row.phrase2phrase]) > 0


def decide_deleting_info(row: NamedTuple, pct_unused=0.3, token_length_ratio=1.2):
    """
    Decide on the delete information (DEL) action.
    Heuristic - if row has removal or summarization, or if the token length ratio is longer that token_length_ratio,
                or if the percent of deleted and unused words is greater than pct_unsued and the token length ration is
                greate than 1.
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :param pct_unused: threshold for the percent of deleted and unused words
    :param token_length_ratio: threshold for the token length ratio
    :return: Boolean for resolution of heuristic
    """
    if row.removal > 0 or row.summarization > 0:
        return True
    if row.token_length_ratio >= token_length_ratio:
        return True
    else:
        return row.percent_deleted_unused > pct_unused and row.token_length_ratio > 1


def decide_adding_info(row: NamedTuple, token_length_ratio=1):
    """
    Decide on the adding information (ADD) action.
    Heuristic - if the row's token length ratio is less that token_length_ratio.
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :param token_length_ratio: threshold for the token length ratio
    :return: Boolean for resolution of heuristic
    """
    return row.token_length_ratio < token_length_ratio


def decide_example(row: Union[NamedTuple, pd.Series]):
    """
    Decide on the add example (EXAMPLE) action.
    Heuristic - if the number of examples add is > 0.
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :return: Boolean for resolution of heuristic
    """
    return row.examples > 0


def decide_explanation(row: Union[NamedTuple, pd.Series]):
    """
    Decide on the add explanation (EXPLAIN) action.
    Heuristic - if the number of explanations added is > 0.
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :return: Boolean for resolution of heuristic
    """
    return row.explanations > 0


def decide_explicitation(row: Union[NamedTuple, pd.Series]):
    """
    Decide on the pronoun explicitation (EXPLICIT) action.
    Heuristic - if the number of pronoun explicitations is > 0.
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :return: Boolean for resolution of heuristic
    """
    return row.pron_explicitation > 0


def decide_intra_sentence_rearrange(row: Union[NamedTuple, pd.Series]):
    """
    Decide on the intra sentence information rearrangement (REORDER) action.
    Heuristic - if the number of moved words that are out-of-order or the number of SVO moves is > 0
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :return: Boolean for resolution of heuristic
    """
    try:
        ooo_list = row.number_moved_words_ooo.replace('[', '').replace(']', '').split(', ')
    except AttributeError:
        return row.num_svo_ooo > 0
    if len(ooo_list[0]) > 0:
        return len(ooo_list) + row.num_svo_ooo > 0
    else:
        return row.num_svo_ooo > 0


def decide_sentence_split(row: Union[NamedTuple, pd.Series]):
    """
    Decide on the sentence splitting (SPLIT) action.
    Heuristic - if the number of sentence splitting is > 0.
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :return: Boolean for resolution of heuristic
    """
    return row.sentence_split > 0


def write_to_text_file(row: NamedTuple, file: TextIO):
    """
    Write a dataset row to a text file.
    The format is: <SIM> {regular sentence} <SEP> {simple sentece} </SIM>, where the relevant actions are prepended
    to the start of {regular sentence}.
    we don't right Deletion or Addition alignments (align type >= 5)
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :param file: the TextIO object to a text file
    :return:
    """
    if row.entry_type_num >= 5:
        return
    file.write('<SIM> ')
    if decide_proximation(row):
        file.write('<PROX> ')
    if decide_rephrasing(row):
        file.write('<REPHRASE> ')
    if decide_deleting_info(row):
        file.write('<DEL> ')
    if decide_adding_info(row):
        file.write('<ADD> ')
    if decide_example(row):
        file.write('<EXAMPLE> ')
    if decide_explanation(row):
        file.write('<EXPLAIN> ')
    if decide_explicitation(row):
        file.write('<EXPLICIT> ')
    if decide_intra_sentence_rearrange(row):
        file.write('<REORDER> ')
    if decide_sentence_split(row):
        file.write('<SPLIT> ')
    file.write(f"{row.reg_sent} <SEP> {row.sim_sent} </SIM>\n")


def row_to_json(row: Union[NamedTuple, pd.Series], corpus: str):
    """
    Convert a data set row to a dictionary format to write as a json.
    The json contains the following keys: {"task","entry_type","corpus","source","target", "actions"}
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :param corpus: The corpus from which the row came from. Calculated from the filename in an external function.
    :return:
    """
    entry = {"task": "simplification",
             "entry_type": row.entry_type_num,
             "corpus": corpus,
             "source": row.reg_sent,
             "target": row.sim_sent,
             "actions": []}
    if row.entry_type_num >= 5:
        return {}
    if decide_proximation(row):
        entry["actions"].append("PROX")
    if decide_rephrasing(row):
        entry["actions"].append("REPHRASE")
    if decide_deleting_info(row):
        entry["actions"].append("DEL")
    if decide_adding_info(row):
        entry["actions"].append("ADD")
    if decide_example(row):
        entry["actions"].append("EXAMPLE")
    if decide_explanation(row):
        entry["actions"].append("EXPLAIN")
    if decide_explicitation(row):
        entry["actions"].append("EXPLICIT")
    if decide_intra_sentence_rearrange(row):
        entry["actions"].append("REORDER")
    if decide_sentence_split(row):
        entry["actions"].append("SPLIT")
    return entry


def write_to_json(row: Union[NamedTuple, pd.Series], file: TextIO, corpus: str):
    """
    Write a dataset row as a json row to a file.
    :param row: pandas DataFrame row returned from DataFrame.itertuples() method
    :param file: the file to which the jsons rows are written to.
    :param corpus: The corpus from which the row came from. Calculated from the filename in an external function.
    :return:
    """
    entry = row_to_json(row, corpus)
    if len(entry) == 0:
        return
    file.write(f"{json.dumps(entry)}\n")


if __name__ == '__main__':  # TODO: Cleanup comment outs
    parser = argparse.ArgumentParser("Create the final dataset formats for pre-analyzed CSVs.\n"
                                     "Can create both txt files and json files.\n"
                                     "Requires pre-analyzed datasets saved in csv format in a single directory")
    parser.add_argument('--dataset_path', type=str, required=True, help="Directory where datasets are saved")
    parser.add_argument('--output_path', type=str, required=True, help="Output directory")
    parser.add_argument('--output_type', type=str, default="both", choices=['both', 'txt', 'json'])

    args = parser.parse_args()

    # dataset_csv_path = pathlib.Path("/Users/eytan.chamovitz/PycharmProjects/CogSimp/csvs")
    dataset_csv_path = pathlib.Path(args.dataset_path)
    # outpath = "/Users/eytan.chamovitz/PycharmProjects/CogSimp/data"
    # outpath = args.output_path

    pathlib.Path(f"{args.output_path}/text_files").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{args.output_path}/jsons").mkdir(parents=True, exist_ok=True)

    for f in tqdm.tqdm(dataset_csv_path.iterdir()):
        if f.is_file() and f.suffix == ".csv":
            df = pd.read_csv(f.as_posix(), delimiter=';')
            f_name = f.stem.split("+")[0]
            with open(args.output_path + f"/text_files/{f_name}.txt", "w") as o, \
                    open(args.output_path + f"/jsons/{f_name.replace('-','_',1)}.json", "w") as j:
                for r in df.itertuples():
                    if args.output_type == "both" or args.output_type == "txt":
                        write_to_text_file(r, o)
                    if args.output_type == "both" or args.output_type == "json":
                        write_to_json(r, j, f_name.replace('-', '_', 1).split('-')[0])
