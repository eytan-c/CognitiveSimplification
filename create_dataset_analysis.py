from logging.handlers import RotatingFileHandler

from util.logger import setup_logging
from annotate_dataset.ops_classify_decisions import *
import pandas as pd
import logging

# TO-DO: Add usage description
# TO-DO: Clean-up comment outs
# TO-DO: Add different file saving types

"""
    Sorting fields:
        ;doc_title;entry_type_num;
    Info fields:
        reg_sent;sim_sent;
    Numerical fields -> for average:
        ter;
        percent_deleted_unused;
        token_length_ratio;
        percent_added_unused;
        nbchars_ratio;reg_nbchars;sim_nbchars;levsim;wordrank_ratio;reg_wordrank;sim_wordrank;deptree_depth_ratio;
        Ready for copying:
            ["ter", "percent_deleted_unused", "percent_added_unused", 
            "token_length_ratio", "nbchars_ratio", "reg_nbchars", "sim_nbchars", 
            "levsim", "wordrank_ratio", "reg_wordrank", "sim_wordrank", 
            "deptree_depth_ratio"]
    Numerical fields -> counts, for ____:
        verb_tense_change;
        simple_synonym;word2phrase;phrase2word;phrase2phrase;
        pron_explicitation - negative if the target has more pronouns than the source.
        num_svo_ooo;
        Ready for copying:
            ["verb_tense_change", "simple_synonym", "word2phrase", "phrase2word", "phrase2phrase", 
            "pron_explicitation", "num_svo_ooo"]
    Dict fields, for joint comparison:
        reg_label_count;sim_label_count;
    List fields:
        reg_labels;sim_labels;
        reg_prox_used;sim_prox_used;
        source_used;target_used;
        number_moved_words_ooo;
        reg_depth;sim_depth
    "Label" fields:
        pov_change - 0 no change, 1 sim has closer person, -1 reg has closer person
        passive_active_sub -  0 no change, 1 sim is active reg is passive, -1 sim is passive reg is active
        deleting_info - 1 yes, 0 no
        removal - 1 yes, 0 no
        summarization - 1 yes, 0 no
        is_adding - 1 yes, 0 no
        examples - 1 yes, 0 no
        explanations - 1 yes, 0 no
        sentence_split - 1 yes, 0 no
        deptree_type - 0 ADDITION/DELETION aligns, 1 same number of sentences in reg and sim, 2 reg has more sentences, 3 reg has less sentences
"""

_NUMERICAL_COLUMNS = ["ter", "percent_deleted_unused", "percent_added_unused", "token_length_ratio",
                     "nbchars_ratio", "reg_nbchars", "sim_nbchars", "levsim",
                     "wordrank_ratio", "reg_wordrank", "sim_wordrank", "deptree_depth_ratio"]

_COUNT_COLUMNS = ["verb_tense_change", "simple_synonym", "word2phrase", "phrase2word",
                  "phrase2phrase", "pron_explicitation", "num_svo_ooo"]


def get_decision_dict(row: Union[pd.Series, NamedTuple]):
    return pd.Series(data={
        "PROX": 1 if decide_proximation(row) else 0,
        "REPHRASE": 1 if decide_rephrasing(row) else 0,
        "DEL": 1 if decide_deleting_info(row) else 0,
        "ADD": 1 if decide_adding_info(row) else 0,
        "EXAMPLE": 1 if decide_example(row) else 0,
        "EXPLAIN": 1 if decide_explanation(row) else 0,
        "EXPLICIT": 1 if decide_explicitation(row) else 0,
        "REORDER": 1 if decide_intra_sentence_rearrange(row) else 0,
        "SPLIT": 1 if decide_sentence_split(row) else 0
    })


def analyze_dataset(dataframe: pd.DataFrame, ops_df: pd.DataFrame, get_per_entry: bool = False):
    if get_per_entry:
        logger.info(f"Analyzing numerical columns {_NUMERICAL_COLUMNS}")
        aggreg = dataframe[_NUMERICAL_COLUMNS].groupby(dataframe['entry_type_num']).agg(['mean', 'median', 'std'])
        logger.info(f"Analyzing count columns {_COUNT_COLUMNS}")
        counts_agg = dataframe[_COUNT_COLUMNS].groupby(dataframe['entry_type_num']).agg(["sum", "mean", "median", "std"])
        logger.info(f"Analyzing operations")
        ops_agg = ops_df.groupby(ops_df['entry_type_num']).agg(["sum", "mean", "median", "std"])
        logger.info(f"Analyzing entry types")
        entry_agg = dataframe['entry_type_num'].groupby(dataframe['entry_type_num']).agg(['count'])
        entry_agg['%'] = entry_agg['count'] / entry_agg['count'].sum()
        logger.info("Finished Analyzing per Entry Type")
        return pd.concat([entry_agg, ops_agg, counts_agg, aggreg], axis=1)
    else:
        logger.info(f"Analyzing numerical columns {_NUMERICAL_COLUMNS}")
        aggreg = dataframe[_NUMERICAL_COLUMNS].agg(["mean", "median", "std"])
        logger.info(f"Analyzing count columns {_COUNT_COLUMNS}")
        counts_agg = dataframe[_COUNT_COLUMNS].agg(["sum", "mean", "median", "std"])
        logger.info(f"Analyzing operations")
        ops_agg = ops_df.agg(["sum", "mean", "median", "std"])
        logger.info("Finished Analyzing full dataframe")
        return pd.concat([ops_agg, counts_agg, aggreg], axis=1)


def dataset_analysis(ds_name: str, data_path: str, out_path: str):
    fpath = f"{data_path}/{ds_name}.csv"
    logger.info(f"Opening DF from {fpath}")
    df = pd.read_csv(fpath, sep=';', encoding='utf-8')
    logger.debug(f"Replacing booleans with 0-1 values")
    df = df.replace({True: 1, False: 0})
    logger.debug("Resetting errors in PROX classification for ADDITIONs and DELETIONs")
    df.loc[df["entry_type_num"] >= 5, ["pov_change", "verb_tense_change", "passive_active_sub"]] = 0
    logger.info("Creating Operations dataframe")
    ops_decide_df = df.apply(get_decision_dict, axis=1)
    ops_decide_df = pd.concat([df["entry_type_num"], ops_decide_df], axis=1)
    logger.info("Fixing ADD classification errors in DELETE alignments")
    ops_decide_df["ADD"][ops_decide_df["entry_type_num"] == 5] = 0
    logger.info("Saving Operations dataframe")
    ops_decide_df.to_csv(f"{outpath}/csvs/{ds_name}-ops.csv", sep=";")
    logger.info("Starting full corpus analysis")
    cc1 = analyze_dataset(df, ops_decide_df)
    cc1.to_csv(f"{outpath}/csvs/{ds_name}-full-analysis.csv", sep=";")
    with open(f"{out_path}/{ds_name}__analysis.txt", "w") as f:
        f.write("$%$% Full Dataset $%$%\n")
        dfAsString = cc1.to_string()
        f.write(dfAsString)
        f.write("\n\n")
    logger.info("Starting analysis per Entry Type")
    cc2 = analyze_dataset(df, ops_decide_df, get_per_entry=True)
    cc2.to_csv(f"{outpath}/csvs/{ds_name}-entry-analysis.csv", sep=";")
    with open(f"{out_path}/{ds_name}__analysis.txt", "a") as f:
        f.write("$%$% Per Entry $%$%\n")
        dfAsString = cc2.to_string()
        f.write(dfAsString)
        f.write("\n\n")
    logger.info("Starting analysis no DELETION and ADDITION SIs")
    cc3 = analyze_dataset(df[df["entry_type_num"] < 5], ops_decide_df[ops_decide_df["entry_type_num"] < 5])
    cc3.to_csv(f"{outpath}/csvs/{ds_name}-no-5-6-analysis.csv", sep=";")
    with open(f"{out_path}/{ds_name}__analysis.txt", "a") as f:
        f.write("$%$% Full Dataset (no DELETION + ADDITION SIs) $%$%\n")
        dfAsString = cc3.to_string()
        f.write(dfAsString)
        f.write("\n\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Analyze simplification datasets with regards to use of Cognitive Simplification Operations.\n"
                                     "Outputs 3 types of files:\n"
                                     "\t1. <dataset_name>-ops.csv : a CSV that saves which operations are used in which SI in the dataset.\n"
                                     "\t2. <ds_name>__analysis.txt : a text output of the full analysis.\n"
                                     "\t3. <ds_name>-<full|entry|no-5-6>-analysis.csv : files that save the statistics for each operation in different ways.\n")
    parser.add_argument("--datapath", default="./data/base_datasets")
    parser.add_argument("--outpath", default="./data/dataset_analysis")

    args = parser.parse_args()

    setup_logging("test")

    logger = logging.getLogger()

    datapath = pathlib.Path(args.datapath)
    outpath = pathlib.Path(args.outpath)

    for file in datapath.iterdir():
        if file.is_dir() or file.stem.startswith("."):
            continue
        dataset_analysis(file.stem, datapath, outpath)

