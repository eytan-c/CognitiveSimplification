import argparse

from annotate_dataset.analyze_ops import *
import pandas as pd
import pathlib
import coreferee
import coreferee_model_en
from typing import Match

SPACE_RE = re.compile(r"( ){2,}")


def replace_multi_space(matchobj: Match):
    return matchobj.group(1)


def create_record(doc_title, alignment, entry_type_num, reg_sent, sim_sent, ter, reg_label_count, sim_label_count,
                  reg_labels, sim_labels, pov_change, verb_tense_change, passive_active_sub, reg_prox_used,
                  sim_prox_used, simple_synonym, word2phrase, phrase2word, phrase2phrase, source_used, target_used,
                  deleting_info, removal, summarization, percent_deleted_unused, token_length_ratio, is_adding,
                  examples, explanations, percent_added_unused, pron_explicitation, number_moved_words_ooo,
                  num_svo_ooo, sentence_split, nbchars_ratio, reg_nbchars, sim_nbchars, levsim, wordrank_ratio,
                  reg_wordrank, sim_wordrank, deptree_depth_ratio, deptree_type, reg_depth, sim_depth):
    """
    Create dictionary from different parameters, to be added as a row to the output of the dataset.
    :param doc_title: the document from which the sentence originated
    :param alignment: alignements between the sentences in the document
    :param entry_type_num: type of alignment (see AlignTypes.py)
    :param reg_sent: the source sentence (regular language)
    :param sim_sent: the target sentence (simpleified language)
    :param ter: TER score
    :param reg_label_count: the number of different ADD, KEEP, DELETE labels, saved as dictionary, for the source sentence
    :param sim_label_count: the number of different ADD, KEEP, DELETE labels, saved as dictionary, for the target sentence
    :param reg_labels: the labels of each token in the source, as a list
    :param sim_labels: the labels of each token in the target, as a list
    :param pov_change: result of the __ function
    :param verb_tense_change: result of the __ function
    :param passive_active_sub: result of the __ function
    :param reg_prox_used: result of the __ function
    :param sim_prox_used: result of the __ function
    :param simple_synonym: result of the __ function
    :param word2phrase: result of the __ function
    :param phrase2word: result of the __ function
    :param phrase2phrase: result of the __ function
    :param source_used: result of the __ function
    :param target_used: result of the __ function
    :param deleting_info: result of the __ function
    :param removal: result of the __ function
    :param summarization: result of the __ function
    :param percent_deleted_unused: result of the __ function
    :param token_length_ratio: result of the __ function
    :param is_adding: result of the __ function
    :param examples: result of the __ function
    :param explanations: result of the __ function
    :param percent_added_unused: result of the __ function
    :param pron_explicitation: result of the __ function
    :param number_moved_words_ooo: result of the __ function
    :param num_svo_ooo: result of the __ function
    :param sentence_split: result of the __ function
    :param nbchars_ratio: result of the __ function
    :param reg_nbchars: result of the __ function
    :param sim_nbchars: result of the __ function
    :param levsim: result of the __ function
    :param wordrank_ratio: result of the __ function
    :param reg_wordrank: result of the __ function
    :param sim_wordrank: result of the __ function
    :param deptree_depth_ratio: result of the __ function
    :param deptree_type: result of the __ function
    :param reg_depth: result of the __ function
    :param sim_depth: result of the __ function
    :return:
    """
    items = locals().items()
    keys = locals().keys()
    res = {k: v for k, v in items if k in keys}
    res.pop('items', None)
    # res.pop('keys', None)
    return res


def classify_dataset(dataset: str, save_path: pathlib.Path, nlp, lex_probability, ppdb,
                     dataset_type="+actions+word_level", sep=';', lang='en',
                     source_folder="."):
    """
    Add operations analysis to dataset.
    Expects files to be saved in CSV format, with at least the following columns:
        'doc_title', 'alignment', 'entry_type', 'reg_sent', 'sim_sent', 'TER_score', 'reg_word_label_counts',
        'sim_word_label_counts', 'reg_auto_labels', 'sim_auto_labels'
    The files are usually saved in the format of {dataset_name}+actions+word_level.csv
    :param lang: language of the model to analyze
    :param dataset: the name of the dataset to analyze
    :param save_path: the path in which to save the analyzed dataset as a CSV
    :param nlp: the spacy model used to calculate parts of the actions
    :param lex_probability: lexical probability of words in the language anallyzed
    :param ppdb: simples paraphrase database as a dictionary
    :param dataset_type: the type of analyzed dataset, defaults to actions+word_level
    :param sep: the separator in which the dataset was saved as CSV
    :param source_folder: the path in which the original datasets are saved in
    :return:
    """
    LOGGER.warning(f"Loading dataset {dataset} DF...")

    data_file = pathlib.Path(f"{source_folder}/{dataset}{dataset_type}.csv")  # filepath to load
    old_df = pd.read_csv(data_file, sep=sep, on_bad_lines='skip')  # the old dataframe to analyze
    if 'alignment' not in old_df.columns:
        old_df['alignment'] = ["" for i in range(len(old_df))]
    old_df.fillna(value="", inplace=True)  # remove NaNs from the old dataframe

    LOGGER.warning(f"Classifying Dataset {dataset}...")
    records = []

    """for each row in the old dataframe"""
    for idx, doc_title, alignment, entry_type_num, \
        reg_sent, sim_sent, ter, \
        reg_label_count, sim_label_count, \
        reg_labels, sim_labels in tqdm(old_df[['doc_title', 'alignment', 'entry_type', 'reg_sent',
                                               'sim_sent', 'TER_score', 'reg_word_label_counts',
                                               'sim_word_label_counts', 'reg_auto_labels',
                                               'sim_auto_labels']].itertuples(), total=len(old_df)):
        # If there is not doc title - continue
        if doc_title == '':
            continue

        # Get the entry type
        entry_type = AlignmentTypes(entry_type_num)

        # convert source and targets to Spacy documents
        # reg_doc = nlp(reg_sent.strip())
        # sim_doc = nlp(sim_sent.strip())
        reg_doc = nlp(re.sub(SPACE_RE, replace_multi_space, reg_sent.strip()))
        sim_doc = nlp(re.sub(SPACE_RE, replace_multi_space, sim_sent.strip()))

        # If this is an ADD or DELETE entry -> skip them
        if len(sim_doc) == 0 and len(reg_doc) == 0:
            continue

        # Prepare variables for used tokens and token labels
        reg_used = []
        sim_used = []
        reg_labels = eval(reg_labels)
        sim_labels = eval(sim_labels)

        # Get the results of proximation and add to used tokens
        pov_change, verb_tense_change, passive_active_sub, \
            reg_prox_used, sim_prox_used = classify_proximation(reg_doc, sim_doc, lang)
        reg_used += reg_prox_used
        sim_used += sim_prox_used

        # Get the results of rephrasing
        simple_synonym, word2phrase, phrase2word, \
            phrase2phrase, source_used, target_used = classify_rephrasing(reg_doc, sim_doc, lang, ppdb, reg_labels)

        # Get the results of deleting
        deleting_info, removal, summarization, \
            percent_deleted_unused, token_length_ratio = classify_deleting_info(reg_doc, sim_doc, lang, entry_type,
                                                                                reg_labels, reg_used)
        # Get the results of adding
        is_adding, examples, \
            explanations, percent_added_unused = classify_adding_info(reg_doc, sim_doc, lang,
                                                                      entry_type, sim_labels, sim_used)

        # ￿Get the results of exlicitation
        pron_explicitation = classify_explicitation(reg_doc, sim_doc, lang, nlp, entry_type)

        # Get the results of intra-sentence rearrangement
        number_moved_words_ooo, num_svo_ooo = classify_intra_sentence_rearrange(reg_doc, sim_doc, lang,
                                                                                reg_labels, sim_labels)

        # Get results of sentence splitting
        sentence_split = classify_sentence_splitting(reg_doc, sim_doc, lang, entry_type)

        # Get ACCESS (Martin2020) results (nbchars, levinstein similarity, wordrank, dep-tree depth)
        nbchars_ratio, reg_nbchars, sim_nbchars = get_nbchars_ratio(reg_doc, sim_doc, entry_type)
        levsim = get_levsim(reg_doc, sim_doc, entry_type)
        wordrank_ratio, reg_wordrank, sim_wordrank = get_wordrank_ratio(reg_doc, sim_doc, lex_probability, entry_type)
        deptree_depth_ratio, deptree_type, reg_depth, sim_depth = get_deptree_depth_ratio(reg_doc, sim_doc, entry_type)

        # Add the record to the results (record is a dictionary)
        records.append(create_record(doc_title, alignment, entry_type_num, reg_sent, sim_sent, ter,
                                     reg_label_count, sim_label_count, reg_labels, sim_labels,
                                     pov_change, verb_tense_change, passive_active_sub, reg_prox_used, sim_prox_used,
                                     simple_synonym, word2phrase, phrase2word, phrase2phrase, source_used, target_used,
                                     deleting_info, removal, summarization, percent_deleted_unused, token_length_ratio,
                                     is_adding, examples, explanations, percent_added_unused,
                                     pron_explicitation, number_moved_words_ooo, num_svo_ooo, sentence_split,
                                     nbchars_ratio, reg_nbchars, sim_nbchars, levsim,
                                     wordrank_ratio, reg_wordrank, sim_wordrank,
                                     deptree_depth_ratio, deptree_type, reg_depth, sim_depth
                                     )
                       )
    # create new dataframe from records
    LOGGER.warning("Creating new DF...")
    new_df = pd.DataFrame.from_records(records)
    LOGGER.warning("Saving new DF...")
    new_df.to_csv((save_path / f"{dataset}+revised.csv").as_posix(), sep=';')
    LOGGER.warning("DONE")


if __name__ == "__main__":  # TO-DO: Cleanup comment outs

    parser = argparse.ArgumentParser("Create CSVs with all the additional metrics and values added for each entry.\n"
                                     "Expect Input datasets to have the columns: \'doc_title\', \'alignment\', "
                                     "\'entry_type\', \'reg_sent\', \'sim_sent\', \'TER_score\', "
                                     "\'reg_word_label_counts\',\'sim_word_label_counts\', "
                                     "\'reg_auto_labels\',\'sim_auto_labels\'\n")
    parser.add_argument('--ppdb_path', type=str, required=True, help="Directory where the simple paraphrase "
                                                                     "database is save.")
    parser.add_argument('--spacy_model', type=str, default="en_core_web_lg", help="Spacy Model name to load")
    parser.add_argument('--lang', type=str, default='en')  # TO-DO: Cleanup usage of the lang argument
    parser.add_argument('--dataset_type', type=str, default='+actions+word_level')
    parser.add_argument('--output_path', type=str, required=True, help="Output directory")
    parser.add_argument('--output_type', required=True, default="both", choices=['both', 'txt', 'json'])
    parser.add_argument('--datasets', required=True, nargs='+', help='The names of the datasets to analyze.')
    parser.add_argument('--source_folder', type=str, required=True)
    parser.add_argument('--sep', default=';', help='Delimiter for existing CSV files.')

    args = parser.parse_args()

    #  TO_DO: get correct SPPDB file
    LOGGER.warning("Loading PPDB...")

    with open(args.ppdb_path, 'r') as f:
        ppdb = json.load(f)

    # TO-DO: Think of ways to read word levels according to the tokenizing model
    LOGGER.warning("Loading Spacy model...")
    nlp = spacy.load(args.spacy_model)  # defaults to en_core_web_lg
    nlp.add_pipe('coreferee')

    if args.lang == 'en':
        with open(spacy_lookups_data.en["lexeme_prob"]) as f:
            lex_probability = json.load(f)
    else:
        lex_probability = {}

    data_path = pathlib.Path(args.output_path)

    for dataset_name in args.datasets:
        classify_dataset(dataset_name, data_path, nlp, lex_probability, ppdb, dataset_type=args.dataset_type,
                         sep=args.sep, lang=args.lang, source_folder=args.source_folder)
