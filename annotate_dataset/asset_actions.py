#from easse.annotation import *
#from easse.annotation.word_level import *
# import easse.annotation.word_level as wl
# import spacy
# from collections import Counter
import os
from typing import TextIO
# import pandas as pd
# import os
import pyter
# from analyze_ops import *
from annotate_dataset.classify_datasets import *
from easse.aligner.aligner import MonolingualWordAligner, STANFORD_CORENLP_DIR, download_stanford_corenlp, CoreNLPClient
from easse.aligner.corenlp_utils import format_parser_output, join_parse_result, split_parse_result
from annotate_dataset.ops_classify_decisions import row_to_json


def syntactic_parse_texts(
    texts: List[str], tokenize=False, sentence_split=False, verbose=False,
    with_constituency_parse=False, client=None
):
    corenlp_annotators = [
        "tokenize",
        "ssplit",
        "pos",
        "lemma",
        "ner",
        "depparse",
    ]
    if with_constituency_parse:
        corenlp_annotators.append("parse")
    annotators_properties = {
        "tokenize.whitespace": not tokenize,
        "ssplit.eolonly": not sentence_split,
        "depparse.model": "edu/stanford/nlp/models/parser/nndep/english_SD.gz",
        "outputFormat": "json",
    }
    if not STANFORD_CORENLP_DIR.exists():
        download_stanford_corenlp()
    os.environ["CORENLP_HOME"] = str(STANFORD_CORENLP_DIR)

    parse_results = []

    if client is None:
        with CoreNLPClient(
            annotators=corenlp_annotators,
            properties=annotators_properties,
            threads=40, be_quiet=True
        ) as client:
            for text in tqdm(texts, disable=(not verbose)):
                if isinstance(text, List):
                    text = " ".join(text)
                raw_parse_result = client.annotate(text)
                parse_result = format_parser_output(raw_parse_result["sentences"])

                if len(parse_result["sentences"]) > 1 and not sentence_split:
                    parse_result = join_parse_result(parse_result)
                elif sentence_split:
                    parse_result = split_parse_result(parse_result["sentences"])

                parse_results.append(parse_result)
    else:
        for text in tqdm(texts, disable=(not verbose)):
            if isinstance(text, List):
                text = " ".join(text)
            raw_parse_result = client.annotate(text)
            parse_result = format_parser_output(raw_parse_result["sentences"])

            if len(parse_result["sentences"]) > 1 and not sentence_split:
                parse_result = join_parse_result(parse_result)
            elif sentence_split:
                parse_result = split_parse_result(parse_result["sentences"])

            parse_results.append(parse_result)

    return parse_results


def get_tokenized_and_parsing_from_file(file: TextIO, nlp, nlpclient):
    print("Tokenizing...")
    tokenized = [" ".join(token.text for token in nlp(sent.strip())) for sent in file.readlines()]

    print("Syntactic Parsing...")
    parses = syntactic_parse_texts(tokenized, client=nlpclient)
    return tokenized, parses


def get_tokenized_and_parsing_from_list(sent_list, nlp, nlpclient):
    print("Tokenizing...")
    tokenized = [" ".join(token.text for token in nlp(sent.strip())) for sent in sent_list]

    print("Syntactic Parsing...")
    parses = syntactic_parse_texts(tokenized, verbose=True, client=nlpclient)
    return tokenized, parses


def get_word_level(reg_tokenized, reg_parses, sim_tokenized, sim_parses, aligner: MonolingualWordAligner, index=None):
    result = []
    for ind, reg_sent_tok, sim_sent_tok, orig_p, ref_p in tqdm(zip(range(index-len(reg_tokenized), index), reg_tokenized, sim_tokenized, reg_parses, sim_parses),
                                                                 total=len(reg_tokenized)):
        reg_count = {}
        sim_count = {}
        reg_auto_labels = []
        sim_auto_labels = []
        try:
            word_align = aligner.get_word_aligns(orig_p, ref_p)[0]
            reg_annots, sim_annots = annotate_sentence(reg_sent_tok.split(), sim_sent_tok.split(),
                                                       word_align, orig_p, ref_p)
            reg_auto_labels = wl._from_annots_to_labels(reg_annots, ORIG_OPS_LABELS, 'C')
            sim_auto_labels = wl._from_annots_to_labels(sim_annots, ['A', 'D', 'M', 'R', 'C'], 'C')
            reg_count = dict(Counter(reg_auto_labels).items())
            sim_count = dict(Counter(sim_auto_labels).items())
        except IndexError:
            if len(reg_sent_tok) == 0:
                reg_count = {}
                sim_count = {'A': len(sim_sent_tok.split())}
                reg_auto_labels = []
                sim_auto_labels = ['A' for word in sim_sent_tok.split()]
            elif len(sim_sent_tok) == 0:
                reg_count = {'D': len(reg_sent_tok.split())}
                sim_count = {}
                reg_auto_labels = ['D' for word in reg_sent_tok.split()]
                sim_auto_labels = []
        except TimeoutError:
            reg_count = {}
            sim_count = {}
            reg_auto_labels = []
            sim_auto_labels = []
            print(f"TimeoutError in line {ind}")
        finally:
            result.append((reg_count, sim_count, reg_auto_labels, sim_auto_labels))
    return pd.DataFrame(result, columns=["reg_word_label_counts", "sim_word_label_counts", "reg_auto_labels",
                                         "sim_auto_labels"])


# TODO: Find the correct get entry-type function.
def get_entry_type(reg_doc, sim_doc):
    """
    Get the alignment type of 2 documents. Assumes that there are two documents with at least 1 sentence in each.
    :param reg_doc: the regular spacy document.
    :param sim_doc: the regular spacy document
    :return: 1 if alignment is 1-to-1. 2 if alignment is 1-to-N. 3 if alignment is M-to-1, and 4 if alignment is M-to-N
    """
    reg_sent_num = len([s for s in reg_doc.sents])
    sim_sent_num = len([s for s in sim_doc.sents])
    if reg_sent_num > 1:
        return 4 if sim_sent_num > 1 else 3
    else:
        return 2 if sim_sent_num > 1 else 1


# TODO: Chose which initial DF function to use.
def get_initial_wiki_dataframe(reg_sents, sim_sents, nlp):
    """
    Create the initial dataframe for analysis.
    Contains the document title, alignment of sentences,  regular sentence, simple sentence, the Entry Type analysis,
    and TER score.
    :param reg_file: the file containing the sentences from the regular document
    :param sim_file: the file containing the sentences from the simplified document
    :param nlp: a Spacy nlp model to parse the sentences.
    :return: Dataframe with relevant columns
    """
    result = []
    for reg_line, sim_line in zip(reg_sents, sim_sents):
        reg_doc = nlp(reg_line.strip())
        sim_doc = nlp(sim_line.strip())
        try:
            ter_score = pyter.ter([token.text for token in reg_doc], [token.text for token in sim_doc])
        except TimeoutError:
            print(f"Timeout error in creating initial df.\n\tDocs: \n\t\t{reg_line}\n\t\t{sim_line}")
            ter_score = 0
        result.append({
            'doc_title': "wiki-auto",
            'alignment': "",
            'entry_type': get_entry_type(reg_doc, sim_doc),
            'reg_sent': reg_line.strip(),
            'sim_sent': sim_line.strip(),
            'TER_score': ter_score
        })
    return pd.DataFrame(result)


def get_initial_dataframe(reg_file: TextIO, sim_file: TextIO, nlp):
    """
    Create the initial dataframe for analysis.
    Contains the document title, alignment of sentences,  regular sentence, simple sentence, the Entry Type analysis,
    and TER score.
    :param reg_file: the file containing the sentences from the regular document
    :param sim_file: the file containing the sentences from the simplified document
    :param nlp: a Spacy nlp model to parse the sentences.
    :return: Dataframe with relevant columns
    """
    result = []
    for reg_line, sim_line in zip(reg_file.readlines(), sim_file.readlines()):
        reg_doc = nlp(reg_line.strip())
        sim_doc = nlp(sim_line.strip())
        result.append({
            'doc_title': "asset",
            'alignment': "",
            'entry_type': get_entry_type(reg_doc, sim_doc),
            'reg_sent': reg_line.strip(),
            'sim_sent': sim_line.strip(),
            'TER_score': pyter.ter([token.text for token in reg_doc],
                                   [token.text for token in sim_doc])
        })
    return pd.DataFrame(result)


def create_datasets():
    print("Loading Spacy model")
    NLP = spacy.load("en_core_web_lg")
    word_aligner = MonolingualWordAligner()
    data_path = "/Users/eytan.chamovitz/PycharmProjects/CogSimp/asset/dataset"
    annotators = range(10)
    split_names = ["test", "valid"]
    for split in split_names:
        orig_filename = f"asset.{split}.orig"
        orig_filepath = f"{data_path}/{orig_filename}"
        with open(orig_filepath, "r") as f:
            orig_tokenized, orig_parses = get_tokenized_and_parsing_from_file(f, NLP)
        for a in annotators:
            simp_filename = f"asset.{split}.simp.{a}"
            simp_filepath = f"{data_path}/{simp_filename}"
            with open(orig_filepath, "r") as orig, open(simp_filepath, "r") as simp:
                df = get_initial_dataframe(orig, simp, NLP)
            with open(simp_filepath, "r") as f:
                simp_tokenized, simp_parses = get_tokenized_and_parsing_from_file(f, NLP)
            res = get_word_level(orig_tokenized, orig_parses, simp_tokenized, simp_parses, word_aligner)
            pd.concat([df, res], axis=1).to_csv(
                f"/Users/eytan.chamovitz/PycharmProjects/CogSimp/csvs/asset/asset-annotator{a}+word_level.csv",
                sep=';', index=False)


def create_wiki_datasets(batch=4000, start=0):
    print("Loading Spacy model")
    NLP = spacy.load("en_core_web_lg")
    word_aligner = MonolingualWordAligner()
    data_path = "/Users/eytan.chamovitz/Documents/GitHub/wiki-auto/wiki-auto/GEM2021/full_with_split"
    split_names = ["train"]

    corenlp_annotators = [
        "tokenize",
        "ssplit",
        "pos",
        "lemma",
        "ner",
        "depparse",
    ]
    # if with_constituency_parse:
    #     corenlp_annotators.append("parse")
    annotators_properties = {
        "tokenize.whitespace": not False,
        "ssplit.eolonly": not False,
        "depparse.model": "edu/stanford/nlp/models/parser/nndep/english_SD.gz",
        "outputFormat": "json",
    }
    if not STANFORD_CORENLP_DIR.exists():
        download_stanford_corenlp()
    os.environ["CORENLP_HOME"] = str(STANFORD_CORENLP_DIR)

    with CoreNLPClient(
            annotators=corenlp_annotators,
            properties=annotators_properties,
            threads=40, be_quiet=True
    ) as client:
        time.sleep(client.CHECK_ALIVE_TIMEOUT+30)
        for split in split_names:
            with open(f"{data_path}/{split}.tsv", "r") as f:
                reg_sents = []
                sim_sents = []
                print(f"Reading from {split}.tsv, batch {start}-{start+batch-1}")
                for i, line in enumerate(f.readlines()):
                    if i < start:
                        continue
                    if i > 40000:
                        break
                    if i % batch == 0 and i != start:
                        # save file for future reference and run everything
                        orig_tokenized, orig_parses = get_tokenized_and_parsing_from_list(reg_sents, NLP, nlpclient=client)
                        simp_tokenized, simp_parses = get_tokenized_and_parsing_from_list(sim_sents, NLP, nlpclient=client)
                        print("Creating initial data frame")
                        df = get_initial_wiki_dataframe(reg_sents, sim_sents, NLP)
                        print("Adding word level")
                        res = get_word_level(orig_tokenized, orig_parses, simp_tokenized, simp_parses, word_aligner, index=i)
                        print("Concating and Saving dataframe")
                        pd.concat([df, res], axis=1).to_csv(f"/Users/eytan.chamovitz/PycharmProjects/CogSimp/wiki-auto/wiki-auto_{i-batch}-{i-1}.csv")
                        reg_sents = []
                        sim_sents = []
                        print(f"Reading from {split}.tsv, batch {i}-{i+batch-1}")
                    reg, sim = line.split('\t')
                    reg_sents.append(reg)
                    sim_sents.append(sim)
                if len(reg_sents) > 0:
                    orig_tokenized, orig_parses = get_tokenized_and_parsing_from_list(reg_sents, NLP, nlpclient=client)
                    simp_tokenized, simp_parses = get_tokenized_and_parsing_from_list(sim_sents, NLP, nlpclient=client)
                    print("Creating initial data frame")
                    df = get_initial_wiki_dataframe(reg_sents, sim_sents, NLP)
                    print("Adding word level")
                    res = get_word_level(orig_tokenized, orig_parses, simp_tokenized, simp_parses, word_aligner, index=i)
                    print("Concating and Saving dataframe")
                    pd.concat([df, res], axis=1).to_csv(
                        f"/Users/eytan.chamovitz/PycharmProjects/CogSimp/wiki-auto/wiki-auto_{i-len(reg_sents)}-{i}.csv")
            start = 0


def prepare_df(splits=('test', 'valid'), annotators=range(10),
               source_folder="/Users/eytan.chamovitz/PycharmProjects/CogSimp/csvs/asset"):
    data_path = pathlib.Path("/Users/eytan.chamovitz/PycharmProjects/CogSimp/csvs")
    datasets = [f"asset-{split}-annotator{a}" for split in splits for a in annotators]
    LOGGER.warning("Loading PPDB...")
    # ppdb_path = '/Users/eytan.chamovitz/PycharmProjects/NLP_HW/project/DHLS/SPPDB_lexicon.json'
    ppdb_path = "/Users/eytan.chamovitz/PycharmProjects/CogSimp/simple_ppdbs/en_sppdb.json"
    with open(ppdb_path, 'r') as f:
        ppdb = json.load(f)

    LOGGER.warning("Loading Spacy model...")
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe('coreferee')

    if lang == 'en':
        with open(spacy_lookups_data.en["lexeme_prob"]) as f:
            lex_probability = json.load(f)
    else:
        lex_probability = {}

    for dataset_name in datasets[-2:]:
        classify_dataset(dataset_name, data_path, nlp=nlp, lex_probability=lex_probability, ppdb=ppdb,
                         dataset_type="+word_level", source_folder=source_folder)


def prepare_wiki_df(source_folder="/Users/eytan.chamovitz/PycharmProjects/CogSimp/wiki-auto"):
    data_path = pathlib.Path("/Users/eytan.chamovitz/PycharmProjects/CogSimp/csvs/wiki-auto")
    datasets = []
    for file in pathlib.Path(source_folder).iterdir():
        if file.is_file() and file.suffix == ".csv":
            datasets.append(file.stem)
    LOGGER.warning("Loading PPDB...")
    # ppdb_path = '/Users/eytan.chamovitz/PycharmProjects/NLP_HW/project/DHLS/SPPDB_lexicon.json'
    ppdb_path = "/Users/eytan.chamovitz/PycharmProjects/CogSimp/simple_ppdbs/en_sppdb.json"
    with open(ppdb_path, 'r') as f:
        ppdb = json.load(f)

    LOGGER.warning("Loading Spacy model...")
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe('coreferee')

    if lang == 'en':
        with open(spacy_lookups_data.en["lexeme_prob"]) as f:
            lex_probability = json.load(f)
    else:
        lex_probability = {}

    for dataset_name in datasets:
        classify_dataset(dataset_name, data_path, sep=',', nlp=nlp, lex_probability=lex_probability, ppdb=ppdb,
                         dataset_type="", source_folder=source_folder)


def classify_wiki(ds_name="wiki-auto", splits=('valid', 'train')):
    dataset_csv_path = pathlib.Path("/Users/eytan.chamovitz/PycharmProjects/CogSimp/csvs/wiki-auto")
    outpath = "/Users/eytan.chamovitz/PycharmProjects/CogSimp/data"
    for split in splits:
        main_name = f"{ds_name}-{split}"
        f_name = f"{ds_name}_{split}-combined+revised.csv"
        print(f"Reading Dataframe {f_name}")
        f_name = (dataset_csv_path / f_name).as_posix()
        dataframe = pd.read_csv(f_name, delimiter=';')
        with open(outpath + f"/jsons/{main_name}.json", "a") as j:
            for row in tqdm(dataframe.itertuples(), total=len(dataframe)):
                entry = row_to_json(row, main_name)
                j.write(f"{json.dumps(entry)}\n")


# TODO: Before publication of datasets, create format which is raw for everything by reference
def classify(ds_name="asset", splits=('test', 'valid'), annotators=range(10)):
    dataset_csv_path = pathlib.Path("/Users/eytan.chamovitz/PycharmProjects/CogSimp/csvs")
    outpath = "/Users/eytan.chamovitz/PycharmProjects/CogSimp/data"
    for split in splits:
        main_name = f"{ds_name}-{split}"
        f_names = [f"{main_name}-annotator{a}" for a in annotators]
        paths = [f"{dataset_csv_path.as_posix()}/{f_name}+revised.csv" for f_name in f_names]
        dataframes = [pd.read_csv(p, delimiter=';') for p in paths]
        with open(outpath + f"/jsons/{main_name}.json", "w") as j:
            for rows in zip(*(df.itertuples() for df in dataframes)):
                entries = [row_to_json(row, main_name) for row in rows]
                action_counter = Counter()
                align_counter = Counter()
                for ent in entries:
                    action_counter.update(ent["actions"])
                    align_counter.update([ent["entry_type"]])
                good_actions = [k for k, v in action_counter.items() if v >= 5]
                if len(good_actions) == 0:
                    good_actions = [k for k, v in action_counter.most_common(2)]
                entry = entries[0]
                entry["action"] = good_actions
                entry["target"] = [e["target"] for e in entries]
                entry["entry_type"] = align_counter.most_common(1)[0][0]
                j.write(f"{json.dumps(entry)}\n")


if __name__ == '__main__':  # TODO: Cleanup comment outs
    # input_file = "/Users/eytan.chamovitz/PycharmProjects/CogSimp/asset/test_file"
    # input_file2 = "/Users/eytan.chamovitz/PycharmProjects/CogSimp/asset/test_file"

    # create_datasets()
    # create_wiki_datasets(500, 39000)
    # prepare_wiki_df()
    classify_wiki()
