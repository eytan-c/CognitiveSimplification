import os
import time

import pyter
from annotate_dataset.classify_datasets import *
from easse.aligner.aligner import MonolingualWordAligner, STANFORD_CORENLP_DIR, download_stanford_corenlp, CoreNLPClient
import easse.annotation.word_level as wl
from easse.aligner.corenlp_utils import format_parser_output, join_parse_result, split_parse_result
from annotate_dataset.ops_classify_decisions import row_to_json
from annotate_dataset.analyze_ops import *

from typing import Dict, List, Tuple, Union

import pandas as pd


def get_entry_type(pair: Tuple[List[str], List[str]]) -> int:
    num_reg_sent = len(pair[0])
    num_sim_sent = len(pair[1])
    if num_sim_sent == 0:
        return 5
    elif num_reg_sent == 0:
        return 6
    elif num_reg_sent > 1:
        return 4 if num_sim_sent > 1 else 3
    else:
        return 2 if num_sim_sent > 1 else 1


def reset_doc() -> Tuple[str, str, int]:
    return "", "", 0


def get_doc_title(line: str) -> str:
    m = re.match(r"#+\s+([^#]+)", line)
    return m.group(1).strip()


def get_sub_doc_id(line: str) -> str:
    m = re.match(r">>>>>\s*(\d+)", line)
    if m is None:
        raise AttributeError()
    return m.group(1)


def split_sentences(line) -> List[str]:
    s = re.split(r"; \(|\D\. ", line.strip())
    if len(s) == 1 and len(s[0]) == 0:
        return []
    else:
        return s


def syntactic_parse_texts(
    texts: List[str],
    tokenize=False,
    sentence_split=False,
    verbose=False,
    with_constituency_parse=False,
    client=None
):
    parse_results = []

    if client is None:
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


def get_tokenized_and_parsing_from_list(sent_list, nlp, nlpclient):
    print("Tokenizing...")
    tokenized = [" ".join(token.text for token in nlp(sent.strip())) for sent in tqdm(sent_list)]

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


def read_files(regular_file: Union[str, pathlib.Path], simple_file: Union[str, pathlib.Path]) -> List[Dict[str, Union[str, int]]]:
    """
    Reads two aligned txt files. Assumes that lines numbers are aligned between regular_file and simple_file.
    E.g. regular_file:10 would match simple_file:10.
    Also assumes particular format of alignment files - Each documents starts with a row:
    ############### <DOC_TITLE> ###############
    and ends with a row:
    ############### END <DOC_TITLE> ###############
    Each document also must contain at least 1 sub-document (chapter, sections, etc.)
    The start of each is marked with a row:
    '>>>>> <sub_doc_number>'
    """
    doc_title = ""
    sub_doc_id = ""
    sent_num = 0
    idx_r, idx_s = 0, 0
    result = []
    # 'doc_title;sub_doc_id;sentence_num;entry_type;reg_sent;sim_sent\n' #
    with open(regular_file, "r") as rfile, open(simple_file, "r") as sfile:
        while True:
            sent_num += 1
            r_line = rfile.readline()
            idx_r += 1
            s_line = sfile.readline()
            idx_s += 1
            if not s_line or not r_line:
                break
            if r_line.startswith("#####"):
                if not s_line.startswith("#####"):
                    raise ValueError(f"Error reading files, unmatched document titles at {regular_file}:{idx_r} and {simple_file}:{idx_s}.")
                # both lines are either start of doc or end of doc
                if "END" in r_line:
                    if not "END" in s_line:
                        raise ValueError(f"Error reading files, unmatched document ends at {regular_file}:{idx_r} and {simple_file}:{idx_s}.")
                    # END in s_line
                    doc_title, sub_doc_id, sent_num = reset_doc()
                else:
                    doc_title = get_doc_title(r_line)
            elif r_line.startswith(">>>"):
                if not s_line.startswith(">>>"):
                    raise ValueError(f"Error reading files, unmatched sub-document IDs at {regular_file}:{idx_r} and {simple_file}:{idx_s}.")
                # we are at sub doc id line
                sub_doc_id = get_sub_doc_id(r_line)
                sent_num = 0
            else:  # we are at lines that we need to add "as is" to the files.
                r_split = split_sentences(r_line)  # need to split the sentences to get alignment type
                s_split = split_sentences(s_line)
                result.append({"doc_title": doc_title, "sub_doc_id": sub_doc_id, "sentence_num": sent_num,
                               "entry_type": get_entry_type((r_split, s_split)),
                               "reg_sent": r_line.strip(), "sim_sent": s_line.strip()})
    return result


def get_TER_scores(reg_sents, sim_sents, nlp):
    result = []
    for reg_line, sim_line in tqdm(zip(reg_sents, sim_sents)):
        reg_doc = nlp(reg_line.strip())
        sim_doc = nlp(sim_line.strip())
        if len([token.text for token in sim_doc]) == 0:
            ter_score = 0
        elif len([token.text for token in reg_doc]) == 0:
            ter_score = len([token.text for token in sim_doc])
        else:
            try:
                ter_score = pyter.ter([token.text for token in reg_doc], [token.text for token in sim_doc])
            except TimeoutError:
                print(f"Timeout error in creating initial df.\n\tDocs: \n\t\t{reg_line}\n\t\t{sim_line}")
                ter_score = 0
        result.append(ter_score)
    return result


def create_wordlevel_datasets(path, ds_name, input=None, lang="en", split_size=None, start_idx=None):
    if input is None:  # we read from txt files, and already saved the correct csv.
        input_file = f"{path}/{ds_name}.csv"
    else:
        input_file = input

    print("Loading Spacy model")

    # TODO: Add Hebrew Support
    # TODO: Add General Language Support

    if lang == "en":
        NLP = spacy.load("en_core_web_lg")

        word_aligner = MonolingualWordAligner()

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
    else:
        raise ValueError(f"Unsupported Language {lang}")

    with CoreNLPClient(
            annotators=corenlp_annotators,
            properties=annotators_properties,
            threads=40, be_quiet=True
    ) as client:
        df = pd.read_csv(input_file, sep=";")
        df["reg_sent"] = df["reg_sent"].fillna("")
        df["sim_sent"] = df["sim_sent"].fillna("")

        time.sleep(client.CHECK_ALIVE_TIMEOUT + 30)
        if split_size is None:
            orig_tokenized, orig_parses = get_tokenized_and_parsing_from_list(df["reg_sent"].to_list(), NLP, nlpclient=client)
            simp_tokenized, simp_parses = get_tokenized_and_parsing_from_list(df["sim_sent"].to_list(), NLP, nlpclient=client)
            print("Adding TER to dataframe")
            df["TER_score"] = get_TER_scores(df["reg_sent"].to_list(), df["sim_sent"].to_list(), NLP)
            print("Adding word level")
            res = get_word_level(orig_tokenized, orig_parses, simp_tokenized, simp_parses, word_aligner, index=len(orig_tokenized))
            print("Concating and Saving dataframe")
            pd.concat([df, res], axis=1).to_csv(
                f"{path}/{ds_name}+actions+word_level.csv", encoding='utf-8', sep=';')
        else:
            assert isinstance(split_size, int)
            assert isinstance(start_idx, int)

            i = start_idx

            while i + split_size < len(df):
                sub_df = df.iloc[i:i+split_size, :]
                print(f"Creating Split {i}-{i+split_size}")
                orig_tokenized, orig_parses = get_tokenized_and_parsing_from_list(sub_df["reg_sent"].to_list(), NLP,
                                                                                  nlpclient=client)
                simp_tokenized, simp_parses = get_tokenized_and_parsing_from_list(sub_df["sim_sent"].to_list(), NLP,
                                                                                  nlpclient=client)
                print("\tAdding TER to dataframe")
                sub_df["TER_score"] = get_TER_scores(sub_df["reg_sent"].to_list(),
                                                     sub_df["sim_sent"].to_list(), NLP)
                print("\tAdding word level")
                res = get_word_level(orig_tokenized, orig_parses, simp_tokenized, simp_parses, word_aligner,
                                     index=len(orig_tokenized))
                print("\tConcating and Saving dataframe")
                pd.concat([sub_df, res.set_index(sub_df.index)], axis=1).to_csv(
                    f"{path}/{ds_name}+split-{i}-{i+split_size}+actions+word_level.csv", encoding='utf-8', sep=';')
                print(f"Done with {i}-{i+split_size}\n")
                i += split_size
            if i < len(df):
                sub_df = df.iloc[i:, :]
                print(f"Creating Split {i}-{len(df)}")
                orig_tokenized, orig_parses = get_tokenized_and_parsing_from_list(sub_df["reg_sent"].to_list(), NLP,
                                                                                  nlpclient=client)
                simp_tokenized, simp_parses = get_tokenized_and_parsing_from_list(sub_df["sim_sent"].to_list(), NLP,
                                                                                  nlpclient=client)
                print("\tAdding TER to dataframe")
                sub_df["TER_score"] = get_TER_scores(sub_df["reg_sent"].to_list(),
                                                     sub_df["sim_sent"].to_list(), NLP)
                print("\tAdding word level")
                res = get_word_level(orig_tokenized, orig_parses, simp_tokenized, simp_parses, word_aligner,
                                     index=len(orig_tokenized))
                print("\tConcating and Saving dataframe")
                pd.concat([sub_df, res.set_index(sub_df.index)], axis=1).to_csv(
                    f"{path}/{ds_name}+split-{i}-{len(df)}+actions+word_level.csv", encoding='utf-8', sep=';')
                print(f"Done with {i}-{len(df)}\n")
                i += split_size


if __name__ == "__main__":  # TO_DO: Cleanup comment outs
    parser = argparse.ArgumentParser("Convert aligned simplification datasets into csvs for further analysis.")
    subparsers = parser.add_subparsers(dest='subcommand')
    parser_txt = subparsers.add_parser("txt_to_base", help="Convert aligned text files to base CSV.")
    parser_csv = subparsers.add_parser("csv_to_base", help="Convert aligned csv file to base CSV format. \n"
                                                           "Expects the following collumns:\n"
                                                           "\tindex;doc_title;sub_doc_id;sentence_num;entry_type;reg_sent;sim_sent")
    parser_txt.add_argument("--reg_file", required=True, type=str)
    parser_txt.add_argument("--sim_file", required=True, type=str)
    parser_txt.add_argument("--data_path", required=True, type=str, help="Path to save data to.")
    parser_txt.add_argument("--dataset_name", required=True,
                            help="Name of the dataset, the final output will be <dataset_name>+actions+word_level.csv")

    parser_csv.add_argument("--input_file", type=str,
                            help="CSV file to read data from. If not provided, assumed to be <data_path>/<dataset_name>.csv")
    parser_csv.add_argument("--data_path", required=True, type=str,
                            help="Path to save data to.")
    parser_csv.add_argument("--dataset_name", required=True, type=str,
                            help="Name of the dataset, the final output will be <dataset_name>+actions+word_level.csv")
    parser_csv.add_argument("--split_size", type=int,
                            help="For larger datasets, split into sub-sets to save processing work")
    parser_csv.add_argument("--start_idx", type=int,
                            help="For larger datasets, index from which to start the processing (for work that was stopped in the middle)")

    args = parser.parse_args()

    if args.subcommand == "txt_to_base":
        input_file = f"{args.data_path}/{args.dataset_name}.csv"
        r = read_files(args.reg_file, args.sim_file)
        df = pd.DataFrame(r)
        df.to_csv(input_file, sep=";")
        args.input_file = None

    create_wordlevel_datasets(args.data_path, args.dataset_name, args.input_file,
                              split_size=args.split_size, start_idx=args.start_idx)
