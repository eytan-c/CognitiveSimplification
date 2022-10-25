import json
import re
import pathlib
import tqdm

sppdb_path = pathlib.Path("/Users/eytan.chamovitz/Downloads/SimplePPDB/SimplePPDB")
outpath = "/Users/eytan.chamovitz/PycharmProjects/CogSimp/simple_ppdbs/en_sppdb.json"


if __name__ == '__main__':
    result = {}
    with open(sppdb_path, "r") as f:
        for line in tqdm.tqdm(f, total=4505888):
            """
            The Simplificaiton dictionary contains 5 tab-separated fields, as follows:
            * Paraphrase Score -- The overall quality of the output prhase as a paraphrase of the input phrase, according PPDB 2.0. 
                Score is from 1 to 5, where higher is better. 
            * Simplification Score -- Our model's confidence that the output phrase is a simplificaiton of the input phrase. 
                Score is from 0.5 to 1.0, where higher is better.
            * Syntactic Category -- The CCG-style syntactic category assigned to the paraphrase rule.
            * Input Phrase -- Word or phrase to be simplified.
            * Output Phrase -- Simplification of input.
            """
            splits = line.strip().split('\t')
            if len(splits) == 5:
                para_score, sim_score, syn_cat, phrase, paraphrase = splits
            else:
                continue
            if phrase not in result:
                result[phrase] = []
            paraphrase_tokenized = paraphrase.split(' ')
            if len(paraphrase_tokenized) == 1:
                result[phrase].append([-1, paraphrase])
            else:
                result[phrase].append([-1, paraphrase_tokenized])
    with open(outpath, "w") as o:
        json.dump(result, o)
