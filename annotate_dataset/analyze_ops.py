import json
from collections import Counter

import spacy_lookups_data
# import pyter
import spacy
import logging
import re
import spacy.tokens.doc
from easse.annotation.word_level import *
from util.logger import setup_logger
from typing import Tuple, List, Union, Pattern, Match
from util.AlignTypes import AlignmentTypes
# from HebPipe_tests import *
from util.utils import get_reorder_sent_id
import Levenshtein


LOGGER = setup_logger('classify_operations', logging.INFO)
DELETE_PCT = 0.2
ADD_PCT = 0.2

"""
    1. Proximation-  All  of  these  operations  are tested a word by word basis using the Universal Dependency parse trees of the source and the target.
        (a) Change of person point of view-  We check if there was a change in person POV from 3rd to 2nd, 3rd to 1st, or 2nd to 1st.
        (b) Modify verb tense - We check if the verbs in the target are in a different tense that the verbs in the source.
        (c) Passive-Active Substitution- We check if there exist any passive verbs in the source that share meaning with active verbs in the target.
    
    2. Rephrasing- A rephrasing operation will fol-low the format of replacing one or more words from the source with one or more words with similar meaning in the target.  
                   Thus to identify  a  rephrasing,  we  tested  every  word  in the  source  sentence  that  did  not  appear 
                   in the target against known paraphrase databases for  the  relevant  language  (such  as  SPPDB (Pavlick and Callison-Burch, 2016) for English)  
                   to  see  if  one  of  their  relevant  para-phrases appears in the target.
                   Phrasing this mathematically, for every word w∈S\\T, we check if pp(w)⊂T, where pp(w) is the result of applying a rule from a paraphrase database on w.
        (a) Simple synonym- These operations are defined when one word is paraphrased to another single word.
        (b) Paraphrasing
            i. Word-to-Phrase- Similar to simple synonym, only a single word is paraphrased into a series of words.
            ii. Phrase-to-Word-  A  phrase  is  converted  to  a  single  word. This  is discovered by checking all 
                                 possible combinations of consecutive words in the source that did not appear in the target for possible paraphrases.
            iii. Phrase-to-Phrase- Similar to Phrase-to-Word, when the paraphrase rule is to another phrase instead of a single word.
    
    3. Deleting  Information-  Any  words  in  the source that don’t appear in the target designate a Deleting Information operation.  
                               We discern between Removal and Summarization mainly according to the alignment type. 
                               Precisely discerning between the two operations for other alignments types is a more complicated task that cannot be resolved by 
                               a simple heuristic, and as such we leave it for future research.  
                               For our analysis’ purpose,  whenever the percentage of deleted words from the source 
                               (i.e. that were removed in the target and were not part of another operation such as Rephrasing) was higher that X% (X is TBD) 
                               we classified it as a Deleting Information operation.
        (a) Removal- If the alignment type is Deletion (M-to-0), we count the operation as Removal.
        (b) Summarization- If the alignment type is Summarization (M-to-1), we count the operation as Summarization.
    4. Adding Information- To discover if an action was of Adding Information, we check if there are new words in the target, that aren’t part of 
                           another modification (such as Rephrasing or Passive-Active Substitution) or are function words.  
                           Once such words exists, we assume that there is additional explicit information in the target that did not appear in the source.  
                           We then test if it is Example Generation or Explanation Generation (see below), and if it is neither, similar to the 
                           general classification in Deleting information, if the percentage of new words is higher than Y% (Y is TBD), we classify 
                           as Adding Information.
        (a) Example Generation- If the new words are part of a clause that starts with indicative phrases for providing 
                                examples (such as ”e.g.”, ”for example”, ”such as”, and more) we classify this operation as Example Generation.
    5. Explicitation- From a modeling perspective, we grouped Pronoun Explicitation and Explanation Generation together, 
                      since their purpose is similar - reducing ambiguities in the text that are related to the implicit information 
                      and assumptions. 
                      However, from a classification perspective each is discovered differently.
        (a) Pronoun Explicitation- EC: How hard would it be to add Coreference Resolution code here?
        (b) Explanation  Generation-  We  identify this operation together with Adding Information, since heuristically they can appear very similar. 
                                      If new words in the target aren’t tied to an example, or are tied to a noun phrase in the source that is part of 
                                      one or more sentences in the target, we assume that this is a form of Explanation. 
                                      Discerning between the different types of explanation generations is a task for future research, 
                                      but we list them here for indexing purposes.
            i.  For term/phrase
            ii.  For logic/reasoning
            iii.  For background information
    6. Intra-Sentence Rearrangement- This operation is identified when the information order in a text is changed.  
                                     We use the UD parse trees of the source and target to discover rearrangements.
        (a) Clause Reordering- If the clauses in the target appear in a different order that in the source, then this is a Clause Reordering operation.
        (b) SVOReordering - For each sentence in the source, we check if the order of subject, verb, and object are maintained int he target.
                            If not, then this is an SVO Reordering.
    7. Operations on Sentences- These operations are checked on a Sub-document level as compared to a Simplification Instance Level.
        (a) Sentence Splitting- This operations is assumed to appear by default in Expansion(1-to-N) Simplification Instances.
        (b) Sentence  Rearrangement-  Part of the manual alignment  process, the  original ordering of sentences in the source sub-document 
                                      and be compared to the order of the original sentences according to their alignment to the
                                      target sub-document.
                                      So, if the source  sub-document  consists  of  sentence[s1, s2, s3, ..., sn]
                                      and their alignment to the target sub-document sentences is some  permutation  of their indexes I,  
                                      such  that  the  source  sentences ordered by the target’s order is [si1, si2, ..., sin], 
                                      we look for the longest increase sub-sequence in this permutation L⊂I.  
                                      Any sentence indexed by i_j /∈ L is a Sentence Rearrangement.
    8. Document Level Operations- We list here the  Document  Level  Operations,   but  for our  analysis  we  only  focused  on  
                                  identifying  Adding/Deleting  Paragraphs  and  Sub-documents,  which were respectively classified as Adding/Deleting Information.  
                                  In addition, as part of our Reordering analysis, we were able to discover 
                                  Cross-Paragraph Sentence Reordering if they occurred in the same Sub-Document.
        (a)  Paragraph Splitting
        (b)  Cross-Paragraph Sentence Reordering
        (c)  Paragraph Rearangement
        (d)  Sub-Document Rearrangement
        (e)  Adding Paragraphs
        (f)  Adding Sub-Documents
        (g)  Deleting Paragraphs
        (h)  Deleting Sub-Documents
"""

"""
    TODO: Think of how to save the syntactic_parse_texts() of a dataset ahead of time
    Already it this, when saving files with the '+word_level' filename!
"""
"""
In order to get word-level deletions/additions/keeps, we need the easse module.
Specifically, we need to take the code from word_align_tests.py as a basis for getting the word level ops.
Also, need to think of a way to save the analysis of a dataset, since running the corenlp server everytime will be very expensive time-wise.
"""


class ContinueOuter(Exception):
    pass


def normalize_string(s, only_heb=False):
    """
    Original project's token normalization
    :param s:
    :param only_heb:
    :return:
    """
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[\u0590-\u05CF]+", "", s)
    if only_heb:
        s = re.sub(r"[^א-ת.!?\"]+", r" ", s)
        s = re.sub(r"(^|\s)\"(\w)", r"\1\2", re.sub(r"(\w)\"(\s|$)", r"\1\2", s))
    # s = re.sub(r"(^|\s)\'(\w)", r"\1\2", re.sub(r"(\w)\'(\s|$)", r"\1\2", s))
    else:
        s = re.sub(r"[^a-zA-Zא-ת.!?\"]+", r" ", s)
        s = re.sub(r"(^|\s)\"(\w)", r"\1\2", re.sub(r"(\w)\"(\s|$)", r"\1\2", s))
    return s


"""
    Add calculations for metrics from ACCESS Martin2020
    1. <NbChars> - ratio between number of characters in source vs. target. They state that this is correlated with 
                   simplicity, and that it accounts for both deletions and summarizations
    2. <LevSim> - normalized character-level Levenshtein similarity between source and target. 
                  Proxy for amount of modification between source and target.
    3. <WordRank> - proxy for lexical complexity, measured sentence level, with ratio between source and target. 
        1. Calculation by taking third-quartile of log-ranks (inverse frequency order) of all words in a sentence. 
        2. Semeval 2016 task 11 showed that word frequency is best indicator for word complexity
    4. <DepTreeDepth> - max depth of the dependency tree of the source divided by that of the target.
        1. This was chosen since in early experiments it was found better than max dependency length and max inter-word 
           dependency flux as indicator for syntactic complexity.
"""


def get_nbchars_ratio(source: spacy.tokens.Doc, target: spacy.tokens.Doc, align_type: AlignmentTypes) -> Tuple[float,int,int]:
    """
    Calaculate the ratio between number of characters in source vs. target.
    ACCESS (Martin2020) state that this is correlated with simplicity, and that it accounts for both deletions
    and summarizations.
    :param source: the source sentence (in regular language)
    :param target: the target sentence (simplification)
    :param align_type: the type of alignment between the source and target
    :return: ratio, source number of chars, target number of chars
    """
    if align_type is AlignmentTypes.DELETION:
        return np.nan, sum([len(t) for t in source]), 0
    elif align_type is AlignmentTypes.ADDITION:
        return 0.0, 0, sum([len(t) for t in target])
    else:
        source_nbchars = sum([len(t) for t in source])
        target_nbchars = sum([len(t) for t in target])
        return source_nbchars/target_nbchars, source_nbchars, target_nbchars


def get_levsim(source: spacy.tokens.Doc, target: spacy.tokens.Doc, align_type: AlignmentTypes) -> float:
    """
    Calaculate the normalized character-level Levenshtein similarity between source and target.
    ACCESS (Martin2020) state that this is a proxy for amount of modification between source and target.
    :param source: the source sentence (in regular language)
    :param target: the target sentence (simplification)
    :param align_type: the type of alignment between the source and target
    :return:
    """
    if align_type is AlignmentTypes.DELETION or align_type is AlignmentTypes.ADDITION:
        return np.nan
    return Levenshtein.distance(source.text.lower(), target.text.lower())


def get_wordrank_ratio(source: spacy.tokens.Doc, target: spacy.tokens.Doc, lex_prob: dict, align_type: AlignmentTypes) -> Tuple[float, float, float]:
    """
    Calculate the WordRank ratio between the source and target.
    According to ACCESS (Martin2020) WordRank is a proxy for lexical complexity, measured sentence level,
    with ratio between source and target.
        1. Calculation by taking third-quartile of log-ranks (inverse frequency order) of all words in a sentence.
        2. Semeval 2016 task 11 showed that word frequency is best indicator for word complexity
    :param source: the source sentence (in regular language)
    :param target: the target sentence (simplification)
    :param lex_prob: the lexical probabilities of words in the langauge.
    :param align_type: the type of alignment between the source and target
    :return: ratio (targe/source), source WordRank, target WordRank
    """
    if align_type is AlignmentTypes.DELETION:
        return 0.0, np.percentile(np.array([lex_prob.get(t.text, -20.0) for t in source]), 25), 0
    elif align_type is AlignmentTypes.ADDITION:
        return np.nan, 0, np.percentile(np.array([lex_prob.get(t.text, -20.0) for t in target]), 25)
    else:
        source_word_probs = np.array([lex_prob.get(t.text, -20.0) for t in source])
        target_word_probs = np.array([lex_prob.get(t.text, -20.0) for t in target])
        # by taking the third-quartile of log-ranks (inverse frequency order) of all words in a sentence
        source_wordrank = np.percentile(source_word_probs, 25)
        target_wordrank = np.percentile(target_word_probs, 25)
        return target_wordrank / source_wordrank, source_wordrank, target_wordrank


def walk_tree(node: spacy.tokens.Token, depth: int) -> int:
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth


def get_deptree_depth_ratio(source: spacy.tokens.Doc, target: spacy.tokens.Doc, align_type: AlignmentTypes) \
        -> Tuple[float, int, List[int], List[int]]:
    """
    Calculate the max depth of the dependency tree of the source divided by that of the target.
     According to ACCESS (Martin2020) this was chosen since in early experiments it was found better than
     max dependency length and max inter-word dependency flux as indicator for syntactic complexity.
    :param source: the source sentence (in regular language)
    :param target: the target sentence (simplification)
    :param align_type: the type of alignment between the source and target
    :return: ratios (source/target), type of dependency parse tree ratios , source depths, target_depths
    """
    if align_type is AlignmentTypes.DELETION:
        return 0.0, 0, [walk_tree(sent.root, 0) for sent in source.sents], []
    elif align_type is AlignmentTypes.ADDITION:
        return np.nan, 0, [], [walk_tree(sent.root, 0) for sent in target.sents]
    source_depths = [walk_tree(sent.root, 0) for sent in source.sents]
    target_depths = [walk_tree(sent.root, 0) for sent in target.sents]
    if len(source_depths) == len(target_depths):
        try:
            return sum(source_depths) / sum(target_depths), 1, source_depths, target_depths
        except ZeroDivisionError:
            return sum(source_depths) / 1, 1, source_depths, target_depths
    elif len(source_depths) > len(target_depths):
        return np.mean(source_depths) / np.mean(target_depths), 2, source_depths, target_depths
    else:  # len(source_depths) < len(target_depths)
        return np.mean(source_depths) / np.mean(target_depths), 3, source_depths, target_depths




"""
1. Proximation-  All  of  these  operations  are tested a word by word basis using the Universal Dependency parse trees of the source and the target.
        (a) Change of person point of view-  We check if there was a change in person POV from 3rd to 2nd, 3rd to 1st, or 2nd to 1st.
        (b) Modify verb tense - We check if the verbs in the target are in a different tense that the verbs in the source.
        (c) Passive-Active Substitution- We check if there exist any passive verbs in the source that share meaning with active verbs in the target.
"""


def get_person_words(document: spacy.tokens.doc.Doc) -> Tuple[List[Tuple[spacy.tokens.Token, int]], List[int]]:
    res = []
    used_idx = []
    for i, t in enumerate(document):
        t_morph = t.morph.to_dict()
        if t.pos_ != 'VERB' and t.pos_ != 'AUX' and "Person" in t_morph:
            res.append((t, int(t_morph["Person"])))
            used_idx.append(i)
    return res, used_idx


def has_pov_change(source: spacy.tokens.doc.Doc, target: spacy.tokens.doc.Doc, lang) -> int:
    source_person_words, source_used_idx = get_person_words(source)
    target_person_words, target_used_idx = get_person_words(target)
    source_person_avg = np.mean([i for _, i in source_person_words]) if len(source_person_words) > 0 else 0
    target_person_avg = np.mean([i for _, i in target_person_words]) if len(target_person_words) > 0 else 0
    if target_person_avg < source_person_avg:
        return 1
    elif target_person_avg > source_person_avg:
        return -1
    else:
        return 0


def has_verb_tense_change(source: spacy.tokens.doc.Doc, target: spacy.tokens.doc.Doc, lang) \
        -> Tuple[int, List[int], List[int]]:
    shared_verbs = get_shared_verbs([(i, t) for i, t in enumerate(source) if t.pos_ == 'VERB'],
                                    [(i, t) for i, t in enumerate(target) if t.pos_ == 'VERB'])
    tense_change = 0
    source_used = []
    target_used = []
    for (i, s), (j, t) in shared_verbs:
        s_morph = s.morph.to_dict()
        t_morph = t.morph.to_dict()
        if s_morph.get("Tense") != t_morph.get("Tense") or \
                s_morph.get("Aspect") != t_morph.get("Aspect") or \
                s_morph.get("VerbForm") != t_morph.get("VerbForm"):
            tense_change += 1
            source_used.append(i)
            target_used.append(j)

    return tense_change, source_used, target_used


def get_shared_verbs(source_verbs: List[Tuple[int, spacy.tokens.Token]],
                     target_verbs: List[Tuple[int, spacy.tokens.Token]]) \
        -> List[Tuple[Tuple[int, spacy.tokens.Token], Tuple[int, spacy.tokens.Token]]]:
    res = []
    used = [0 for t in target_verbs]
    for j, s in source_verbs:
        for i, (k, t) in enumerate(target_verbs):
            if used[i] == 0 and s.lemma == t.lemma:
                res.append(((j, s), (k, t)))
                used[i] = 1
    return res


def has_passive_active_sub(source: spacy.tokens.doc.Doc, target: spacy.tokens.doc.Doc, lang) -> Tuple[int, List[int]]:
    passive_in_source = 0
    passive_in_target = 0
    source_used_idx = []
    target_used_idx = []
    for i, tok in enumerate(source):
        if tok.dep_.endswith("pass"):
            passive_in_source += 1
            source_used_idx.append(i)

    # if we reach here and source_has_passive == 0 then we can return 0
    # if passive_in_source == 0:
    #     return 0

    for i, tok in enumerate(target):
        if tok.dep_.endswith("pass"):
            passive_in_target += 1

    if passive_in_source > passive_in_target:
        return 1, source_used_idx
    elif passive_in_source < passive_in_target:
        return -1, []
    else:
        return 0, []


#  TO_DO: Understand how to add morphological parser to spacy pipeline
def classify_proximation(source: spacy.tokens.doc.Doc, target: spacy.tokens.doc.Doc, lang) \
        -> Tuple[int, int, int, List[int], List[int]]:
    """
    Get a proximation classification. This is either a pov change, a verb tense change or a passive-active voice change.
    :param source: Source sentence (in regular language) -> sentence to be simplified
    :param target: Target sentence (in Simple Language) -> result of simplification
    :param lang: Language specifier
    :return: ( pov_change [1 is decrease voice, -1 is increase voice, 0 no change],
               verb_tense_change [# of changes],
               passve_active [1 passive->active, -1 active->passive, 0 no change],
               source_used [list of token idxs that are used in proximation from the source],
               target_used [list of token idxs that are used in proximation from the target])
    """
    pov_change = has_pov_change(source, target, lang)
    verb_tense_change, verb_source_used, verb_target_used = has_verb_tense_change(source, target, lang)
    passive_active_sub, pa_source_used = has_passive_active_sub(source, target, lang)

    source_used = sorted(set.union(set(verb_source_used), pa_source_used))

    return pov_change, verb_tense_change, passive_active_sub, source_used, sorted(verb_target_used)


"""
2. Rephrasing- A rephrasing operation will fol-low the format of replacing one or more words from the source with one or more words with similar meaning in the target.  
                   Thus to identify  a  rephrasing,  we  tested  every  word  in the  source  sentence  that  did  not  appear 
                   in the target against known paraphrase databases for  the  relevant  language  (such  as  SPPDB (Pavlick and Callison-Burch, 2016) for English)  
                   to  see  if  one  of  their  relevant  para-phrases appears in the target.
                   Phrasing this mathematically, for every word w∈S\\T, we check if pp(w)⊂T, where pp(w) is the result of applying a rule from a paraphrase database on w.
        (a) Simple synonym- These operations are defined when one word is paraphrased to another single word.
        (b) Paraphrasing
            i. Word-to-Phrase- Similar to simple synonym, only a single word is paraphrased into a series of words.
            ii. Phrase-to-Word-  A  phrase  is  converted  to  a  single  word. This  is discovered by checking all 
                                 possible combinations of consecutive words in the source that did not appear in the target for possible paraphrases.
            iii. Phrase-to-Phrase- Similar to Phrase-to-Word, when the paraphrase rule is to another phrase instead of a single word.
"""


def get_deleted_word_groups(source: spacy.tokens.Doc, source_labels: List[str]) -> List[List[Tuple[int, str]]]:
    """
    Returns the spans of tokens from the source that are deleted.
    Each groups is save as a list of tuples of (<token.i>, <token.lemma_>).
    Return list of groups.
    :param source:
    :param source_labels:
    :return: List[List[Tuple[<token.i>, <token.lemma_>]]
    """
    res = []
    group = []
    for i, label in enumerate(source_labels):
        if label == 'D' or label == 'R':
            # group.append((i, source[i].text))
            try:
                group.append((i, source[i].lemma_))
            except IndexError:
                continue
        else:  # label == 'C' or label == 'M'
            if len(group) > 0:
                if len(group) > 1:  # since single words already test
                    res.append(group)
                group = []
    if len(group) > 1:  # since single words already test
        res.append(group)
    return res


def num_paraphrases_in_group(g, target, ppdb) -> Tuple[int, int, List[int]]:
    """
    Looks for paraphrases of increasing length from group g in the target document.
    :param g: group to look for paraphrases in
    :param target:
    :param ppdb:
    :return: (phrase2word [# of phrase2word translations],
              phrase2phrase [# of phrase2phrase translations],
              sorted(source_used) [list of token idxs that are used in paraphrasing from the source])
    """
    phrase2word = 0
    phrase2phrase = 0
    source_used = set()
    target_used = set()
    words = [t[1] for t in g]
    idxs = [t[0] for t in g]
    for l in range(2, len(g) + 1):
        for i in range(len(g) - (l - 1)):
            end = i + l
            word = " ".join(words[i:end])
            try:
                pps = find_paraphrases(word, target, ppdb)
                if pps == 1:
                    phrase2word += 1
                    source_used = source_used.union(idxs[i:end])
                elif pps == 2:
                    phrase2phrase += 1
                    source_used = source_used.union(idxs[i:end])
            except KeyError:
                continue
    return phrase2word, phrase2phrase, sorted(source_used)


def find_paraphrases(word: str, target: spacy.tokens.Doc, ppdb: dict) -> int:
    """
    Look in ppdb for rules regarding word that have a match in the target.
    :param word:
    :param target:
    :param ppdb:
    :return: 1 if paraphrase length == 1 word, 2 if paraphrase length > 1 word, 0 if none found
    """
    pp_rules = ppdb[word]
    for _, pp in pp_rules:  # instances in ppdb are of format [-1, Union[str, List[str]]]
        if isinstance(pp, list):  # to deal with List[str] option for second arg
            pp = " ".join(pp)
        if pp in target.text:
            if len(pp.split(" ")) == 1:  # to count number of words in the phrase
                return 1
            else:
                return 2
    return 0


def classify_rephrasing(source: spacy.tokens.Doc, target: spacy.tokens.Doc, lang: str, ppdb: dict,
                        source_labels: List[str]) -> Tuple[int, int, int, int, List[int], List[int]]:
    """
    For every word w in the source, check if there exists a word u or phrase p = [u_1,...,u_k] in the target such that:
        ppdb(w) = u or ppdb(w) = p
    If | ppdb(w) | = 1 then Simple synonym
    If | ppwd(w) | > 1, then Word-to-Phrase
    For every group of consecutive words in the source sentence that doesn't appear in the target, do the following:
        For every n-gram in the group, n going from 2 to l=group length, check if there is a rule in the ppdb.
            (meaning, is ppdb(n-gram) valid).
        If there is a rule, check if the result of that rule exists in the target, meaning ppdb(n-gram) ⊂ T.
        If |ppdb(n-gram)| > 1, then we are in Phrase-to-Phrase.
        If |ppdb(n-gram)| = 1, then we are in Phrase-to-Word.

    :param source:
    :param target:
    :param lang:
    :param ppdb:
    :param source_labels:
    :return: (simple_synonym [# of simple_synonym translations],
              word2phrase [# of word2phrase translations],
              phrase2word [# of phrase2word translations],
              phrase2phrase [# of phrase2phrase translations],
              sorted(set(source_used)) [list of token idxs that are used in paraphrasing from the source],
              sorted(set(target_used)) [list of token idxs that are used in paraphrasing from the target]
              )
    """
    simple_synonym = 0
    word2phrase = 0
    phrase2word = 0
    phrase2phrase = 0
    source_used = []
    target_used = []
    LOGGER.debug("Starting paraphrasing")
    LOGGER.debug("Finding Single word paraphrases")
    for i, w in enumerate(source):
        try:
            # pps = find_paraphrases(w.lower_, target, ppdb)
            pps = find_paraphrases(w.lemma_, target, ppdb)
            if pps == 1:
                simple_synonym += 1
                source_used.append(i)
                raise ContinueOuter  # use the exception to continue the outer loop
            elif pps == 2:
                word2phrase += 1
                source_used.append(i)
                raise ContinueOuter  # use the exception to continue the outer loop
        except ContinueOuter:
            continue
        except KeyError:
            LOGGER.debug(f"Word {w} has no paraphrases")
            continue

    LOGGER.debug("Finding phrases paraphrases")
    groups = get_deleted_word_groups(source, source_labels)
    for g in groups:
        num_phrase2word, num_phrase2phrase, group_source_used = num_paraphrases_in_group(g, target, ppdb)
        source_used += group_source_used
        phrase2word += num_phrase2word
        phrase2phrase += num_phrase2phrase
    LOGGER.debug(f"Found paraphrases: {simple_synonym}, {word2phrase}, {phrase2word}, {phrase2phrase}")
    return simple_synonym, word2phrase, phrase2word, phrase2phrase, sorted(set(source_used)), sorted(set(target_used))


"""
3. Deleting  Information-  Any  words  in  the source that don’t appear in the target designate a Deleting Information operation.  
                           We discern between Removal and Summarization mainly according to the alignment type. 
                           Precisely discerning between the two operations for other alignments types is a more complicated task that cannot be resolved by 
                           a simple heuristic, and as such we leave it for future research.  
                           For our analysis’ purpose,  whenever the percentage of deleted words from the source 
                           (i.e. that were removed in the target and were not part of another operation such as Rephrasing) was higher that X% (X is TBD) 
                           we classified it as a Deleting Information operation.
        (a) Removal- If the alignment type is Deletion (M-to-0), we count the operation as Removal.
        (b) Summarization- If the alignment type is Summarization (M-to-1), we count the operation as Summarization.
"""


def get_deleted_unused(source, target, source_labels, used_mask) -> Tuple[int, int]:
    deleted_groups = get_deleted_word_groups(source, source_labels)
    deleted = 0
    deleted_unused = 0
    for group in deleted_groups:
        deleted += len(group)
        deleted_unused += len([w for w in group if w[0] not in used_mask])
    return deleted, deleted_unused


def classify_deleting_info(source: spacy.tokens.Doc, target: spacy.tokens.Doc, lang: str,
                           align_type: AlignmentTypes, word_labels: List[str], used_mask: List[int],
                           percent: float = DELETE_PCT) -> Tuple[int, int, int, Union[int, float], Union[int, float]]:
    """

    :param source:
    :param target:
    :param lang:
    :param align_type:
    :param word_labels:
    :param used_mask:
    :param percent:
    :return: deleting_info - classification decision,
             removal - classification decision,
             summarization - classification decision,
             percent_deleted_unused - ration, unused_deleted_word / len(source),
             length_ratio - len(source) / len(target)
    """
    deleting_info = 0
    percent_deleted_unused = 0
    length_ratio = 0
    removal = 0
    summarization = 0
    LOGGER.debug("Starting deleting information classification")
    if align_type is AlignmentTypes.DELETION:
        removal = 1
    elif align_type is AlignmentTypes.ADDITION:
        return deleting_info, removal, summarization, percent_deleted_unused, length_ratio
    else:
        if align_type is AlignmentTypes.SUMMARIZATION:
            summarization = 1
        LOGGER.debug("Looking for unused words in the source that were deleted.")
        deleted, deleted_unused = get_deleted_unused(source, target, word_labels, used_mask)
        percent_deleted_unused = deleted_unused / len(source)
        length_ratio = len(source) / len(target)
        if percent_deleted_unused > percent:
            LOGGER.debug("Found unused words in high enough percentage")
            deleting_info = 1

    return deleting_info, removal, summarization, percent_deleted_unused, length_ratio


"""
4. Adding Information- To discover if an action was of Adding Information, we check if there are new words in the target, that aren’t part of 
                       another modification (such as Rephrasing or Passive-Active Substitution) or are function words.  
                       Once such words exists, we assume that there is additional explicit information in the target that did not appear in the source.  
                       We then test if it is Example Generation or Explanation Generation (see below), and if it is neither, similar to the 
                       general classification in Deleting information, if the percentage of new words is higher than Y% (Y is TBD), we classify 
                       as Adding Information.
    (a) Example Generation- If the new words are part of a clause that starts with indicative phrases for providing 
                            examples (such as ”e.g.”, ”for example”, ”such as”, and more) we classify this operation as Example Generation.
5. Explicitation- From a modeling perspective, we grouped Pronoun Explicitation and Explanation Generation together, 
                  since their purpose is similar - reducing ambiguities in the text that are related to the implicit information 
                  and assumptions. 
                  However, from a classification perspective each is discovered differently.
    (a) Pronoun Explicitation- EC: How hard would it be to add Co-reference Resolution code here?
    (b) Explanation  Generation-  We  identify this operation together with Adding Information, since heuristically they can appear very similar. 
                                  If new words in the target arn’t tied to an example, or are tied to a noun phrase in the source that is part of 
                                  one or more sentences in the target, we assume that this is a form of Explanation. 
                                  Discerning between the different types of explanation generations is a task for future research, 
                                  but we list them here for indexing purposes.
        i.  For term/phrase
        ii.  For logic/reasoning
        iii.  For background information
"""


def get_example_tokens(doc: spacy.tokens.Doc, example_match: Match) -> Tuple[List[spacy.tokens.Token], spacy.tokens.Token]:
    potential_tokens = [t for t in doc if example_match.start() <= t.idx <= example_match.end()]
    potential_roots = [t for t in potential_tokens if t.head not in potential_tokens]
    if len(potential_roots) == 0:
        potential_roots = [t for t in potential_tokens if t.head == t]
    example_root = potential_roots[0]
    return potential_tokens, example_root


def get_examples(source, target, lang, target_used_mask, example_regex) -> Tuple[int, List[int]]:
    potential_examples_target = example_regex.finditer(target.text)
    potential_examples_source = example_regex.finditer(source.text)
    target_used = []
    examples = 0
    for pot_ex in potential_examples_target:
        # t = target.text[pot_ex.start():pot_ex.end()]
        example_tokens, example_root = get_example_tokens(target, pot_ex)
        example_span = [t for t in example_root.subtree]
        found_equal = 0
        for source_ex in potential_examples_source:
            source_tokens, source_root = get_example_tokens(source, source_ex)
            if example_root.lower_ == source_root.lower_:
                source_span = [t for t in source_root.subtree]
                intersect = set([t.lower_ for t in source_span]).intersection([t.lower_ for t in example_span])
                if len(intersect) > len(source_tokens):
                    found_equal = 1
                    break
        if found_equal == 0:
            target_used += [t.i for t in example_span]
            examples += 1
    return examples, sorted(set(target_used))


def get_explanation(source, target, used_mask, example_mask, example_regex, explain_regex) -> Tuple[int, List[int]]:
    explain = 0
    explain_used = []
    for potential_explain in explain_regex.finditer(target.text):
        if len(potential_explain.groups()) > 0:
            group_num = 1
        else:
            group_num = 0
        if explain_regex.search(potential_explain.group(group_num)) is None:
            explain += 1
            explain_tokens, _ = get_example_tokens(target, potential_explain)
            explain_used += [t.i for t in explain_tokens]
    return explain, sorted(set(explain_used))


def get_added_word_groups(target: spacy.tokens.Doc, target_labels: List[str]) -> List[List[Tuple[int, str]]]:
    result = []
    group = []
    for i, label in enumerate(target_labels):
        if label == 'A' or label == 'R':
            try:
                group.append((i, target[i].text))
            except IndexError:
                continue
        else:  # label == 'C' or label == 'M'
            if len(group) > 0:
                if len(group) > 1:  # since single words already test
                    result.append(group)
                group = []
    if len(group) > 1:  # since single words already test
        result.append(group)
    return result


def get_added_unused(source, target, target_labels, used_mask) -> Tuple[int, int]:
    added_groups = get_added_word_groups(target, target_labels)
    added = 0
    added_unused = 0
    for group in added_groups:
        added += len(group)
        added_unused += len([w for w in group if w[0] not in used_mask])
    return added, added_unused


def get_example_regex(lang: str) -> Pattern:
    if lang == 'en':
        return re.compile(r"[Ff]or example|[Ii]\.e\.|[Ee]\.g\.|[Ss]uch as")
    elif lang == 'he':
        return re.compile(r"לדוגמה|לדוגמא|למשל|כגון")
    else:
        return re.compile(r"")


def classify_adding_info(source: spacy.tokens.Doc, target: spacy.tokens.Doc, lang: str, align_type: AlignmentTypes,
                         target_annot: List[str], used_mask, percent: float = ADD_PCT) -> Tuple[int, int, int, float]:
    if align_type is AlignmentTypes.DELETION:
        return 0, 0, 0, 0.0
    is_adding = 1 if align_type is AlignmentTypes.ADDITION else 0
    example_regex = get_example_regex(lang)
    explain_regex = re.compile(r"\w+ \(([^)]+)\)")
    examples, example_mask = get_examples(source, target, lang, used_mask, example_regex)
    explanations, explain_mask = get_explanation(source, target, used_mask, example_mask,  example_regex, explain_regex)

    added, added_unused = get_added_unused(source, target, target_annot, sorted(set(used_mask).union(example_mask, explain_mask)))
    percent_added_unused = added_unused / len(target)
    if percent_added_unused > percent:
        is_adding = 1
    return is_adding, examples, explanations, percent_added_unused


def get_pronoun_explicitation(document: spacy.tokens.Doc, source_span, target_span) -> int:
    if len(document._.coref_chains) == 0:
        return np.nan
    for chain in document._.coref_chains:
        # print(chain)
        source_pron = 0
        target_pron = 0
        source_other = 0
        target_other = 0
        for mention in chain.mentions:
            for idx in mention.token_indexes:
                if target_span[0] <= idx <= target_span[1]:
                    if document[idx].pos_ == 'PRON':
                        target_pron += 1
                    else:
                        target_other += 1
                elif source_span[0] <= idx <= source_span[1]:
                    if document[idx].pos_ == 'PRON':
                        source_pron += 1
                    else:
                        source_other += 1
        source_sum = source_pron + source_other
        target_sum = target_pron + target_other
        if source_sum == target_sum:
            return source_pron - target_pron
        else:
            if target_other == source_other:
                return 0
            else:
                return target_other - source_other
    return 0


def classify_explicitation(source: spacy.tokens.Doc, target: spacy.tokens.Doc, lang: str, nlp: spacy.Language, align_type: AlignmentTypes) -> int:
    if align_type is AlignmentTypes.DELETION or align_type is AlignmentTypes.ADDITION:
        return 0
    doc_4_ref = nlp("\t".join((target.text, source.text)))
    target_span = (0, len(target))
    source_span = (len(target)+1, len(target)+1+len(source))
    pron_expli = get_pronoun_explicitation(doc_4_ref, source_span, target_span)
    return pron_expli


"""
6. Intra-Sentence Rearrangement- This operation is identified when the information order in a text is changed.  
                                 We use the UD parse trees of the source and target to discover rearrangements.
    (a) Clause Reordering- If the clauses in the target appear in a different order that in the source, then this is a Clause Reordering operation.
    (b) SVOReordering - For each sentence in the source, we check if the order of subject, verb, and object are maintained in the target.
                        If not, then this is an SVO Reordering.
"""


def get_moved_words_out_of_order(source, target, source_labels, target_labels):
    if len(source_labels) == 0 or len(target_labels) == 0:
        return []
    try:
        moved_source = [tok for tok in source if (source_labels[tok.i] == 'M' or source_labels[tok.i] == 'C') and not tok.is_space]
        kept_target = [tok for tok in target if target_labels[tok.i] == 'C' and not tok.is_space]
    except IndexError:
        return np.nan
    token_counter = Counter()
    moved_target = []
    for m in moved_source:
        token_counter.update([m])
        potential_matches = [t for t in kept_target if t.lower_ == m.lower_]
        if len(potential_matches) == 0:
            continue
        match = potential_matches[token_counter[m] - 1]
        moved_target.append(match)
    target_order = [t.i for t in moved_target]  # indexes in target in order according to source. Use LIS to find ooo
    out_of_order = [moved_target[i] for i in get_reorder_sent_id(target_order)]
    return out_of_order


def get_noun_phrases_out_of_order(source, target, source_labels, target_labels):
    # TO_DO: Try to match with BERTscores
    if len(source_labels) == 0 or len(target_labels) == 0:
        return []
    root_counter = Counter()
    potential_moves = []
    for source_chunk in source.noun_chunks:
        root_counter.update([source_chunk.root])
        match = [t for t in target.noun_chunks if t.root.lower_ == source_chunk.root.lower_][root_counter[source_chunk.root] - 1]
        potential_moves.append(match)
    target_order = [m.start for m in potential_moves]
    out_of_order_roots = [potential_moves[i] for i in get_reorder_sent_id(target_order)]
    return len(out_of_order_roots)


def get_svo_order(doc: spacy.tokens.Doc):
    order = []
    for token in doc:
        if "subj" in token.dep_ or "obj" in token.dep_ or token.dep_ == 'ROOT':
            order.append((token.text, token.i, token.dep_))
    order_text = '|'.join([o[2] for o in order])
    svo_regex = re.compile(r"(\w*subj\w*\|(?:[^|]+\|)*ROOT\|\w*obj\w*)")
    ovs_regex = re.compile(r"(\w*obj\w*\|(?:[^|]+\|)*ROOT\|\w*subj\w*)")
    svo_order = len(svo_regex.findall(order_text))
    ovs_order = len(ovs_regex.findall(order_text))
    return svo_order, ovs_order


def get_svo_out_of_order(source, target):
    source_svo_order = get_svo_order(source)
    target_svo_order = get_svo_order(target)
    total_source = sum(source_svo_order)
    total_target = sum(target_svo_order)
    total_diff = total_source - total_target
    svo_diff = source_svo_order[0] - target_svo_order[0]
    ovs_diff = source_svo_order[1] - target_svo_order[1]
    if total_diff == svo_diff or total_diff == ovs_diff:
        return 0
    if ovs_diff > 0:
        return 1
    return None


def classify_intra_sentence_rearrange(source, target, lang, source_labels, target_labels):
    # TO_DO: Think of how to find ooo noun phrases
    number_moved_words_ooo = get_moved_words_out_of_order(source, target, source_labels, target_labels)
    # noun_phrases_ooo = get_noun_phrases_out_of_order(source, target, source_labels, target_labels)
    num_svo_ooo = get_svo_out_of_order(source, target)
    return number_moved_words_ooo, num_svo_ooo  #, noun_phrases_ooo


"""
7. Operations on Sentences- These operations are checked on a Sub-document level as compared to a Simplification Instance Level.
    (a) Sentence Splitting- This operations is assumed to appear by default in Expansion(1-to-N) Simplification Instances.
    (b) Sentence  Rearrangement-  Part of the manual alignment  process, the  original ordering of sentences in the source sub-document 
                                  and be compared to the order of the original sentences according to their alignment to the
                                  target sub-document.
                                  So, if the source  sub-document  consists  of  sentence[s1, s2, s3, ..., sn]
                                  and their alignment to the target sub-document sentences is some  permutation  of their indexes I,  
                                  such  that  the  source  sentences ordered by the target’s order is [si1, si2, ..., sin], 
                                  we look for the longest increase sub-sequence in this permutation L⊂I.  
                                  Any sentence indexed by i_j /∈ L is a Sentence Rearrangement.
8. Document Level Operations- We list here the  Document  Level  Operations,   but  for our  analysis  we  only  focused  on  
                              identifying  Adding/Deleting  Paragraphs  and  Sub-documents,  which were respectively classified as Adding/Deleting Information.  
                              In addition, as part of our Reordering analysis, we were able to discover 
                              Cross-Paragraph Sentence Reordering if they occurred in the same Sub-Document.
    (a)  Paragraph Splitting
    (b)  Cross-Paragraph Sentence Reordering
    (c)  Paragraph Rearangement
    (d)  Sub-Document Rearrangement
    (e)  Adding Paragraphs
    (f)  Adding Sub-Documents
    (g)  Deleting Paragraphs
    (h)  Deleting Sub-Documents
"""


def classify_sentence_splitting(source, target, lang, align_type: AlignmentTypes) -> int:
    if align_type is AlignmentTypes.EXPANSION or align_type is AlignmentTypes.MULTIPLE:
        return 1
    else:
        return 0


def load_testing_info():
    #  TO_DO: get correct SPPDB file
    ppdb_path = 'simple_ppdbs/en_sppdb.json'
    with open(ppdb_path, 'r') as f:
        ppdb = json.load(f)
    ppdb["absentee ballot"] = [[-1, ["vote", "from", "home"]]]
    ppdb["hire me"] = [[-1, ["give", "me", "a", "job"]]]
    nlp = spacy.load("en_core_web_md")
    # config = {"model": DEFAULT_TOK2VEC_MODEL}
    # nlp.add_pipe("tok2vec", config=config)
    test_examples = [
        "Thanks in part to high-profile appeals by celebs such as will.i.am, Sean Penn, and Shakira, 73 percent of all donors to Haiti relief were younger than 50, according to a Pew survey.	Famous people such as will.i.am, Sean Penn and Shakira asked people to give money. Three-quarters of the people who gave to help Haiti were younger than 50.",
        "\n	People will want to hear me.",
        "So move all your furniture out of the way.	\n",
        "I mean, look at me. I'm a triple threat. And I don't just mean the acting and the singing and the writing.	Look at me for example. I am acting and singing and writing.",
        "It is in collaboration with Queens Theater and features Queens Theater for All Initiative.	This piece was created together with Queens Theater as part of the Queens Theater for All project.",
        "We have much to celebrate this July, but we also know that advocacy efforts must continue.	We are celebrating now, but we know that there is still a lot of work to do.",
        "I followed his lead and wrote my own material, knowing that without doing so, people might not give me that first break and hire me otherwise.	He taught me that I should write my own shows so people will want to give me a job.",
        "Know that there are absentee ballot options available, and there may be other options available depending on what situation we find ourselves in.	You should know if you could vote from home. You should check if there are other ways of voting this year because of covid-19.",
        "So therefore, you get to choose how you wish to enter into this field and decide what your story's going to be.	We can choose how to become a doctor or a nurse. You can tell your own story."]
    test_annotations = [(['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'C', 'C', 'C', 'C', 'C', 'C', 'D', 'C', 'C',
                          'D', 'D', 'D', 'D', 'D', 'D', 'D', 'C', 'D', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D', 'D',
                          'C'],
                         ['A', 'A', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
                          'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'C', 'C']),
                        ([], ['A', 'A', 'A', 'A', 'A', 'A', 'A']),
                        (['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
                         []),
                        (['D', 'D', 'D', 'C', 'C', 'C', 'C', 'C', 'R', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D',
                          'D', 'C', 'C', 'D', 'C', 'C', 'D', 'C', 'C'],
                         ['C', 'C', 'C', 'A', 'A', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']),
                        (['D', 'D', 'D', 'D', 'C', 'C', 'C', 'D', 'D', 'C', 'C', 'C', 'C', 'D', 'C'],
                         ['A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'A', 'C']),
                        (['R', 'R', 'D', 'D', 'R', 'D', 'D', 'C', 'C', 'C', 'D', 'C', 'C', 'R', 'D', 'D', 'D', 'C'],
                         ['C', 'C', 'C', 'A', 'C', 'C', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'C', 'A', 'A', 'A', 'A',
                          'C']),
                        (['M', 'R', 'D', 'D', 'D', 'R', 'M', 'M', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'M', 'D', 'D',
                          'M', 'M', 'M', 'D', 'D', 'D', 'D', 'C', 'D', 'C'],
                         ['A', 'C', 'C', 'C', 'C', 'A', 'C', 'C', 'C', 'A', 'A', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'A',
                          'C']),
                        (['C', 'D', 'M', 'M', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'D',
                          'D', 'D', 'D', 'D', 'R', 'C'],
                         ['A', 'A', 'C', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'A', 'C', 'C', 'A', 'A', 'A',
                          'A', 'A', 'A', 'A', 'C', 'A', 'C']),
                        (['D', 'D', 'D', 'D', 'D', 'D', 'C', 'C', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'C',
                          'C', 'D', 'D', 'D', 'D', 'C'],
                         ['A', 'A', 'C', 'C', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'A', 'C',
                          'C'])
                        ]
    test_align_types = [AlignmentTypes.EXPANSION,
                        AlignmentTypes.ADDITION,
                        AlignmentTypes.DELETION,
                        AlignmentTypes.SUMMARIZATION,
                        AlignmentTypes.BASIC,
                        AlignmentTypes.BASIC,
                        AlignmentTypes.BASIC,
                        AlignmentTypes.EXPANSION,
                        AlignmentTypes.EXPANSION]
    test_used_idx = [([5], []),
                     ([], []),
                     ([], []),
                     ([], []),
                     ([3, 13], []),
                     ([4], [2]),
                     ([5], [6]),
                     ([4, 5, 7, 12, 15], [15]),
                     ([9, 16], [])
                     ]
    return nlp, test_examples, test_annotations, test_align_types, test_used_idx, ppdb


def test_proximation():
    nlp, test_examples, _, _, _, _ = load_testing_info()
    for example in test_examples:
        test_reg, test_sim = example.split("\t")
        test_doc_reg = nlp(test_reg)
        test_doc_sim = nlp(test_sim)
        print(classify_proximation(test_doc_reg, test_doc_sim, 'en'))


def test_paraphrasing():
    nlp, test_examples, test_annotations, _, _, ppdb = load_testing_info()
    test_used_idx = [([], []),
                     ([], []),
                     ([], []),
                     ([], []),
                     ([], []),
                     ([4], [2]),
                     ([5], [6]),
                     ([12], [15]),
                     ([], [])
                     ]
    # TO_DO: Add annotations for the example sentences
    for example, annotation in zip(test_examples, test_annotations):
        test_reg, test_sim = example.split("\t")
        test_doc_reg = nlp(test_reg)
        test_doc_sim = nlp(test_sim)
        test_annot_reg, test_annot_sim = annotation
        print(classify_rephrasing(test_doc_reg, test_doc_sim, 'en', ppdb, test_annot_reg))


def test_deleting_info():
    nlp, test_examples, test_annotations, test_align_types, test_used_idx, ppdb = load_testing_info()
    # TO_DO: Add annotations for the example sentences
    for example, annotation, align_type, used_idx in zip(test_examples, test_annotations, test_align_types, test_used_idx):
        test_reg, test_sim = example.split("\t")
        test_doc_reg = nlp(test_reg)
        test_doc_sim = nlp(test_sim)
        test_annot_reg, test_annot_sim = annotation
        test_used_reg, test_used_sim = used_idx
        print(classify_deleting_info(test_doc_reg, test_doc_sim, 'en', align_type, test_annot_reg, test_used_reg))


def test_adding_info():
    nlp, test_examples, test_annotations, test_align_types, test_used_idx, ppdb = load_testing_info()
    for example, annotation, align_type, used_idx in zip(test_examples, test_annotations, test_align_types, test_used_idx):
        test_reg, test_sim = example.split("\t")
        test_doc_reg = nlp(test_reg)
        test_doc_sim = nlp(test_sim)
        test_annot_reg, test_annot_sim = annotation
        test_used_reg, test_used_sim = used_idx
        print(classify_adding_info(test_doc_reg, test_doc_sim, 'en', align_type, test_annot_sim, test_used_sim))


def test_intra_reorder():
    nlp, test_examples, test_annotations, test_align_types, test_used_idx, ppdb = load_testing_info()
    for example, annotation, align_type, used_idx in zip(test_examples, test_annotations, test_align_types, test_used_idx):
        test_reg, test_sim = example.split("\t")
        test_doc_reg = nlp(test_reg)
        test_doc_sim = nlp(test_sim)
        test_annot_reg, test_annot_sim = annotation
        test_used_reg, test_used_sim = used_idx
        print(classify_intra_sentence_rearrange(test_doc_reg, test_doc_sim, 'en', test_annot_reg, test_annot_sim))


def test_explicitation():
    nlp, test_examples, test_annotations, test_align_types, test_used_idx, ppdb = load_testing_info()
    nlp.add_pipe('coreferee')
    for example, annotation, align_type, used_idx in zip(test_examples, test_annotations, test_align_types, test_used_idx):
        test_reg, test_sim = example.split("\t")
        test_doc_reg = nlp(test_reg)
        test_doc_sim = nlp(test_sim)
        test_annot_reg, test_annot_sim = annotation
        test_used_reg, test_used_sim = used_idx
        print(classify_explicitation(test_doc_reg, test_doc_sim, 'en', nlp, align_type))


def test_access():
    nlp, test_examples, test_annotations, test_align_types, test_used_idx, ppdb = load_testing_info()
    nlp.add_pipe('coreferee')
    with open(spacy_lookups_data.en["lexeme_prob"]) as f:
        lex_probability = json.load(f)
    for example, annotation, align_type, used_idx in zip(test_examples, test_annotations, test_align_types,
                                                         test_used_idx):
        test_reg, test_sim = example.split("\t")
        test_doc_reg = nlp(test_reg)
        test_doc_sim = nlp(test_sim)
        test_annot_reg, test_annot_sim = annotation
        test_used_reg, test_used_sim = used_idx
        print(f"{get_nbchars_ratio(test_doc_reg, test_doc_sim, align_type)=}"
              f"\t{get_levsim(test_doc_reg, test_doc_sim, align_type)=}"
              f"\t{get_wordrank_ratio(test_doc_reg, test_doc_sim, lex_probability, align_type)=}"
              f"\t{get_deptree_depth_ratio(test_doc_reg, test_doc_sim, align_type)=}")


"""
Use the lines below to test each of the functions
"""
# if __name__ == '__main__':
#     # TO_DO: Think of how to preprocess the sentences in the dataset (for each function and sub-classification)
#     # TO_DO: In areas where word meaning count (such as word rank / moved words) remove punctuations
#     # TO_DO: Cleanup comment outs
#     time.sleep(1)
#     LOGGER.warning("Proximation Test")
#     test_proximation()
#     time.sleep(0.001)
#     LOGGER.warning("Paraphrasing Test")
#     test_paraphrasing()
#     time.sleep(0.001)
#     LOGGER.warning("Deletion Test")
#     test_deleting_info()
#     time.sleep(0.001)
#     LOGGER.warning("Adding Test")
#     test_adding_info()
#     time.sleep(0.001)
#     LOGGER.warning("Intra-reorder Test")
#     test_intra_reorder()
#     time.sleep(0.001)
#     LOGGER.warning("Explicitation Test")
#     test_explicitation()
#     time.sleep(0.001)
#     LOGGER.warning("ACCESS Test")
#     test_access()
