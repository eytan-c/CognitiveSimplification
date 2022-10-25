from typing import List, Pattern
import numpy as np
import re


def sort_fnames_with_numbers(element, pattern: Pattern):
    m = pattern.match(element)
    if m is None:
        return "", 0
    if m.group(1) is None:
        return "a", m.group(2)
    else:
        return m.group(1), m.group(2)


def get_reorder_sent_id(idx_list: List):
    """
    return the indexes of sentences in the edited doc that are out of order
    - i.e. appear in different position from the unedited doc, or don't appear in the unedited doc.
    :param idx_list:
    :return:
    """
    # Get the LIS and parent list
    # longest_subsequence_list, parent_list = LIS(list(filter(lambda a: a != -3, idx_list)))
    longest_subsequence_list, parent_list = LIS(idx_list)
    if len(longest_subsequence_list) == 0:
        return []
    # Find the end of the LIS
    longest_subsequence_end = np.argmax(longest_subsequence_list)
    is_correct_order = np.zeros(len(idx_list))  # array to save which idxs are in the correct order
    is_correct_order[longest_subsequence_end] = 1  # the end of the LIS is always in order
    parent = parent_list[longest_subsequence_end]  # the parent of the end is also in order
    # Go back by parents, until reaching -1 (i.e. root of the parent path)
    while parent != -1:
        is_correct_order[parent] = 1
        parent = parent_list[parent]
    result = list(
        np.arange(len(idx_list))[np.logical_not(is_correct_order)])  # convert to list all idxs that are not in order
    # return list(filter(lambda a: a != 2, result))
    return result


# def LISKE(idx_list: List[int], k: int):
#     """
#     Implements the Longest
#     :param idx_list: list of sentence indexes to find longest increasing subsequence in
#     :param k: number of exceptions
#     :return:
#     """
#     dp = np.zeros((len(idx_list), k))


def LIS(idx_list: List[int]):
    """
    Implements the Longest Increasing Subsequence algorithm
    :param idx_list: list of sentence indexes to find longest increasing subsequence in
    :return:
    """
    if len(idx_list) == 0:
        return [], []
    parents = [-1 for i in range(len(idx_list))]
    LIS = [0 for i in range(len(idx_list))]
    LIS[0] = 1
    for i in range(1, len(idx_list)):
        max_LIS = -1
        parent_idx = -1
        for j in range(i):
            if idx_list[j] < idx_list[i]:
                local_max = LIS[j] + 1
                if local_max > max_LIS:
                    max_LIS = local_max
                    parent_idx = j
        parents[i] = parent_idx
        LIS[i] = max_LIS if max_LIS != -1 else 1
    return LIS, parents
