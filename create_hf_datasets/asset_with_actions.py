# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ASSET: a dataset for sentence simplification evaluation"""


import csv
import json
import os

import datasets


# TO_DO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{alva-manchego-etal-2020-asset,
    title = "{ASSET}: {A} Dataset for Tuning and Evaluation of Sentence Simplification Models with Multiple Rewriting Transformations",
    author = "Alva-Manchego, Fernando  and
      Martin, Louis  and
      Bordes, Antoine  and
      Scarton, Carolina  and
      Sagot, Benoit  and
      Specia, Lucia",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.424",
    pages = "4668--4679",
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
ASSET is a dataset for evaluating Sentence Simplification systems with multiple rewriting transformations,
as described in "ASSET: A Dataset for Tuning and Evaluation of Sentence Simplification Models with Multiple Rewriting Transformations".
The corpus is composed of 2000 validation and 359 test original sentences that were each simplified 10 times by different annotators.
The corpus also contains human judgments of meaning preservation, fluency and simplicity for the outputs of several automatic text simplification systems.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://github.com/facebookresearch/asset"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "Creative Common Attribution-NonCommercial 4.0 International"

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL_LIST = [
    ("human_ratings.csv", "https://github.com/facebookresearch/asset/raw/master/human_ratings/human_ratings.csv"),
    ("asset.valid.orig", "https://github.com/facebookresearch/asset/raw/master/dataset/asset.valid.orig"),
    ("asset.test.orig", "https://github.com/facebookresearch/asset/raw/master/dataset/asset.test.orig"),
]
_URL_LIST += [
    (
        f"asset.{spl}.simp.{i}",
        f"https://github.com/facebookresearch/asset/raw/master/dataset/asset.{spl}.simp.{i}",
    )
    for spl in ["valid", "test"]
    for i in range(10)
]

_URLs = dict(_URL_LIST)

_DATA_DIR = "data/jsons"

_CORPORA = ("asset",)


class ASSETWithActionsConfig(datasets.BuilderConfig):
    """Builder config for ASSET with Actions"""

    def __init__(self, corpora=_CORPORA, actions=9, **kwargs):
        description = f"Dataset for Sentence Simplification, that include Cognitive Simplification actions.\n" \
                      f"includes data from the following corpora:\n{', '.join(corpora)}"

        super(ASSETWithActionsConfig, self).__init__(
            description=description,
            version=datasets.Version("1.1.0", ""),
            **kwargs,
        )
        self.corpora = corpora
        self.actions = actions


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class ASSETWithActions(datasets.GeneratorBasedBuilder):
    """
        TODO: Short description of my dataset.
        Simplification Datasets (WikiLarge + NewSela) with Simplification Actions
    """

    VERSION = datasets.Version("0.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    # BUILDER_CONFIGS = [
    #     datasets.BuilderConfig(name="first_domain", version=VERSION, description="This part of my dataset covers a first domain"),
    #     datasets.BuilderConfig(name="second_domain", version=VERSION, description="This part of my dataset covers a second domain"),
    # ]
    corpora_name_dict = {
        "all": _CORPORA,
        "asset": ("asset",)
    }
    action_nums = (0, 1, 9)
    action_dict = {0: "act-none", 1: "act-single", 9: "act-all"}
    # print(action_nums, corpora_name_dict, action_dict)
    BUILDER_CONFIGS = []
    for num in action_nums:
        for k, v in corpora_name_dict.items():
            fname = f"{k}_json_{action_dict[num]}"
            BUILDER_CONFIGS.append(ASSETWithActionsConfig(name=fname, corpora=v, actions=num))

    DEFAULT_CONFIG_NAME = "all_json_act-all"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TO_DO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        # if self.config.name == "first_domain":  # This is the name of the configuration selected in BUILDER_CONFIGS above
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option1": datasets.Value("string"),
        #             "answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        # else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option2": datasets.Value("string"),
        #             "second_domain_answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        features = datasets.Features(
            {
                "source": datasets.Value("string"),
                "target": datasets.Sequence(datasets.Value("string")),
                "actions": datasets.Sequence(datasets.Value("string")),
                "corpus": datasets.Value("string"),
                "entry_type": datasets.Value("int32")
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=self.config.description,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # my_urls = _URLs[self.config.name]
        # data_dir = dl_manager.download_and_extract(my_urls)
        data_dir = _DATA_DIR

        split2paths = {
            splitname: [os.path.join(data_dir, f"{corpus}-{split}.json")
                        for corpus in self.config.corpora
                        for split in ["test", "valid"]
                        if split == splitname
                        ]
            for splitname in ["test", "valid"]
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": split2paths["test"],
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepaths": split2paths["valid"],
                    "split": "valid"
                },
            ),
        ]

    def _generate_examples(
        self, filepaths, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        for file_id_, filepath in enumerate(filepaths):
            with open(filepath) as f:
                for row_id_, row in enumerate(f):
                    data = json.loads(row)
                    id_ = f"{file_id_}_{row_id_}"
                    if self.config.actions == 9:
                        yield id_, {
                            "source": f"{' '.join([f'<{a}>' for a in data['actions']])} {data['source']}",
                            "target": data['target'],
                            "actions": data['actions'],
                            "corpus": data['corpus'],
                            "entry_type": data['entry_type']
                        }
                    elif self.config.actions == 1:
                        yield id_, {
                            "source": f"{' '.join([f'<{a}>' for a in data['actions'] if a =='ADD' or a =='DEL'])} {data['source']}",
                            "target": data['target'],
                            "actions": data['actions'],
                            "corpus": data['corpus'],
                            "entry_type": data['entry_type']
                        }
                    else:  # self.config.actions == 0
                        yield id_, {
                            "source": f"{data['source']}",
                            "target": data['target'],
                            "actions": data['actions'],
                            "corpus": data['corpus'],
                            "entry_type": data['entry_type']
                        }
