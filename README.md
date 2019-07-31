# BioRelEx: Biological Relation Extraction Benchmark

BioRelEx is a dataset of 2000+ sentences from biological journals with complete annotations of proteins, genes, chemicals and other entities along with binding interactions between them.

[A paper describing the dataset](https://www.aclweb.org/anthology/papers/W/W19/W19-5019/) is accepted at [ACL BioNLP Workshop 2019](https://aclweb.org/aclwiki/BioNLP_Workshop).

We invite everyone to submit their relation extraction systems to [our Codalab competition](https://competitions.codalab.org/competitions/20468).

## Dataset format

Training and development sets are provided as JSON files. Each version of the dataset is one [release](https://github.com/YerevaNN/BioRelEx/releases) of this repository.

Each JSON file is a list of objects, one per sentence. _More details will be added soon._

## Evaluation

We propose two main metrics for evaluation, one for **entity recognition** and another one for **relation extraction**. We provide a [script](https://github.com/YerevaNN/BioRelEx/blob/master/evaluate.py) for the main evaluation metrics and several additional metrics designed for error analysis.

The test set is not released. **Please submit your solution in [this Codalab competition](https://competitions.codalab.org/competitions/20468).**

## Baselines

The paper describes two non-trivial baselines. One is an existing rule-based system called [REACH](https://github.com/clulab/reach), and the other one is based on a neural multitask architecture called [SciIE](http://nlp.cs.washington.edu/sciIE/). The baselines are implemented in [another repository](https://github.com/YerevaNN/Relation-extraction-pipeline).

## Citation

If you use the dataset, please cite:

    @inproceedings{khachatrian-etal-2019-biorelex,
        title = "{B}io{R}el{E}x 1.0: Biological Relation Extraction Benchmark",
        author = "Khachatrian, Hrant  and Nersisyan, Lilit  and Hambardzumyan, Karen  and Galstyan, Tigran  and Hakobyan, Anna  and Arakelyan, Arsen  and Rzhetsky, Andrey  and Galstyan, Aram",
        booktitle = "Proceedings of the 18th BioNLP Workshop and Shared Task",
        month = aug,
        year = "2019",
        address = "Florence, Italy",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/W19-5019",
        pages = "176--190"
    }

  
