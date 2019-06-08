# BioRelEx: Biological Relation Extraction Benchmark

BioRelEx is a dataset of 2000+ sentences from biological journals with complete annotations of proteins, genes, chemicals and other entities along with binding interactions between them.

A paper describing the dataset is accepted at ACL BioNLP Workshop 2019.

## Dataset format

Training and development sets are provided as JSON files. Each version of the dataset is one [release](https://github.com/YerevaNN/BioRelEx/releases) of this repository.

Each JSON file is a list of objects, one per sentence. _More details will be added soon._

## Evaluation

We propose two main metrics for evaluation, one for **entity recognition** and another one for **relation extraction**. We provide a [script](https://github.com/YerevaNN/BioRelEx/blob/master/evaluate.py) for the main evaluation metrics and several additional metrics designed for error analysis.

As the test set is not released, we are going to setup an evaluation server. 

## Baselines

The paper describes two non-trivial baselines. One is an existing rule-based system called [REACH](https://github.com/clulab/reach), and the other one is based on a neural multitask architecture called [SciIE](http://nlp.cs.washington.edu/sciIE/). The baselines are implemented in [another repository](https://github.com/YerevaNN/Relation-extraction-pipeline).

## Citation

_Will be available soon._
