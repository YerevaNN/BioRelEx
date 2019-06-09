#!/usr/bin/python

"""
Original evaluation script for BioRelEx: Biomedical Relation Extraction Benchmark.
Only Python 3.5+ supported.

Copyright (c) ...
"""


import re
import json
import argparse

import numpy as np

from tqdm import tqdm

from sklearn import utils as sk_utils

from typing import Tuple, Set, Dict, Any, Union, Iterable, Callable

Hash = Union[int, str]
Number = Union[int, float]
JSON_Object = Dict[str, Any]
Mention = Tuple[str, int, int]


def hash_sentence(item: JSON_Object, match_by: str = 'id') -> Hash:
    """
    Get hash of sentence object for matching. Default is id.
    Useful for debugging and/or when id's are changed.

    :param item: Object to calculate hash for
    :param match_by: Matching criteria / method
    :return: Hash representation for the object
    """
    if match_by == 'text':
        text = item['text']
        text = text.lower()
        text = re.sub('[^0-9a-z]+', '', text)
        return hash(text)
    else:
        return str(item[match_by])


def get_sentences(data: Iterable[JSON_Object],
                  match_by: str) -> Dict[Hash, JSON_Object]:
    """
    Collect sentence objects w.r.t. matching criteria.
    :param data: Iterable of sentence objects
    :param match_by: Matching criteria / method
    :return: Dict of hash: sentence objects
    """
    return {
        hash_sentence(sentence, match_by): sentence
        for sentence in data
    }


def get_entity_mentions(sentence: JSON_Object) -> Set[Mention]:
    """
    Get all entity mentions given the sentence object.
    :param sentence: Sentence object
    :return: Tuple of entity name, inclusive start and exclusive end indices
    """
    return {
        (alias, start, end)
        for cluster in sentence['entities']
        for alias, entity in cluster['names'].items()
        for start, end in entity['mentions']
        if entity['is_mentioned']
    }


def unordered_pair(a, b):
    """
    Build unordered pair. Useful for matching pairs.
    :param a: Object
    :param b: Object
    :return: Normalized (Sorted) state of the tuple
    """
    if a > b:
        return b, a
    else:
        return a, b


def get_entity_coreferences(sentence: JSON_Object) -> Set[Tuple[str, str]]:
    """
    Get all the entity coreferences.
    :return: Tuples of unordered pairs of coreference participants
    """
    return {
        unordered_pair(a, b)
        for cluster in sentence['entities']
        for a in cluster['names']
        for b in cluster['names']
        if a != b
        and cluster['names'][a]['is_mentioned']
        and cluster['names'][b]['is_mentioned']
    }


# noinspection PyPep8Naming
class PRFScores(object):
    """
    Store and calculate Precision / Recall and F_1 scores.
    Supports namespaces (useful for different files / runs / sets).
    """
    def __init__(self, name: str):
        self.name = name

        self.TP = 0
        self.FN = 0
        self.FP = 0

        self.by_id = {}

    def store_by_id(self, id: Hash,
                    TP: int, FN: int, FP: int):
        if id not in self.by_id:
            self.by_id[id] = PRFScores(self.name)

        self.by_id[id].TP += TP
        self.by_id[id].FN += FN
        self.by_id[id].FP += FP

    def add_sets(self, id: Hash,
                 truth_set: Set[Any],
                 prediction_set: Set[Any]):
        """
        Update state of the score: store new sets
        :param id: Namespace id
        :param truth_set: Set of truth data
        :param prediction_set: Set of predictions
        """
        intersection = truth_set & prediction_set

        TP = len(intersection)
        FN = len(truth_set) - TP
        FP = len(prediction_set) - TP

        self.TP += TP
        self.FN += FN
        self.FP += FP
        self.store_by_id(id, TP, FN, FP)

    def get_scores(self) -> Dict[str, Number]:
        """
        Calculate scores w.r.t. current state
        :return: Dict of { metric name : metric value }
        """
        if self.TP + self.FP == 0:
            precision = 0
        else:
            precision = self.TP / (self.TP + self.FP)

        if self.TP + self.FN == 0:
            recall = 0
        else:
            recall = self.TP / (self.TP + self.FN)

        if precision + recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'f_score': f_score,
            'TP': self.TP,
            'FN': self.FN,
            'FP': self.FP
        }

    def print_scores(self):
        """
        Calculate score and print the results.
        """
        print('\n')
        print(self.name)

        print('          | Pred 0 | Pred 1')
        print('   True 0 |        | {:>6}'.format(self.FP))
        print('   True 1 | {:>6} | {:>6}'.format(self.FN, self.TP))

        scores = self.get_scores()

        print('      Precision: {:>5.2f}%\n'
              '      Recall:    {:>5.2f}% \n'
              '      F-score:   {:>5.2f}%'.format(scores['precision'] * 100,
                                                  scores['recall'] * 100,
                                                  scores['f_score'] * 100))


# noinspection PyPep8Naming
class PRFScoresFlatMentions(PRFScores):
    def add_sets(self, id: Hash,
                 truth_set: Set[Mention],
                 prediction_set: Set[Mention]):
        intersection = truth_set & prediction_set
        # remove the ones which intersect with TPs
        intersection_with_truth = {
            (e, start, end) for e, start, end in truth_set
            for c_e, c_start, c_end in intersection
            if c_start < end and c_end > start and e != c_e
        }
        intersection_with_pred = {
            (e, start, end) for e, start, end in prediction_set
            for c_e, c_start, c_end in intersection
            if c_start < end and c_end > start and e != c_e
        }

        truth_set -= intersection_with_truth
        prediction_set -= intersection_with_pred

        # remove the ones that are in a larger entity
        shorts_in_truth = {
            (e1, start1, end1) for e1, start1, end1 in truth_set
            for e2, start2, end2 in truth_set
            if end2 <= start1 <= start2 and e1 != e2
        }
        shorts_in_pred = {
            (e1, start1, end1) for e1, start1, end1 in prediction_set
            for e2, start2, end2 in prediction_set
            if end2 <= start1 <= start2 and e1 != e2
        }

        truth_set -= shorts_in_truth
        prediction_set -= shorts_in_pred

        TP = len(intersection)
        FN = len(truth_set) - TP
        FP = len(prediction_set) - TP

        self.TP += TP
        self.FN += FN
        self.FP += FP

        self.store_by_id(id, TP, FN, FP)


def evaluate_sentences(truth_sentences: Dict[Hash, Dict[str, Any]],
                       pred_sentences: Dict[Hash, Dict[str, Any]],
                       keys: Iterable[Hash] = None) -> Tuple[PRFScores, PRFScores]:
    relex_any_score = PRFScores('Relation Extraction (any)')
    relex_all_score = PRFScores('Relation Extraction (all)')
    mentions_score = PRFScores('Entity Mentions')
    mentions_flat_score = PRFScoresFlatMentions('Entity Mentions (flat)')
    entities_score = PRFScores('Entities')
    coref_score = PRFScores('Entity Coreferences')

    if keys is None:
        keys = truth_sentences.keys()

    for id in keys:
        # match unique entities
        if id not in pred_sentences:
            print('No prediction for sentence with ID={}'.format(id))
            continue

        truth = truth_sentences[id]
        pred = pred_sentences[id]

        truth_entity_mentions = get_entity_mentions(truth)
        pred_entity_mentions = get_entity_mentions(pred)

        mentions_score.add_sets(id, truth_entity_mentions, pred_entity_mentions)
        mentions_flat_score.add_sets(id, truth_entity_mentions, pred_entity_mentions)

        st_entities = {entity for entity, start, end in truth_entity_mentions}
        sp_entities = {entity for entity, start, end in pred_entity_mentions}
        entities_score.add_sets(id, st_entities, sp_entities)

        st_entity_coreferences = get_entity_coreferences(truth)
        sp_entity_coreferences = get_entity_coreferences(pred)
        coref_score.add_sets(id, st_entity_coreferences, sp_entity_coreferences)

        # pred_ue_to_truth_ue = {}
        #
        # for ue, ue_obj in pred['unique_entities'].items():
        #     ue = int(ue)
        #     for ve, ve_obj in ue_obj['versions'].items():
        #         if ve in truth['entity_map']:
        #             true_ue_id = int(truth['entity_map'][ve])
        #             if ue in pred_ue_to_truth_ue and pred_ue_to_truth_ue[ue] != true_ue_id:
        #                 # another version of this entity cluster was matched to a different cluster
        #                 entity_version_mismatch += 1
        #             else:
        #                 pred_ue_to_truth_ue[ue] = true_ue_id
        #         else:
        #             # pred_ue_to_truth_ue[ue] = -ue
        #             # this version does not exist in the ground truth
        #             fp_entities += 1

        # st_unique_entities = set([int(x) for x in truth['unique_entities'].keys()])
        # sp_unique_entities = set(pred_ue_to_truth_ue.values())
        # unique_entities_score.add_sets(st_unique_entities, sp_unique_entities)

        # interactions
        predicted_pairs_with_names = {
            unordered_pair(a, b)
            for interaction in pred['interactions']
            for a, a_meta in pred['entities'][interaction['participants'][0]]['names'].items()
            for b, b_meta in pred['entities'][interaction['participants'][1]]['names'].items()
            if a_meta['is_mentioned'] and b_meta['is_mentioned']
        }
        # sometimes duplicates exist

        predicted_pairs_with_names_matched = set()

        for interaction in truth['interactions']:
            # if 'implicit' in interaction and interaction['implicit']:
            #     continue
            ta, tb = interaction['participants']
            true_pairs_with_names = {
                unordered_pair(a, b)
                for a, a_obj in truth['entities'][ta]['names'].items()
                if a_obj['is_mentioned']
                for b, b_obj in truth['entities'][tb]['names'].items()
                if b_obj['is_mentioned']
            }  # no duplicates detected

            intersection = true_pairs_with_names & predicted_pairs_with_names
            predicted_pairs_with_names_matched = predicted_pairs_with_names_matched | intersection

            true_to_add = {unordered_pair(ta, tb)}

            predicted_any_to_add = set()
            predicted_all_to_add = set()

            if intersection:
                predicted_any_to_add = true_to_add

            if len(intersection) == len(true_pairs_with_names):
                predicted_all_to_add = true_to_add

            relex_any_score.add_sets(id, true_to_add, predicted_any_to_add)
            relex_all_score.add_sets(id, true_to_add, predicted_all_to_add)

        predicted_pairs_with_names_unmatched = predicted_pairs_with_names - predicted_pairs_with_names_matched
        relex_any_score.add_sets(id, set(), predicted_pairs_with_names_unmatched)
        relex_all_score.add_sets(id, set(), predicted_pairs_with_names_unmatched)

        # TODO: check labels!

    return mentions_score, relex_all_score
    # , mentions_flat_score, entities_score,
    # coref_score, relex_any_score


class BootstrapEvaluation(object):
    def __init__(self, truth_objects: Dict[Hash, Dict[str, Any]],
                 prediction_objects: Dict[str, Dict[Hash, Dict[str, Any]]],
                 evaluate_fn: Callable,
                 bootstrap_count: int):
        self.bootstrap_count = bootstrap_count
        self.truth = truth_objects
        self.prediction_dict = prediction_objects
        self.evaluate_fn = evaluate_fn
        self.runs = {}
        self.results = {}
        self.score_types = ['precision', 'recall', 'f_score']

    def initialize_runs(self, name: str):
        self.runs[name] = {
            filename: {
                score_type: [] for score_type in self.score_types
            } for filename in self.prediction_dict
        }

    def add_run(self, filename: str, score: PRFScores):
        scores = score.get_scores()
        for score_type in self.score_types:
            self.runs[score.name][filename][score_type].append(scores[score_type])

    def evaluate(self) -> Dict[str, Dict[str, Dict[str, Dict[str, Number]]]]:
        keys = list(self.truth)
        print('Starting to bootstrap for {} times'.format(self.bootstrap_count))
        for _ in tqdm(range(self.bootstrap_count)):
            cur_keys = sk_utils.resample(keys, n_samples=len(keys))
            for filename, prediction in self.prediction_dict.items():
                all_scores = self.evaluate_fn(self.truth, prediction, cur_keys)
                for score in all_scores:
                    if not isinstance(score, PRFScores):
                        continue
                    if score.name not in self.runs:
                        self.initialize_runs(score.name)
                    self.add_run(filename, score)

        self.results = {}
        for score_name, score_data in self.runs.items():
            self.results[score_name] = {}
            for filename in self.prediction_dict.keys():
                self.results[score_name][filename] = {}
                for score_type, values in score_data[filename].items():
                    self.results[score_name][filename][score_type] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        '2.5%': np.percentile(values, 2.5),
                        '97.5%': np.percentile(values, 97.5),
                    }

        print('Bootstrapping completed')

        return self.results

    def print_results(self):
        for filename in self.prediction_dict:
            print('\n{}'.format(filename))
            for score_name, score_obj in self.results.items():
                print('   {} (n={})'.format(score_name, self.bootstrap_count))
                for score_type, score_stats in score_obj[filename].items():
                    print('      {:<10} {:>5.2f} ± {:>5.2f} ({:5.2f} - {:5.2f})'.format(
                        score_type,
                        score_stats['mean'] * 100,
                        score_stats['std'] * 100,
                        score_stats['2.5%'] * 100,
                        score_stats['97.5%'] * 100,
                    ))

        for score_name, score_obj in self.results.items():
            print('\n{} (n={})'.format(score_name, self.bootstrap_count))

            for score_type in self.score_types:
                print('   {:<10} {:>23}: '.format(score_type, ' '), end='')

                for idx, _ in enumerate(self.prediction_dict):
                    print('({}) '.format(idx + 1), end='')

                print(' ')
                for idx, filename1 in enumerate(self.prediction_dict):
                    print('   ({}) {:>30}: '.format(idx + 1, filename1[-30:]), end='')

                    for filename2 in self.prediction_dict:
                        if filename1 == filename2:
                            cell = ''
                        else:
                            file1_scores = self.runs[score_name][filename1][score_type]
                            file2_scores = self.runs[score_name][filename2][score_type]

                            cell = sum(file1_score >= file2_score
                                       for file1_score, file2_score
                                       in zip(file1_scores, file2_scores))

                        print('{:>3} '.format(cell), end='')

                    print(' ')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_path', '-t', required=True, type=str)
    parser.add_argument('--prediction_path', '-path', nargs='*', required=True, type=str)
    # parser.add_argument('--include_negatives', action='store_true')
    parser.add_argument('--bootstrap_count', default=0, type=int)
    # parser.add_argument('--multiword', default=0, type=int, help='values: +1 or -1')
    # parser.add_argument('--tags', default='', type=str, help='example: complex+1,abbr-1')
    # parser.add_argument('--has_sdg', default=0, type=int, help='+1 or -1')
    # parser.add_argument('--sentence_stats', action='store_true')
    parser.add_argument('--match_by', '-mb', default='id', type=str)

    args = parser.parse_args()

    print(args)

    # positive_labels = [-1, 1] if args.include_negatives else [1]

    with open(args.truth_path, 'r', encoding='utf-8') as f:
        truth = json.load(f)

    predictions = {}
    for path in args.prediction_path:
        with open(path, 'r', encoding='utf-8') as f:
            prediction = json.load(f)
            predictions[path] = prediction

    truth_sentences = get_sentences(truth, args.match_by)
    print('{} truth sentences read from {}. {} objects extracted'.format(len(truth),
                                                                         args.truth_path,
                                                                         len(truth_sentences)))
    pred_sentences_dict = {}
    for filename, prediction in predictions.items():
        pred_sentences = get_sentences(prediction, args.match_by)
        print('{} pred sentences read from {}. {} objects extracted'.format(len(prediction),
                                                                            filename,
                                                                            len(pred_sentences)))
        pred_sentences_dict[filename] = pred_sentences

    if args.bootstrap_count > 0:
        be = BootstrapEvaluation(truth_sentences, pred_sentences_dict,
                                 evaluate_sentences, args.bootstrap_count)
        be.evaluate()
        be.print_results()

    for filename, pred_sentences in pred_sentences_dict.items():
        print('\n' + '=' * 80)
        print('Results for {}:'.format(filename))
        scores = evaluate_sentences(truth_sentences, pred_sentences)

        for score in scores:
            score.print_scores()

        sentences_with_scores = []
        for sentence in pred_sentences.values():
            sentence['scores'] = {}
            for score in scores:
                key = hash_sentence(sentence, args.match_by)
                sentence['scores'][score.name] = score.by_id[key].get_scores()
            sentences_with_scores.append(sentence)

        with open(filename + '.scores', 'w', encoding='utf-8') as f:
            json.dump(sentences_with_scores, f, indent=True)


if __name__ == '__main__':
    main()
