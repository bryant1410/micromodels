"""
Experiment Script
"""
from typing import Tuple, List, Mapping, Any

import os
import json
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from nltk import tokenize

from mm.EmpathyClassifier2 import EmpathyClassifier, SEED
from mm.config import MM_CONFIG

#SEED = 420
#random.seed(SEED)


def build_configs(orig_config, train_data):
    """
    Dynamically build micromodel config files.
    More specifically, set bert-query micromodel's seed values
    based on train_data.
    """
    all_rationales = {
        "emotional_reactions": {
            "1": [],
            "2": [],
        },
        "interpretations": {
            "1": [],
            "2": [],
        },
        "explorations": {
            "1": [],
            "2": [],
        },
    }
    for instance in train_data:
        for task in ["emotional_reactions", "explorations", "interpretations"]:
            level = instance[task]["level"]
            if level != "0":
                rationales = instance[task]["rationales"].split("|")
                rationales = [x for x in rationales if x != ""]
                all_rationales[task][level].extend(rationales)
    for config in orig_config:
        if config["name"] == "empathy_interpretations_1":
            config["setup_args"]["seed"] = all_rationales["interpretations"][
                "1"
            ]
        if config["name"] == "empathy_interpretations_2":
            config["setup_args"]["seed"] = all_rationales["interpretations"][
                "2"
            ]
        if config["name"] == "empathy_explorations_1":
            config["setup_args"]["seed"] = all_rationales["explorations"]["1"]
        if config["name"] == "empathy_explorations_2":
            config["setup_args"]["seed"] = all_rationales["explorations"]["2"]
        if config["name"] == "empathy_emotional_reactions_1":
            config["setup_args"]["seed"] = all_rationales[
                "emotional_reactions"
            ]["1"]
        if config["name"] == "empathy_emotional_reactions_2":
            config["setup_args"]["seed"] = all_rationales[
                "emotional_reactions"
            ]["2"]
    return orig_config


def load_data(
    data_path: str = None, **kwargs
) -> Tuple[
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
]:
    """
    Load IMDB data, including sentence-tokenizing the input text and
    splitting the data into train and test splits.

    :param data_path: Filepath to imdb data in json form.
    :param train_ratio: ratio of data to use for training.
    :param val_ratio: ratio of data to use for validation.
    :return: A pair of lists, one for training and one for test.
        Each list contains a tuple pair, which represents a single
        instance of data point. The first element is a list of sentences
        and the second element is the label.
    """
    train_ratio = kwargs.get("train_ratio", 0.7)
    val_ratio = kwargs.get("val_ratio", 0.15)

    with open(data_path, "r") as file_p:
        data = json.load(file_p)

    for _, data_obj in data.items():
        data_obj["seeker_tokenized"] = tokenize.sent_tokenize(
            data_obj["seeker_post"]
        )
        data_obj["response_tokenized"] = tokenize.sent_tokenize(
            data_obj["response_post"]
        )

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    keys = list(data.keys())
    random.shuffle(keys)

    train_keys = keys[:train_size]
    val_keys = keys[train_size : train_size + val_size]
    test_keys = keys[train_size + val_size :]

    train = [data[key] for key in train_keys]
    val = [data[key] for key in val_keys]
    test = [data[key] for key in test_keys]

    return train, val, test


def setup():
    """
    Setup
    """
    mm_base_path = os.environ.get("MM_HOME")
    emp_base_path = os.environ.get("EMP_HOME")
    emp_data_path = os.path.join(emp_base_path, "dataset/combined_iob.json")

    clf = EmpathyClassifier(mm_base_path)
    train_data, val_data, test_data = clf.load_data(
        emp_data_path, train_ratio=0.7, val_ratio=0.15
    )

    config = build_configs(MM_CONFIG, train_data)
    #config = config[2:8]
    clf.set_configs(config)
    emp_base_path = os.environ.get("EMP_HOME")
    feature_dir = os.path.join(emp_base_path, "mm/featurized")
    config_path = os.path.join(feature_dir, "config_%s_wtf3.json" % str(SEED))
    with open(config_path, "w") as file_p:
        _config = [x for x in config if x["name"].startswith("empathy")]
        json.dump(_config, file_p)

    return clf, config, train_data, val_data, test_data


def featurize(clf, train_data, val_data, test_data):
    """
    Featurize data.
    """
    emp_base_path = os.environ.get("EMP_HOME")
    feature_dir = os.path.join(emp_base_path, "mm/featurized")

    print("Featurizing training data")
    featurized = clf.featurize_data(train_data)
    featurized["original_data"] = train_data
    clf.dump_features(
        featurized, os.path.join(feature_dir, "train_%s_wtf3.json" % str(SEED))
    )
    print("Featurizing val data")
    featurized = clf.featurize_data(val_data)
    featurized["original_data"] = val_data
    clf.dump_features(
        featurized, os.path.join(feature_dir, "val_%s_wtf3.json" % str(SEED))
    )
    print("Featurizing test data")
    featurized = clf.featurize_data(test_data)
    featurized["original_data"] = test_data
    clf.dump_features(
        featurized, os.path.join(feature_dir, "test_%s_wtf3.json" % str(SEED))
    )


def run(clf):
    """
    Train and test.
    """
    emp_base_path = os.environ.get("EMP_HOME")
    feature_dir = os.path.join(emp_base_path, "mm/featurized")
    train_features = os.path.join(feature_dir, "train_%s_wtf3.json" % str(SEED))
    dev_features = os.path.join(feature_dir, "val_%s_wtf3.json" % str(SEED))
    test_features = os.path.join(feature_dir, "test_%s_wtf3.json" % str(SEED))

    train_features = clf.load_features(train_features)
    dev_features = clf.load_features(dev_features)
    test_features = clf.load_features(test_features)
    x_train = train_features["feature_vector"]
    x_dev = dev_features["feature_vector"]
    x_test = test_features["feature_vector"]

    # TODO: Cleaner way of doing this
    #x_train = x_train[:, [2, 3, 4, 5, 6, 7]]
    #x_dev = x_dev[:, [2, 3, 4, 5, 6, 7]]
    #x_test = x_test[:, [2, 3, 4, 5, 6, 7]]

    y_train_emotional_reactions = [x[0] for x in train_features["labels"]]
    y_dev_emotional_reactions = [x[0] for x in dev_features["labels"]]
    y_test_emotional_reactions = [x[0] for x in test_features["labels"]]

    #clf.fit(x_train, y_train_emotional_reactions)
    clf.train_featurized(x_train, y_train_emotional_reactions)
    print("Testing emotional reactions, test")
    print(
        json.dumps(
            clf._test_featurized(x_test, y_test_emotional_reactions), indent=2
        )
    )

    print("Testing emotional reactions, dev")
    print(
        json.dumps(
            clf._test_featurized(x_dev, y_dev_emotional_reactions), indent=2
        )
    )

    y_train_interpretations = [x[1] for x in train_features["labels"]]
    y_dev_interpretations = [x[1] for x in dev_features["labels"]]
    y_test_interpretations = [x[1] for x in test_features["labels"]]
    #clf.fit(x_train, y_train_interpretations)
    clf.train_featurized(x_train, y_train_interpretations)

    print("Testing interpretations, test")
    print(
        json.dumps(
            clf._test_featurized(x_test, y_test_interpretations), indent=2
        )
    )
    print("Testing interpretations, dev")
    print(
        json.dumps(clf._test_featurized(x_dev, y_dev_interpretations), indent=2)
    )

    y_train_explorations = [x[2] for x in train_features["labels"]]
    y_dev_explorations = [x[2] for x in dev_features["labels"]]
    y_test_explorations = [x[2] for x in test_features["labels"]]
    #clf.fit(x_train, y_train_explorations)
    clf.train_featurized(x_train, y_train_explorations)
    print("Testing explorations, test")
    print(
        json.dumps(clf._test_featurized(x_test, y_test_explorations), indent=2)
    )
    print("Testing explorations, dev")
    print(json.dumps(clf._test_featurized(x_dev, y_dev_explorations), indent=2))


def get_rationales(dev_features, task):
    """
    Collect rationales
    """
    spans = []
    for data in dev_features["original_data"]:
        span = data[task]["iob_format"]
        spans.append(span)
    return spans


def get_spans(array):
    """
    Get spans
    """
    spans = []
    span_start_idx = -1
    span_end_idx = -1
    for inner_idx, inner_elem in enumerate(array):
        if inner_elem == 1:
            if span_start_idx == -1:
                span_start_idx = inner_idx
            else:
                continue
        if inner_elem == 0:
            if span_start_idx == -1:
                continue
            span_end_idx = inner_idx
            spans.append((span_start_idx, span_end_idx))

            span_start_idx = -1
            span_end_idx = -1
    if span_start_idx != -1:
        span_end_idx = inner_idx
        spans.append((span_start_idx, span_end_idx))
    return spans


def _f1(_p, _r):
    if _p == 0 or _r == 0:
        return 0
    return 2 * _p * _r / (_p + _r)


def iou_f1(predictions, gold, threshold=0.5):
    """
    IOU-F1
    """
    assert len(predictions) == len(gold)
    all_f1_vals = []
    for idx, pred in tqdm(enumerate(predictions)):
        gold_instance = gold[idx]
        assert len(pred) == len(gold_instance)

        pred_spans = get_spans(pred)
        gold_spans = get_spans(gold_instance)

        ious = defaultdict(dict)
        for pred_span in pred_spans:
            best_iou = 0.0
            for gold_span in gold_spans:
                num = len(
                    set(range(pred_span[0], pred_span[1]))
                    & set(range(gold_span[0], gold_span[1]))
                )
                denom = len(
                    set(range(pred_span[0], pred_span[1]))
                    | set(range(gold_span[0], gold_span[1]))
                )
                iou = 0 if denom == 0 else num / denom

                if iou > best_iou:
                    best_iou = iou
            ious[pred_span] = best_iou

        threshold_tps = sum(int(x >= threshold) for x in ious.values())

        micro_r = threshold_tps / len(gold_spans) if len(gold_spans) > 0 else 0
        micro_p = threshold_tps / len(pred_spans) if len(pred_spans) > 0 else 0
        micro_f1 = _f1(micro_r, micro_p)
        if len(pred_spans) == 0 and len(gold_spans) == 0:
            all_f1_vals.append(1)
        else:
            all_f1_vals.append(micro_f1)

    return np.mean(all_f1_vals)


def t_f1(predictions, gold):
    """
    T-F1
    """
    assert len(predictions) == len(gold)
    all_f1s = []
    _all_f1s = []
    for idx, pred in tqdm(enumerate(predictions)):
        gold_instance = gold[idx]
        assert len(pred) == len(gold_instance)

        macro_f1 = f1_score(gold_instance, pred, average="macro")
        _f1 = f1_score(gold_instance, pred, zero_division=1)
        all_f1s.append(macro_f1)
        _all_f1s.append(_f1)
    print("Testing")
    print(np.mean(all_f1s))
    print(np.mean(_all_f1s))
    print("--")
    return np.mean(all_f1s)


def evaluate(predictions, gold, iou_threshold=0.5):
    """
    T-F1, IOU-F1
    """
    t_f1_score = t_f1(predictions, gold)
    iou_f1_score = iou_f1(predictions, gold, threshold=iou_threshold)
    return t_f1_score, iou_f1_score

def rationales_experiment(clf):
    """
    Rationale experiment.
    """
    emp_base_path = os.environ.get("EMP_HOME")
    feature_dir = os.path.join(emp_base_path, "mm/featurized")
    dev_features_file = os.path.join(feature_dir, "val_%s.json" % str(SEED))
    dev_features = clf.load_features(dev_features_file)

    rationales = clf.extract_rationale(dev_features, "emotional_reactions")
    groundtruth_rationales = get_rationales(
        dev_features, "emotional_reactions"
    )
    score = evaluate(rationales, groundtruth_rationales)
    print(score)

    rationales = clf.extract_rationale(dev_features, "explorations")
    groundtruth_rationales = get_rationales(dev_features, "explorations")
    score = evaluate(rationales, groundtruth_rationales)
    print(score)

    rationales = clf.extract_rationale(dev_features, "interpretations")
    groundtruth_rationales = get_rationales(dev_features, "interpretations")
    score = evaluate(rationales, groundtruth_rationales)
    print(score)



def main():
    """ Driver """
    clf, config, train, val, test = setup()
    featurize(clf, train, val, test)
    run(clf)
    # rationales_experiment(clf)
    #emp_base_path = os.environ.get("EMP_HOME")
    #feature_dir = os.path.join(emp_base_path, "mm/featurized")
    #dev_features_file = os.path.join(feature_dir, "val_%s.json" % str(SEED))
    #exploration_2_hits = clf.inspect_provenance(dev_features_file, "empathy_explorations_2_bert_query")
    #breakpoint()
    #clf.run_micromodel(config[3], [exploration_2_hits])
    #breakpoint()

if __name__ == "__main__":
    main()
