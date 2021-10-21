"""
Emotion Reaction Classification
"""
from typing import Tuple, List, Mapping, Any, Optional

import os
import json
import random

import numpy as np
from nltk import tokenize
from src.TaskClassifier import TaskClassifier, _to_binary_vectors
from src.metrics import f1
from mm.config import MM_CONFIG

# TODO: Clean up
ONLY_RESPONSES = True

SEED = 100
random.seed(SEED)


class EmpathyClassifier(TaskClassifier):
    """
    Classifier for Emotional Reaction
    """

    label_0 = "0"
    label_1 = "1"
    label_2 = "2"

    def __init__(
        self, mm_basepath: str, configs: List[Mapping[str, Any]] = None
    ) -> None:
        """
        Initialize IMDB sentiment classifier
        """
        super().__init__(mm_basepath, configs)

    # TODO: This interface is different from TaskClassifier's interface..
    def load_data(
        self, data_path: str = None, **kwargs
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
    


    def featurize_data(
        self, data: List[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        """
        NOTE: This function assumes the input includes seeker utterances.
        Otherwise, use parent's featurize_data method.

        Featurize data, where the input is a list of tuples.

        :param data: list of data instances as a list of tuples.
        The first element of the tuple is the seeker's list of utterances.
        The second element is the responder's list of utterances.
        The last element is their label.

        :return: A dictionary with the following format:
            {
                "binary_vectors": {
                    "micromodel_name_{r/s}: {
                        utterance_group_idx: List[int]
                    }, ...
                },
                "feature_vector": ndarray, shape (len(data), # of micromodels),
                "labels": List of labels
            }
        """
        seeker_utterances = [instance["seeker_tokenized"] for instance in data]
        response_utterances = [
            instance["response_tokenized"] for instance in data
        ]
        labels = [
            [
                instance["emotional_reactions"]["level"],
                instance["interpretations"]["level"],
                instance["explorations"]["level"],
            ]
            for instance in data
        ]

        # seeker_micromodel_output = self.run_micromodels(seeker_utterances)
        response_micromodel_output = self.run_micromodels(response_utterances)
        utterance_lengths = {
            idx: len(sentences)
            for idx, sentences in enumerate(response_utterances)
        }
        featurized = None

        for mm_output in response_micromodel_output.values():
            # mm_output: {utt_idx: list[int]}:
            # map utterance ids to list of matched indices

            # mm_output is a list of binary vectors, represented as ndarrays
            # Convert to actual binary vectors.
            mm_outputs = _to_binary_vectors(mm_output, utterance_lengths)

            feature_values = self.aggregator.aggregate(mm_outputs)
            if featurized is None:
                featurized = feature_values
            else:
                featurized = np.vstack([featurized, feature_values])

        featurized = np.transpose(featurized)
        return {
            "response_binary_vectors": response_micromodel_output,
            "feature_vector": featurized,
            "labels": labels,
        }

    def inspect_provenance(
        self, features_filepath: str, micromodel_name: str
    ) -> None:
        """
        Show the input text that corresponds to "hits" based on binary vectors.

        :param features_filepath: Filepath to features.
        :param micromodel_name: Name of the micromodel to inspect.
        """
        features = self.load_features(features_filepath)
        text_data = features["original_data"]
        binary_vectors = features["response_binary_vectors"]
        if micromodel_name not in binary_vectors:
            raise RuntimeError(
                "Could not find binary vectors for %s" % micromodel_name
            )
        binary_vectors = binary_vectors[micromodel_name]
        assert len(text_data) == len(binary_vectors)

        all_hits = []
        for idx, input_data in enumerate(text_data):
            # Input data is in the format of
            # [ ([sentence 1, sentence 2, ...], label), ...]
            sentences = input_data["response_tokenized"]
            hit_idxs = binary_vectors[idx]
            hits = np.array(sentences)[hit_idxs]
            all_hits.extend(hits)
        return all_hits

    def extract_rationale(self, features: str, task: str):
        """
        Extract rationales.

        TODO: Input should be features not features_filepath
        """
        orig_data = features["original_data"]
        binary_vectors = features["response_binary_vectors"]

        micromodel_names = [
            "empathy_%s_1_bert_query" % task,
            "empathy_%s_2_bert_query" % task,
        ]
        outputs = []
        for idx, input_data in enumerate(orig_data):
            response_sentences = input_data["response_tokenized"]
            hit_idxs = []
            for name in micromodel_names:
                if name not in binary_vectors:
                    raise RuntimeError(
                        "Could not find binary vectors for %s" % name
                    )
                mm_binary_vectors = binary_vectors[name]
                assert len(orig_data) == len(mm_binary_vectors)

                _hit_idxs = mm_binary_vectors[idx]
                hit_idxs.extend(_hit_idxs)
            hit_idxs = list(set(hit_idxs))

            iob = []
            for idx, sentence in enumerate(response_sentences):
                tokenized = tokenize.word_tokenize(sentence)
                if idx in hit_idxs:
                    iob.extend([1] * len(tokenized))
                else:
                    iob.extend([0] * len(tokenized))

            outputs.append(iob)
        return outputs

    def _test_featurized(
        self, featurized_data: np.ndarray, labels: List[str]
    ) -> Mapping[str, Any]:
        """
        Test task-specific classifier using already featurized data.

        :param featurized_data: ndarray of shape
            (len(input_data), number of micromodels).
        :param labels: list of labels.
        :return: test results.
        """
        predictions = []
        for row in featurized_data:
            prediction, _ = self.infer_featurized([row])
            predictions.append(prediction)

        num_correct = 0
        for idx, pred in enumerate(predictions):
            if pred == labels[idx]:
                num_correct += 1

        accuracy = num_correct / len(labels)
        return {
            "accuracy": accuracy,
            "f1": f1(
                predictions,
                labels,
            ),
        }
