"""
    Invariant Causal Set Covering Machine -- A invariance-oriented version of the Set Covering Machine in Python
    Copyright (C) 2023 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from __future__ import print_function, division, absolute_import, unicode_literals

from six import iteritems

import math
import logging
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)
from warnings import warn

import warnings
from itertools import chain, combinations  # for powerset
from gsq import ci_tests  # conditional independence tests
from pyscm import SetCoveringMachineClassifier


def _class_to_string(instance):
    """
    Returns a string representation of the public attributes of a class.

    Parameters:
    -----------
    instance: object
        An instance of any class.

    Returns:
    --------
    string_rep: string
        A string representation of the class and its public attributes.

    Notes:
    -----
    Private attributes must be marked with a leading underscore.
    """
    return (
        instance.__class__.__name__
        + "("
        + ",".join(
            [
                str(k) + "=" + str(v)
                for k, v in iteritems(instance.__dict__)
                if str(k[0]) != "_"
            ]
        )
        + ")"
    )


class BaseModel(object):
    def __init__(self):
        self.rules = []
        super(BaseModel, self).__init__()

    def add(self, rule):
        self.rules.append(rule)

    def predict(self, X):
        raise NotImplementedError()

    def predict_proba(self, X):
        raise NotImplementedError()

    def remove(self, index):
        del self.rules[index]

    @property
    def example_dependencies(self):
        return [d for ba in self.rules for d in ba.example_dependencies]

    @property
    def type(self):
        raise NotImplementedError()

    def _to_string(self, separator=" "):
        return separator.join([str(a) for a in self.rules])

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __iter__(self):
        for ba in self.rules:
            yield ba

    def __len__(self):
        return len(self.rules)

    def __str__(self):
        return self._to_string()


class ConjunctionModel(BaseModel):
    def predict(self, X):
        predictions = np.ones(X.shape[0], bool)
        for a in self.rules:
            predictions = np.logical_and(predictions, a.classify(X))
        return predictions.astype(np.uint8)

    @property
    def type(self):
        return "conjunction"

    def __str__(self):
        return self._to_string(separator=" and ")


class BaseRule(object):
    """
    A rule mixin class

    """

    def __init__(self):
        super(BaseRule, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the rule.

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features), dtype=np.float
            The feature vectors of examples to classify.

        Returns:
        --------
        classifications: array-like, shape=(n_examples,), dtype=bool
            The outcome of the rule (True or False) for each example.

        """
        raise NotImplementedError()

    def inverse(self):
        """
        Creates a rule that is the opposite of the current rule (self).

        Returns:
        --------
        inverse: BaseRule
            A rule that is the inverse of self.

        """
        raise NotImplementedError()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return _class_to_string(self)


class DecisionStump(BaseRule):
    """
    A decision stump is a rule that applies a threshold to the value of some feature

    Parameters:
    -----------
    feature_idx: uint
        The index of the feature
    threshold: float
        The threshold at which the outcome of the rule changes
    kind: str, default="greater"
        The case in which the rule returns 1, either "greater" or "less_equal".

    """

    def __init__(self, feature_idx, threshold, kind="greater"):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.kind = kind
        super(DecisionStump, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the decision stump.

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features), dtype=np.float
            The feature vectors of examples to classify.

        Returns:
        --------
        classifications: array-like, shape=(n_examples,), dtype=bool
            The outcome of the rule (True or False) for each example.

        """
        if self.kind == "greater":
            c = X[:, self.feature_idx] > self.threshold
        else:
            c = X[:, self.feature_idx] <= self.threshold
        return c

    def inverse(self):
        """
        Creates a rule that is the opposite of the current rule (self).

        Returns:
        --------
        inverse: BaseRule
            A rule that is the inverse of self.

        """
        return DecisionStump(
            feature_idx=self.feature_idx,
            threshold=self.threshold,
            kind="greater" if self.kind == "less_equal" else "less_equal",
        )

    def __str__(self):
        return "X[{0:d}] {1!s} {2:.3f}".format(
            self.feature_idx, ">" if self.kind == "greater" else "<=", self.threshold
        )


class InvariantCausalPredictionSetCoveringMachine(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        p=1.0,
        model_type="conjunction",
        max_rules=10,
        threshold=0.05,
        random_state=None,
    ):
        self.p = p
        self.model_type = model_type
        self.max_rules = max_rules
        self.threshold = threshold
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            "p": self.p,
            "model_type": self.model_type,
            "max_rules": self.max_rules,
            "threshold": self.threshold,
            "random_state": self.random_state,
        }

    def set_params(self, **parameters):
        for parameter, value in iteritems(parameters):
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        """
        Find the set of causal parents of the target variable.
        Fit a SCM model on this set.

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The first columns contains the environment id of the example.
            The features of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.

        Returns:
        --------
        self: object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        

        if self.model_type == "conjunction":
            self._add_attribute_to_model = self._append_conjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_conjunction
        elif self.model_type == "disjunction":
            self._add_attribute_to_model = self._append_disjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_disjunction
        else:
            raise ValueError("Unsupported model type.")

        # Initialize callbacks
        if iteration_callback is None:
            iteration_callback = lambda x: None

        # Parse additional fit parameters
        logging.debug("Parsing additional fit parameters")
        utility_function_additional_args = {}
        if fit_params is not None:
            for key, value in iteritems(fit_params):
                if key[:9] == "utility__":
                    utility_function_additional_args[key[9:]] = value

        # Validate the input data
        logging.debug("Validating the input data")
        # X, y = check_X_y(X, y)
        # X = np.asarray(X, dtype=np.double)
        self.classes_, y, total_n_ex_by_class = np.unique(
            y, return_inverse=True, return_counts=True
        )
        if len(self.classes_) != 2:
            raise ValueError("y must contain two unique classes.")
        logging.debug(
            "The data contains {0:d} examples. Negative class is {1!s} (n: {2:d}) and positive class is {3!s} (n: {4:d}).".format(
                len(y),
                self.classes_[0],
                total_n_ex_by_class[0],
                self.classes_[1],
                total_n_ex_by_class[1],
            )
        )

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise ValueError("unexpected type for X:", type(X))

        if isinstance(y, list):
            y = np.array(y)
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise ValueError("unexpected type for y:", type(X))

        # Create an empty model
        logging.debug("Initializing empty model")
        self.model_ = ConjunctionModel()
        logging.debug("Training start")
        ones = np.ones(len(y))
        remaining_N = ones - y  # remaining negative examples
        remaining_P = y

        # first column of X: environment
        env_of_examples = X[:, 0]

        # Extract features only
        X_original_with_env = (
            X.copy()
        )  # useful for prediction for feature importance computation
        X = X[:, 1:]
        # print('Extract features only, X.shape = ', X.shape)

        variables_idx = list(range(X_original_with_env.shape[1]))[
            1:
        ]  # because feature 0 in X_original_with_env is the environment
        sets = list(
            chain.from_iterable(
                combinations(variables_idx, r) for r in range(X.shape[1])
            )
        )
        sets_that_creates_indep = []
        conditional_indep_calculation_df = pd.DataFrame(
            X_original_with_env, columns=["e"] + variables_idx
        )
        conditional_indep_calculation_df["y"] = y
        y_position = conditional_indep_calculation_df.columns.get_loc("y")
        e_position = conditional_indep_calculation_df.columns.get_loc("e")
        # print('e_position', e_position)
        # print('y_position', y_position)
        # print(list(conditional_indep_calculation_df))
        for s in sets:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                p_value = ci_tests.ci_test_bin(
                    conditional_indep_calculation_df.values,
                    e_position,
                    y_position,
                    list(s),
                )
            if p_value > self.threshold:
                sets_that_creates_indep.append(s)
        # print('sets_that_creates_indep', sets_that_creates_indep)
        # intersection of sets:
        if len(sets_that_creates_indep) == 0:
            intersection_sets_that_creates_indep = []
        else:
            intersection_sets_that_creates_indep = list(
                set.intersection(*map(set, sets_that_creates_indep))
            )
        # print('intersection_sets_that_creates_indep', intersection_sets_that_creates_indep)

        # Calculate classic SCM training set made of noly parent variables
        X_restricted_to_parents = np.zeros(X_original_with_env.shape)
        for i in intersection_sets_that_creates_indep:
            X_restricted_to_parents[:, i] = X_original_with_env[:, i]
        model = SetCoveringMachineClassifier(
            p=self.p,
            model_type=self.model_type,
            max_rules=self.max_rules,
            random_state=self.random_state,
        )
        model.fit(X_restricted_to_parents, y)
        self.model_ = model
        # print('model_.model_', model.model_)

        logging.debug("Training completed")

        logging.debug("Calculating rule importances")
        if len(intersection_sets_that_creates_indep) == 0:
            self.feature_importances_ = [0] * X_original_with_env.shape[1]
        else:
            self.feature_importances_ = [
                int(i in intersection_sets_that_creates_indep)
                for i in range(X_original_with_env.shape[1])
            ]
        logging.debug("Done.")

        return self

    def predict(self, X):
        """
        Predict class

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The feature of the input examples.

        Returns:
        --------
        predictions: numpy_array, shape=[n_examples]
            The predicted class for each example.

        """
        check_is_fitted(self, ["model_", "feature_importances_", "classes_"])
        X = check_array(X)
        return self.classes_[self.model_.predict(X)]

    def predict_proba(self, X):
        """
        Predict class probabilities

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features)
            The feature of the input examples.

        Returns:
        --------
        p : array of shape = [n_examples, 2]
            The class probabilities for each example. Classes are ordered by lexicographic order.

        """
        warn(
            "SetCoveringMachines do not support probabilistic predictions. The returned values will be zero or one.",
            RuntimeWarning,
        )
        check_is_fitted(self, ["model_", "feature_importances_", "classes_"])
        X = check_array(X)
        pos_proba = self.classes_[self.model_.predict(X)]
        neg_proba = 1.0 - pos_proba
        return np.hstack((neg_proba.reshape(-1, 1), pos_proba.reshape(-1, 1)))

    def score(self, X, y):
        """
        Predict classes of examples and measure accuracy

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features)
            The feature of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.

        Returns:
        --------
        accuracy: float
            The proportion of correctly classified examples.

        """
        check_is_fitted(self, ["model_", "feature_importances_", "classes_"])
        X, y = check_X_y(X, y)
        return accuracy_score(y_true=y, y_pred=self.predict(X))

    def _append_conjunction_model(self, new_rule):
        self.model_.add(new_rule)
        logging.debug("Attribute added to the model: " + str(new_rule))
        return new_rule

    def _append_disjunction_model(self, new_rule):
        new_rule = new_rule.inverse()
        self.model_.add(new_rule)
        logging.debug("Attribute added to the model: " + str(new_rule))
        return new_rule

    def _get_example_idx_by_class_conjunction(self, y):
        positive_example_idx = np.where(y == 1)[0]
        negative_example_idx = np.where(y == 0)[0]
        return positive_example_idx, negative_example_idx

    def _get_example_idx_by_class_disjunction(self, y):
        positive_example_idx = np.where(y == 0)[0]
        negative_example_idx = np.where(y == 1)[0]
        return positive_example_idx, negative_example_idx

    def __str__(self):
        return _class_to_string(self)
