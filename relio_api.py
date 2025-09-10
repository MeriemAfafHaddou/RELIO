import random
from enum import Enum

import numpy as np
import ot
import pandas as pd


class DriftType(Enum):
    """
    An enumeration of the four drift types, we also added NONE if the drift
    has no type
    """

    GRADUAL = 1
    SUDDEN = 2
    INCREMENTAL = 3
    RECURRENT = 4
    NONE = 5


class OTMetric(Enum):
    """
    Enumeration of the Optimal Transport metrics that we considered in our
    solution
    """

    WASSERSTEIN1 = 1
    WASSERSTEIN2 = 2
    SINKHORN = 3


class CostFunction(Enum):
    """
    Enumartion of the cost function we want to consider when quantifying the
    transport cost between two distributions
    """

    EUCLIDEAN = "euclidean"
    SEUCLIDEAN = "seuclidean"
    MAHALANOBIS = "mahalanobis"


class Concept:
    """
    Concept represents the actual relation between the variables X and the
    output y
    """

    __start_time = 0  # The window index from which this concept has started
    __length = 0  # number of windows which belong to this concept
    __ref_distr = []  # a representative distribution of this concept

    def __init__(self, start_time, reference_distribution):
        self.__start_time = start_time
        self.__length = 1
        self.__ref_distr = reference_distribution

    def get_length(self):
        return self.__length

    def get_time(self):
        return self.__start_time

    def get_ref_distr(self):
        return self.__ref_distr

    def increment_length(self):
        self.__length += 1

    def set_ref_distr(self, ref_distr):
        self.__ref_distr = ref_distr


class ConceptDrift:
    """
    A class tha represents the concept drift, an object of this class is
    created once a concept drift is created
    """

    __drift_type = DriftType.NONE  # the concept drift type

    def __init__(self, drift_type):
        self.__drift_type = drift_type

    def set_drift_type(self, drift_type):
        self.__drift_type = drift_type

    def get_drift_type(self):
        return self.__drift_type


class RelioApi:
    def __init__(
        self,
        window_size,
        alert_thold,
        detect_thold,
        ot_metric,
        cost_function,
        stability_threshold,
        df: pd.DataFrame,
        time=0,
    ):
        self.__win_size = 0  # Window size in number of samples
        self.__curr_win = []  # Stores the current window
        self.__curr_win_num = 0  # Index of the current window
        self.__curr_concept = Concept(None, None)  # The current concept
        self.__alert_thold = 0.0  # Threshold for small changes
        self.__detect_thold = 0.0  # Threshold for small changes
        self.__ot_metric = OTMetric.WASSERSTEIN2
        self.__cost_fun = CostFunction.SEUCLIDEAN
        self.__stblty_thold = 0  # Stability threshold
        self.__retrain_model = False  # Retrain model on drift
        self.__partial_fit = False  # Adjust model on small change
        self.__ot_distances = []  # Stores OT distance values
        self.__alert = False  # True if an alert is sent
        self.__abrupt = False  # True for abrupt drift
        self.__reappearing_count = 0  # Count reappearing distributions
        self.__concept_drifts = []  # Stores concept drifts
        self.__concepts = []  # Stores detected concepts
        self.__time = time  # 0 = logical time, 1 = physical time

        if self.__time == 0:
            self.__win_size = window_size
            estimated_mean = self.estimate_thold(df)
            self.__alert_thold = estimated_mean * (1 + alert_thold / 100)
            self.__detect_thold = estimated_mean * (1 + detect_thold / 100)
        elif self.__time == 1:
            self.__alert_thold = alert_thold
            self.__detect_thold = detect_thold
        self.__stblty_thold = stability_threshold
        self.__ot_distances = []
        self.__cost_fun = cost_function
        self.__ot_metric = ot_metric

    def set_curr_win(self, current_window):
        self.__curr_win = current_window
        self.__curr_win_num += 1

    def set_curr_concept(self, current_concept):
        self.__curr_concept = current_concept

    def get_action(self):
        if self.__retrain_model:
            return 0

        if self.__partial_fit:
            return 1

        return 2

    def get_distances(self):
        return self.__ot_distances

    def get_concepts(self):
        return self.__concepts

    def get_alert_thold(self):
        return self.__alert_thold

    def get_detect_thold(self):
        return self.__detect_thold

    def get_retrain_model(self):
        return self.__retrain_model

    def get_partial_fit(self):
        return self.__partial_fit

    def reset_retrain_model(self):
        self.__retrain_model = False

    def add_concept(self, new_concept):
        self.__concepts.append(new_concept)

    def reset_partial_fit(self):
        self.__partial_fit = False

    def compare_distributions(self, distr1, dist2):
        """
        a function for comparing two distributions according to the alert
        threshold and the detection threshold,
        using the Optimal Transport metric and the cost function specified
        previously. It returns :
          0 if the two distributions are similar,
          1 if drift can occur (an alert)
          2 if they are different, i.e. drift is detected.

        """
        # Calculate the cost matrix using a certain cost function
        metric = self.__cost_fun.value
        cost_matrix = ot.dist(distr1, dist2, metric=metric)

        # calculate the OT distance
        if self.__ot_metric.value == 2:
            ot_dist = np.sqrt(ot.emd2([], [], np.square(cost_matrix)))
        elif self.__ot_metric.value == 1:
            ot_dist = ot.emd2([], [], cost_matrix)
        else:
            plan = ot.sinkhorn([], [], cost_matrix, 0.1)
            ot_dist = np.sum(plan * cost_matrix)

        # Comparaison
        if ot_dist > self.__detect_thold:
            return 2, ot_dist

        if ot_dist > self.__alert_thold:
            return 1, ot_dist

        return 0, ot_dist

    def is_stable(self, concept_length):
        """
        a function that tells us whether the current concept is stable or not,
        according to the stability threshold we introduced earlier.
        Stability Threshold : the threshold of data stability on a single
        concept. In some types of concept drift,
        we can only detect it once the data has stabilized on a single concept
        (i.e. there are no drifts occurring).
        This threshold is expressed as a number of windows.
        For example, if it is set to 10, then if the length of the current
        concept exceeds this threshold,
        we assume that there is stability on this concept. If another drift
        occurs after stability, we consider it a new drift.

        """
        if concept_length < self.__stblty_thold:
            return False

        return True

    def is_gradual(self, drifts_lengths):
        """
        Verifies if the drift is gradual

        Parameters:
        drifts_lengths (array): an array containing the length of the last
        concept drifts.

        Returns:
        bool : true if the drift is gradual, false if not

        The drift is gradual if the probability of an instance of source
        1 arriving decreases, while the probability of an instance of source
        S2 arriving increases until stabilizations on source 2
        since each drift represents a switching from a source to another,
        and to achieve the stability we must have an impair number of
        switching between the sources
        the pair elements represent the first source, and the impair ones
        represent the second source
        """

        source_1 = drifts_lengths[::2]
        source_2 = drifts_lengths[1::2]
        # if soucre 1 decreases, and source 2 increases than return true
        if all(source_1[i] >= source_1[i + 1]
               for i in range(len(source_1) - 1)):
            if all(source_2[i] <= source_2[i + 1]
                   for i in range(len(source_2) - 1)):
                return True
        return False

    def monitor_drift(self):
        """'
        a function to monitor the data, and react according to the value
        returned by Compare_distr().
        """
        ref_distr = self.__curr_concept.get_ref_distr()
        result, ot_dist = self.compare_distributions(
            ref_distr, self.__curr_win)
        self.__ot_distances.append(ot_dist)
        if result == 0:
            self.__alert = False
            self.__curr_concept.increment_length()
            if self.__time == 0:
                # index=random.randint(0, len(self.__curr_win)-1)
                # ref_distr[index]=np.mean(self.__curr_win, axis=0)
                index_ref = random.randint(0, len(self.__curr_win) - 1)
                index_cur = random.randint(0, len(self.__curr_win) - 1)
                ref_distr[index_ref] = self.__curr_win[index_cur]
                self.__curr_concept.set_ref_distr(ref_distr)
        elif result == 1:
            self.__alert = True
            self.__abrupt = False
            self.__partial_fit = True
            self.__curr_concept.increment_length()
        elif result == 2:
            self.__retrain_model = True
            concept = Concept(self.__curr_win_num, self.__curr_win)
            self.__curr_concept = concept
            self.__concepts.append(concept)
            concept_drift = ConceptDrift(self.__curr_win_num)
            self.__concept_drifts.append(concept_drift)
            if len(self.__concepts) > 2:
                diff, _ = self.compare_distributions(
                    self.__curr_win, self.__concepts[-3].get_ref_distr()
                )
                if diff == 0:
                    self.__reappearing_count += 1

                else:
                    self.__reappearing_count = 0

    def estimate_thold(self, dataset: pd.DataFrame):
        """
        a function to estimate the alert and detection thresholds, using the
        Optimal Transport metric and the cost function specified previously.
        """
        dist1 = dataset.sample(n=self.__win_size, random_state=42)
        dist2 = dataset.sample(n=self.__win_size, random_state=2024)
        distance = self.compare_distributions(
            np.array(dist1), np.array(dist2))[1]
        return round(distance, 2)

    def identify_type(self):
        # INCREMENTAL DRIFT
        if self.__alert and self.__retrain_model:
            if len(self.__ot_distances) > 2:
                if (
                    self.__ot_distances[-3] < self.__ot_distances[-2]
                    and self.__ot_distances[-2] < self.__detect_thold
                ):
                    self.__alert = False
                    self.__concept_drifts[-1].set_drift_type(
                        DriftType.INCREMENTAL)
                    return DriftType.INCREMENTAL
            elif self.__ot_distances[-1] > self.__ot_distances[-2]:
                self.__alert = False
                self.__concept_drifts[-1].set_drift_type(DriftType.INCREMENTAL)
                return DriftType.INCREMENTAL

        # SUDDEN DRIFT
        elif not self.__alert and self.__retrain_model:
            self.__abrupt = True
        if (
            self.__abrupt
            and self.is_stable(self.__curr_concept.get_length())
            and self.__reappearing_count == 0
        ):
            self.__concept_drifts[-1].set_drift_type(DriftType.SUDDEN)
            self.__abrupt = False
            return DriftType.SUDDEN

        # GRADUAL DRIFT
        if self.__reappearing_count > 2 and self.is_stable(
            self.__curr_concept.get_length()
        ):
            drifts_length = []

            for concept in self.__concepts[-(self.__reappearing_count + 2):]:
                drifts_length.append(concept.get_length())

            if self.is_gradual(drifts_length):
                self.__abrupt = False
                self.__concept_drifts[-1].set_drift_type(DriftType.GRADUAL)
                self.__reappearing_count = 0
                return DriftType.GRADUAL

        # RECURRENT DRIFT
        elif self.__reappearing_count == 2:
            drifts_length = []

            for concept in self.__concepts[-(self.__reappearing_count + 2):]:
                drifts_length.append(concept.get_length())

            if (
                not self.is_gradual(drifts_length)
                and self.__concepts[-1].get_length() != 1
            ):
                self.__abrupt = False
                self.__concept_drifts[-1].set_drift_type(DriftType.RECURRENT)
                self.__reappearing_count = 0
                return DriftType.RECURRENT

        return DriftType.NONE
