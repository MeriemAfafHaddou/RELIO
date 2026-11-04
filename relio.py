import itertools
import random
from enum import Enum, auto

import numpy as np
import ot
import pandas as pd


class DriftType(Enum):
    """
    An enumeration of the four drift types, we also added NONE if the drift
    has no type
    """

    GRADUAL = auto()
    SUDDEN = auto()
    INCREMENTAL = auto()
    RECURRENT = auto()
    NONE = auto()


class OTMetric(Enum):
    """
    Enumeration of the Optimal Transport metrics that we considered in our
    solution
    """

    WASSERSTEIN1 = auto()
    WASSERSTEIN2 = auto()
    SINKHORN = auto()


class CostFunction(Enum):
    """
    Enumartion of the cost function we want to consider when quantifying the
    transport cost between two distributions
    """

    EUCLIDEAN = "euclidean"
    SEUCLIDEAN = "seuclidean"
    MAHALANOBIS = "mahalanobis"


class TimeMode(Enum):
    """Define how time is represented and tracked in the data stream."""

    LOGICAL = "logical"
    PHYSICAL = "physical"


class DriftStatus(Enum):
    """Represent the outcome of comparing two feature distributions."""

    NO_DRIFT = auto()
    WARNING = auto()
    DRIFT = auto()


class Concept:
    """
    Represent a concept in the data stream by tracking its start time,
    duration, and reference feature distribution.
    """

    def __init__(self, start_time, reference_distribution):
        self.__start_time = start_time
        self.__length = 1
        self.__ref_distr = reference_distribution

    def get_length(self):
        return self.__length

    def get_start_time(self):
        return self.__start_time

    def get_ref_distr(self):
        return self.__ref_distr

    def increment_length(self):
        self.__length += 1

    def set_ref_distr(self, ref_distr):
        self.__ref_distr = ref_distr


class ConceptDrift:
    """
    Represent a detected concept drift by recording its detection window
    and its identified drift type.
    """

    def __init__(self, detection_window):
        self.__detection_window = detection_window
        self.__drift_type = DriftType.NONE

    def get_detection_window(self):
        return self.__detection_window

    def get_drift_type(self):
        return self.__drift_type

    def set_drift_type(self, drift_type):
        self.__drift_type = drift_type


class Relio:
    def __init__(
        self,
        window_size,
        warning_threshold,
        detection_threshold,
        ot_metric=OTMetric.WASSERSTEIN1,
        cost_function=CostFunction.EUCLIDEAN,
        stability_threshold=3,
        sudden_drift_threshold=1,
        df: pd.DataFrame = None,
        time_mode=TimeMode.LOGICAL,
    ):
        # Configuration
        self.__time_mode = time_mode
        self.__win_size = window_size if time_mode == TimeMode.LOGICAL else None

        self.__ot_metric = ot_metric
        self.__cost_fun = cost_function

        self.__sudden_drift_threshold = sudden_drift_threshold
        self.__stability_threshold = stability_threshold

        # Threshold estimation
        if self.__time_mode == TimeMode.LOGICAL:
            baseline_distance = self.estimate_thold(df)
            self.__warning_thold = baseline_distance * (1 + warning_threshold / 100)
            self.__detect_thold = baseline_distance * (1 + detection_threshold / 100)
        else:
            self.__warning_thold = warning_threshold
            self.__detect_thold = detection_threshold

        # Window and concept state
        self.__curr_win = []
        self.__curr_win_num = 0
        self.__curr_concept = Concept(None, None)

        # Drift detection state
        self.__warning = False
        self.__model_adaptation_required = False
        self.__model_adjustment_required = False

        # OT history
        self.__ot_distance_history = []

        # Concept history
        self.__concepts = []
        self.__concept_drifts = []

        self.__transition_start = None
        self.__transition_end = None
        self.__drift_onset = None

    # ——— SETTERS ——————————————————————————————
    def set_curr_win(self, current_window):
        self.__curr_win = current_window
        self.__curr_win_num += 1

    def set_curr_concept(self, current_concept):
        self.__curr_concept = current_concept

    # ——— GETTERS ——————————————————————————————
    def get_action(self):
        if self.__model_adaptation_required:
            return DriftStatus.DRIFT

        if self.__model_adjustment_required:
            return DriftStatus.WARNING

        return DriftStatus.NO_DRIFT

    def get_distance_history(self):
        return self.__ot_distance_history

    def get_warning_thold(self):
        return self.__warning_thold

    def get_detect_thold(self):
        return self.__detect_thold

    def get_adapt_model(self):
        return self.__model_adaptation_required

    def get_adjust_model(self):
        return self.__model_adjustment_required

    # ——— RE-SETTERS ——————————————————————————————
    def reset_adapt_model(self):
        self.__model_adaptation_required = False

    def reset_adjust_model(self):
        self.__model_adjustment_required = False

    # ——— CONCEPT OPERATIONS ——————————————————————
    def add_concept(self, new_concept):
        self.__concepts.append(new_concept)

    # ——— OT DISTANCE —————————————————————————————
    def calculate_ot_distance(self, distr1, distr2):
        """
        Calculate the Optimal Transport distance between two distributions.

        Args:
            distr1: First empirical distribution.
            distr2: Second empirical distribution.

        Returns:
            float: Calculated OT distance.
        """
        metric = self.__cost_fun.value
        cost_matrix = ot.dist(distr1, distr2, metric=metric)

        if self.__ot_metric == OTMetric.WASSERSTEIN2:
            return np.sqrt(ot.emd2([], [], np.square(cost_matrix)))

        elif self.__ot_metric == OTMetric.WASSERSTEIN1:
            return ot.emd2([], [], cost_matrix)

        elif self.__ot_metric == OTMetric.SINKHORN:
            plan = ot.sinkhorn([], [], cost_matrix, 0.1)
            return np.sum(plan * cost_matrix)

        raise ValueError(f"Unsupported OT metric: {self.__ot_metric}")

    # ——— THRESHOLD ESTIMATION ————————————————————
    def estimate_thold(self, dataset: pd.DataFrame):
        """
        Estimate a baseline OT distance using two random samples
        from the provided dataset.
        """
        dist1 = dataset.sample(n=self.__win_size, random_state=42)
        dist2 = dataset.sample(n=self.__win_size, random_state=2024)
        distance = self.calculate_ot_distance(dist1.to_numpy(), dist2.to_numpy())
        return round(distance, 2)

    # ——— COMPARISON ——————————————————————————————
    def compare_distributions(self, distr1, distr2):
        """
        Compare two distributions using the configured OT metric
        and classify their similarity based on the configured thresholds.

        Returns:
            tuple: DriftStatus and the calculated OT distance.
        """
        # Calculate the cost matrix using a certain cost function
        ot_dist = self.calculate_ot_distance(distr1, distr2)

        if ot_dist > self.__detect_thold:
            result = DriftStatus.DRIFT
        elif ot_dist > self.__warning_thold:
            result = DriftStatus.WARNING
        else:
            result = DriftStatus.NO_DRIFT

        return result, ot_dist

    # ——— DRIFT DETECTION —————————————————————————
    def detect_drift(self):
        """
        Compare the current window with the reference distribution and
        update the concept and transition states accordingly.

        Returns the identified drift type when classification is complete,
        or None otherwise.
        """
        ref_distr = self.__curr_concept.get_ref_distr()

        result, ot_dist = self.compare_distributions(ref_distr, self.__curr_win)

        self.__ot_distance_history.append(ot_dist)

        if result == DriftStatus.NO_DRIFT:
            self.__curr_concept.increment_length()

            if self.__time_mode == TimeMode.LOGICAL:
                index_ref = random.randint(0, len(self.__curr_win) - 1)
                index_cur = random.randint(0, len(self.__curr_win) - 1)

                ref_distr[index_ref] = self.__curr_win[index_cur]
                self.__curr_concept.set_ref_distr(ref_distr)

        elif result == DriftStatus.WARNING:
            if self.__drift_onset is None:
                self.__drift_onset = self.__curr_win_num

            self.__model_adjustment_required = True
            self.__curr_concept.increment_length()

        elif result == DriftStatus.DRIFT:
            if self.__drift_onset is None:
                self.__drift_onset = self.__curr_win_num

            self.__model_adaptation_required = True
            self.__transition_start = self.__curr_win_num

            concept = Concept(self.__curr_win_num, self.__curr_win)
            self.__curr_concept = concept
            self.__concepts.append(concept)

            concept_drift = ConceptDrift(self.__curr_win_num)
            self.__concept_drifts.append(concept_drift)

        # Identify the drift type once the new concept reaches stability.
        if self.__transition_start is not None:
            current_length = self.__curr_concept.get_length()
            stable = self.is_stable(current_length)

            if stable:
                self.__transition_end = self.__curr_win_num

                drift_type = self.identify_drift_type()

                if drift_type != DriftType.NONE:
                    self.__transition_start = None
                    self.__transition_end = None

                    return drift_type
        return None

    # ——— TYPE IDENTIFICATION ————————————————————————
    def is_stable(self, concept_length):
        """
        Check whether the current concept has remained stable for enough windows
        to trigger drift type identification.
        """
        return not concept_length < self.__stability_threshold

    def is_drift_sudden(self):
        """
        Determine whether the transition completes within the maximum
        duration configured for sudden drift.
        """
        if self.__drift_onset is None or self.__transition_start is None:
            return False

        transition_duration = self.__transition_start - self.__drift_onset
        return transition_duration < self.__sudden_drift_threshold

    def is_drift_incremental(self):
        """
        Determine whether a drift transition is incremental.

        An incremental transition lasts longer than the sudden drift
        threshold and exhibits a non-decreasing OT distance from the
        window preceding its onset until drift detection.
        """
        if self.__drift_onset is None or self.__transition_start is None:
            return False

        if self.__drift_onset == 0:
            return False

        transition_distances = self.__ot_distance_history[
            self.__drift_onset - 2 : self.__transition_start
        ]
        if len(transition_distances) < 3:
            return False

        return all(
            current <= following
            for current, following in itertools.pairwise(transition_distances)
        )

    def is_drift_recurrent(self):
        """
        Check whether the current concept matches a previously observed concept,
        with at least one different concept occurring in between.
        """
        if len(self.__concepts) < 3:
            return False

        current_concept = self.__concepts[-1]

        for historical_concept in self.__concepts[:-2]:
            result, _ = self.compare_distributions(
                current_concept.get_ref_distr(), historical_concept.get_ref_distr()
            )

            if result == DriftStatus.NO_DRIFT:
                return True

        return False

    def is_drift_gradual(self):
        """
        Determine whether alternating concept durations indicate gradual drift.
        """
        drifts_lengths = [
            concept.get_length()
            for concept in self.__concepts
            if self.__transition_start
            <= concept.get_start_time()
            <= self.__transition_end
        ]

        if len(drifts_lengths) < 5:
            return False

        source_1 = drifts_lengths[::2]
        source_2 = drifts_lengths[1::2]

        source_1_decreases = all(
            a >= b for a, b in itertools.pairwise(source_1)
        ) and any(a > b for a, b in itertools.pairwise(source_1))

        source_2_increases = all(
            a <= b for a, b in itertools.pairwise(source_2)
        ) and any(a < b for a, b in itertools.pairwise(source_2))

        return source_1_decreases and source_2_increases

    def identify_drift_type(self):
        """Identify the drift type once the new concept has stabilized."""

        if self.__transition_start is None or self.__transition_end is None:
            return DriftType.NONE

        if not self.is_stable(self.__curr_concept.get_length()):
            return DriftType.NONE

        if self.is_drift_sudden():
            drift_type = DriftType.SUDDEN

        elif self.is_drift_recurrent():
            drift_type = DriftType.RECURRENT

        elif self.is_drift_incremental():
            drift_type = DriftType.INCREMENTAL

        elif self.is_drift_gradual():
            drift_type = DriftType.GRADUAL

        else:
            return DriftType.NONE

        self.__concept_drifts[-1].set_drift_type(drift_type)
        return drift_type
