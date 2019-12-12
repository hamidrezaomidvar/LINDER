import numpy as np
from eolearn.core import (
    EOTask,
    EOPatch,
    LinearWorkflow,
    FeatureType,
    OverwritePermission,
    LoadFromDisk,
    SaveToDisk,
    EOExecutor,
)
from enum import Enum


class SentinelHubValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """

    def __call__(self, eopatch):
        return np.logical_and(
            eopatch.mask["IS_DATA"].astype(np.bool),
            np.logical_not(eopatch.mask["CLM"].astype(np.bool)),
        )


class CountValid(EOTask):
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """

    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch):
        eopatch.add_feature(
            FeatureType.MASK_TIMELESS,
            self.name,
            np.count_nonzero(eopatch.mask[self.what], axis=0),
        )

        return eopatch


class NormalizedDifferenceIndex(EOTask):
    """
    The tasks calculates user defined Normalised Difference Index (NDI) between two bands A and B as:
    NDI = (A-B)/(A+B).
    """

    def __init__(self, feature_name, band_a, band_b):
        self.feature_name = feature_name
        self.band_a_fetaure_name = band_a.split("/")[0]
        self.band_b_fetaure_name = band_b.split("/")[0]
        self.band_a_fetaure_idx = int(band_a.split("/")[-1])
        self.band_b_fetaure_idx = int(band_b.split("/")[-1])

    def execute(self, eopatch):
        band_a = eopatch.data[self.band_a_fetaure_name][..., self.band_a_fetaure_idx]
        band_b = eopatch.data[self.band_b_fetaure_name][..., self.band_b_fetaure_idx]

        ndi = (band_a - band_b) / (band_a + band_b)

        eopatch.add_feature(FeatureType.DATA, self.feature_name, ndi[..., np.newaxis])

        return eopatch


class EuclideanNorm(EOTask):
    """
    The tasks calculates Euclidian Norm of all bands within an array:
    norm = sqrt(sum_i Bi**2),
    where Bi are the individual bands within user-specified feature array.
    """

    def __init__(self, feature_name, in_feature_name):
        self.feature_name = feature_name
        self.in_feature_name = in_feature_name

    def execute(self, eopatch):
        arr = eopatch.data[self.in_feature_name]
        norm = np.sqrt(np.sum(arr ** 2, axis=-1))

        eopatch.add_feature(FeatureType.DATA, self.feature_name, norm[..., np.newaxis])
        return eopatch


class LULC(Enum):
    OTHER = (0, "Paved+Built+other", "red")
    VEG = (1, "Veg", "green")
    Water = (2, "Water", "blue")

    def __init__(self, val1, val2, val3):
        self.id = val1
        self.class_name = val2
        self.color = val3


class ConcatenateData(EOTask):

    """ Task to concatenate data arrays along the last dimension
    """

    def __init__(self, feature_name, feature_names_to_concatenate):
        self.feature_name = feature_name
        self.feature_names_to_concatenate = feature_names_to_concatenate

    def execute(self, eopatch):
        arrays = [eopatch.data[name] for name in self.feature_names_to_concatenate]
        eopatch.add_feature(
            FeatureType.DATA, self.feature_name, np.concatenate(arrays, axis=-1)
        )
        return eopatch


class ValidDataFractionPredicate:
    """ Predicate that defines if a frame from EOPatch's time-series is valid or not. Frame is valid, if the
    valid data fraction is above the specified threshold.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
        return coverage > self.threshold


class PredictPatch(EOTask):
    """
            Task to make model predictions on a patch. Provide the model and the feature,
            and the output names of labels and scores (optional)
            """

    def __init__(
        self,
        model,
        features_feature,
        predicted_labels_name,
        pic_n,
        predicted_scores_name=None,
    ):
        self.model = model
        self.features_feature = features_feature
        self.predicted_labels_name = predicted_labels_name
        self.predicted_scores_name = predicted_scores_name
        self.pic_n = pic_n

    def execute(self, eopatch):
        ftrs = eopatch[self.features_feature[0]][self.features_feature[1]][
            self.pic_n, :, :, :
        ]
        # ftrs = np.mean(eopatch[self.features_feature[0]][self.features_feature[1]][:,:,:,:],axis=0)
        # ftrs = eopatch[self.features_feature[0]][self.features_feature[1]][pic_n,:,:,7]

        # t, w, h, f = ftrs.shape
        w, h, f = ftrs.shape
        # w, h = ftrs.shape

        # ftrs = np.moveaxis(ftrs, 0, 2).reshape(w * h, 1 * f)
        ftrs = ftrs.reshape(w * h, 1 * f)
        # ftrs = ftrs.reshape(w * h, 1 * 1)

        plabels = self.model.predict(ftrs)
        plabels = plabels.reshape(w, h)
        plabels = plabels[..., np.newaxis]
        eopatch.add_feature(
            FeatureType.MASK_TIMELESS, self.predicted_labels_name, plabels
        )

        if self.predicted_scores_name:
            pscores = self.model.predict_proba(ftrs)
            _, d = pscores.shape
            pscores = pscores.reshape(w, h, d)
            eopatch.add_feature(
                FeatureType.DATA_TIMELESS, self.predicted_scores_name, pscores
            )

        return eopatch
