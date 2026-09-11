import numpy as np
import seaborn as sns
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder

class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.

        TODO: Support numerical, ordinal, categorical, histogram features.
    """
    def __init__(self, feature_config, num_bins=5):
        self.feature_config = feature_config
        self.feature_transforms = {}
        self.is_fit = False

    def clean_numeric(self,v):
        if v == '' or v.strip().lower() == 'nan':
            return 0.0
        return float(v)

    def get_numerical_vectorizer(self, values, verbose=False):
        """
        :return: function to map numerical x to a zero mean, unit std dev normalized score.
        """
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std == 0:
            std = 1.0


        def vectorizer(x):
            x = float(x) if x != '' else mean  # impute missing with the training mean
            return (x - mean) / std

        return vectorizer

    def get_histogram_vectorizer(self, values):
        sns.histplot(data=self, x=values)
#        raise NotImplementedError("Histogram vectorizer not implemented yet")

    def get_categorical_vectorizer(self, values):
        """
        :return: function to map categorical x to one-hot feature vector
        """
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoding = encoder.fit_transform(values.reshape(-1, 1))

    # build a fast lookup: category value -> its one-hot row
        lookup = {}
        for val, row in zip(values, encoding):
            lookup[val] = row

        num_categories = encoding.shape[1]
        unknown_vector = np.zeros(num_categories)  # for unseen categories at val/test time

        def vectorizer(x):
            return lookup.get(x, unknown_vector)

        return vectorizer
#        raise NotImplementedError("Categorical vectorizer not implemented yet")
    
    def get_ordinal_vectorizer(self, values):

        categories = sorted(set(values))
        mapping = {cat: i for i, cat in enumerate(categories)}

        def vectorizer(x):
            return mapping.get(x, -1)  # -1 for unseen categories at val/test time

        return vectorizer

    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self.

            This implementation will depend on how you design your feature config.
        """


#        #raise NotImplementedError("Not implemented yet")
        self.feature_transforms = {}
        for types, names in self.feature_config.items():
            for namee in names:
                raw_values = [row[namee] for row in X]
                if types == "numerical":
                    values = np.array(
                        [self.clean_numeric(v) if v != '' else np.nan for v in raw_values],
                        dtype=float
                )
                    self.feature_transforms[namee] = self.get_numerical_vectorizer(values)
                elif types == "categorical":
                    values = np.array(raw_values)
                    self.feature_transforms[namee] = self.get_categorical_vectorizer(values)
                elif types == "ordinal":
                    values = np.array(raw_values)
                    self.feature_transforms[namee] = self.get_ordinal_vectorizer(values)
        self.is_fit = True


    def transform(self, X):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: list of dicts, each dict is a datapoint
        """

        if not self.is_fit:
            raise Exception("Vectorizer not intialized! You must first call fit with a training set" )

        all_names = [namee for names in self.feature_config.values() for namee in names]
        type_lookup = {namee: t for t, names in self.feature_config.items() for namee in names}

        transformed_data = []
        for row in X:
            row_features = []
            for namee in all_names:
                val = row[namee]
                if type_lookup[namee] == "numerical":
                    val = self.clean_numeric(val)
                tx = self.feature_transforms[namee](val)


                row_features.append(tx)
            transformed_data.append(np.concatenate([np.atleast_1d(t) for t in row_features]))
        return np.array(transformed_data)