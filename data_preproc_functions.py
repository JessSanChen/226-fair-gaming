from aif360.datasets import GermanDataset
import pandas as pd
import numpy as np

"""
ALTERNATIVE GERMAN DATASET.

Drops all features except for age, credit_history=Other, savings=<500

"""


def load_preproc_data_german(protected_attributes=None):
    """
    Load and pre-process german credit dataset.
    Args:
        protected_attributes(list or None): If None use all possible protected
            attributes, else subset the protected attributes to the list.

    Returns:
        GermanDataset: An instance of GermanDataset with required pre-processing.

    """
    def custom_preprocessing(df):
        """ Custom pre-processing for German Credit Data
        """

        def group_credit_hist(x):
            if x in ['A30', 'A31', 'A32']:
                return 'None/Paid'
            elif x == 'A33':
                return 'Delay'
            elif x == 'A34':
                return 'Other'
            else:
                return 'NA'

        def group_employ(x):
            if x == 'A71':
                return 'Unemployed'
            elif x in ['A72', 'A73']:
                return '1-4 years'
            elif x in ['A74', 'A75']:
                return '4+ years'
            else:
                return 'NA'

        def group_savings(x):
            if x in ['A61', 'A62']:
                return '<500'
            elif x in ['A63', 'A64']:
                return '500+'
            elif x == 'A65':
                return 'Unknown/None'
            else:
                return 'NA'

        def group_status(x):
            if x in ['A11', 'A12']:
                return '<200'
            elif x in ['A13']:
                return '200+'
            elif x == 'A14':
                return 'None'
            else:
                return 'NA'

        status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
                    'A92': 0.0, 'A95': 0.0}
        df['sex'] = df['personal_status'].replace(status_map)


        # group credit history, savings, and employment
        print(df['credit_history'])
        df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
        print(df['credit_history'])
        df['savings'] = df['savings'].apply(lambda x: group_savings(x))
        df['employment'] = df['employment'].apply(lambda x: group_employ(x))
        df['age'] = (df['age'] >= 26).astype(float)
        df['status'] = df['status'].apply(lambda x: group_status(x))

        return df

    # Feature partitions
    XD_features = ['credit_history', 'savings', 'employment', 'sex', 'age']
    D_features = ['sex', 'age'] if protected_attributes is None else protected_attributes
    Y_features = ['credit']
    X_features = list(set(XD_features)-set(D_features))
    print(X_features)
    categorical_features = ['credit_history', 'savings', 'employment']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "age": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "age": {1.0: 'Old', 0.0: 'Young'}}

    return GermanDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={ 'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
                   'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)
