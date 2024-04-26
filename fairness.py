from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.postprocessing.reject_option_classification\
        import RejectOptionClassification

# Take the weights returned from this function and multiply them by the agent features
def run_reweighing(num_agents, num_pos, num_priv, num_pos_priv, num_neg, num_nonpriv, num_neg_priv, num_pos_nonpriv, num_neg_nonpriv):
    pos_priv_weight = (num_pos * num_priv) / (num_agents * num_pos_priv)
    neg_priv_weight = (num_neg * num_priv) / (num_agents * num_neg_priv)
    pos_nonpriv_weight = (num_pos * num_nonpriv) / (num_agents * num_pos_nonpriv)
    neg_nonpriv_weight = (num_neg * num_nonpriv) / (num_agents * num_neg_nonpriv)
    return (pos_priv_weight, neg_priv_weight, pos_nonpriv_weight, neg_nonpriv_weight)


# 
def run_ORC():
    ROC = RejectOptionClassification(unprivileged_groups=[{'group': 0}], 
                                    privileged_groups=[{'group': 1}], 
                                    low_class_thresh=0.01, high_class_thresh=0.99,
                                    num_class_thresh=100, num_ROC_margin=50,
                                    metric_name="Equal opportunity difference",
                                    metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)