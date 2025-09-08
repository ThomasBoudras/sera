import numpy as np


class get_metrics_global:
    def __init__(self, metrics_set) :
        self.metrics_set = metrics_set

    def __call__(self, metrics_local):
        metrics = {}
        for metric_name, metric_computer in self.metrics_set.items():
            metrics[metric_name] =  metric_computer.compute_metrics(metrics_local)
        return metrics

class precision_global_computer:
    def __init__(self, name_true_positive, name_false_positive):
        self.name_true_positive = name_true_positive
        self.name_false_positive = name_false_positive
        
    def compute_metrics(self, metrics_local) :
        if (self.name_true_positive not in metrics_local) or (self.name_false_positive not in metrics_local) :
            Exception(f"You must first compute {self.name_true_positive} and {self.name_false_positive}")

        true_positive = np.sum(metrics_local[self.name_true_positive])
        false_positive = np.sum(metrics_local[self.name_false_positive])

        # Calculate Precision
        if true_positive + false_positive == 0:
            return 0
        precision = true_positive / (true_positive + false_positive)
        return precision
    

class  recall_global_computer :
    def __init__(self, name_true_positive, name_false_negative):
        self.name_true_positive = name_true_positive
        self.name_false_negative = name_false_negative
        
    def compute_metrics(self, metrics_local) :
        if (self.name_true_positive not in metrics_local) or (self.name_false_negative not in metrics_local) :
            Exception(f"You must first compute {self.name_true_positive} and {self.name_false_negative}")

        true_positive = np.sum(metrics_local[self.name_true_positive])
        false_negative = np.sum(metrics_local[self.name_false_negative])
        
        # Calculate Recall
        if true_positive + false_negative == 0:
            return 0
        recall = true_positive / (true_positive + false_negative)
        return recall
    

class f1_score_global_computer:
    def __init__(self, name_true_positive, name_false_negative, name_false_positive):
        self.name_true_positive = name_true_positive
        self.name_false_negative = name_false_negative
        self.name_false_positive = name_false_positive
        
    def compute_metrics(self, metrics_local) :
        if (self.name_true_positive not in metrics_local) or (self.name_false_negative not in metrics_local) or (self.name_false_positive not in metrics_local) :
            Exception(f"You must first compute {self.name_true_positive}, {self.name_false_positive} and {self.name_false_negative}")

        true_positive = np.sum(metrics_local[self.name_true_positive])
        false_negative = np.sum(metrics_local[self.name_false_negative])
        false_positive = np.sum(metrics_local[self.name_false_positive])
                
        # Calculate Precision and Recall
        if true_positive + false_positive == 0:
            precision = 0
        else:
            precision = true_positive / (true_positive + false_positive)
        
        if true_positive + false_negative == 0:
            recall = 0
        else:
            recall = true_positive / (true_positive + false_negative)
        
        # Calculate F1 Score
        if precision + recall == 0 :
            return 0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


class mean_global_computeur :
    def __init__(self, name_metric, root=False):
        self.name_metric =name_metric 
        self.root = root
    
    def compute_metrics(self, metrics_local) :
        if self.name_metric not in metrics_local :
            Exception(f"You must first compute {self.name_metric}")
        
        metric = metrics_local[self.name_metric]

        sum = np.sum([value[0] for value in metric])
        nb_value = np.sum([value[1] for value in metric])

        if self.root :
            return np.sqrt(sum/nb_value)
        return sum/nb_value
    
class nb_values_global_computer :
    def __init__(self, name_metric):
        self.name_metric = name_metric
    
    def compute_metrics(self, metrics_local) :
        if self.name_metric not in metrics_local :
            Exception(f"You must first compute {self.name_metric}")
        
        return np.sum([value[1] for value in metrics_local[self.name_metric]])

class concat_global_computer:
    def __init__(self, name_metric, nb_dim):
        self.name_metric = name_metric
        self.nb_dim = nb_dim
    
    def compute_metrics(self, metrics_local) :
        if self.name_metric not in metrics_local :
            Exception(f"You must first compute {self.name_metric}")
        
        metric = metrics_local[self.name_metric]
        if len(metric) == 1:
            return np.concatenate(metric)
        return  tuple([np.concatenate([value[i] for value in metric]) for i in range(self.nb_dim)])
