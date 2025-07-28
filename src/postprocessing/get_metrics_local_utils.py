import numpy as np


class get_metrics_local:
    def __init__(self, metrics_set) :
        self.metrics_set = metrics_set

    def __call__(self, images):
        metrics = {}
        for metric_name, metric_computer in self.metrics_set.items():
            metrics[metric_name] =  metric_computer.compute_metrics(images)
        return metrics


class precision_local_computer:
    def __init__(self, name_image_1, name_image_2):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        
    def compute_metrics(self, images) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]

        nan_mask = np.isnan(image_1) | np.isnan(image_2)

        # Ensure the images are binary
        image_1 = image_1[~nan_mask].astype(bool)
        image_2 = image_2[~nan_mask].astype(bool)
        
        if image_2.sum() == 0 :
            return np.nan
        
        # Calculate True Positives (TP) and False Positives (FP)
        true_positives = np.sum(image_1 & image_2)
        false_positives = np.sum(image_1 & ~image_2)
        
        # Calculate Precision
        if true_positives + false_positives == 0:
            return 0
        precision = true_positives / (true_positives + false_positives)
        return precision


class  recall_local_computer :
    def __init__(self, name_image_1, name_image_2):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        
    def compute_metrics(self, images) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]
        
        nan_mask = np.isnan(image_1) | np.isnan(image_2)

        # Ensure the images are binary
        image_1 = image_1[~nan_mask].astype(bool)
        image_2 = image_2[~nan_mask].astype(bool)
        
        if image_2.sum() == 0 :
            return np.nan
        
        # Calculate True Positives (TP) and False Negatives (FN)
        true_positives = np.sum(image_1 & image_2)
        false_negatives = np.sum(~image_1 & image_2)
        
        # Calculate Recall
        if true_positives + false_negatives == 0:
            return 0
        recall = true_positives / (true_positives + false_negatives)
        return recall
    

class f1_score_local_computer:
    def __init__(self, name_image_1, name_image_2):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        
    def compute_metrics(self, images) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]
        
        nan_mask = np.isnan(image_1) | np.isnan(image_2)
        # Ensure the images are binary
        image_1 = image_1[~nan_mask].astype(bool)
        image_2 = image_2[~nan_mask].astype(bool)

        if image_2.sum() == 0 :
            return np.nan
        
        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        true_positives = np.sum(image_1 & image_2)
        false_positives = np.sum(image_1 & ~image_2)
        false_negatives = np.sum(~image_1 & image_2)
        
        # Calculate Precision and Recall
        if true_positives + false_positives == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)
        
        if true_positives + false_negatives == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)
        
        # Calculate F1 Score
        if precision + recall == 0 :
            return 0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


class true_positive_local_computer:
    def __init__(self, name_image_1, name_image_2):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        
    def compute_metrics(self, images) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]

        nan_mask = np.isnan(image_1) | np.isnan(image_2)

        # Ensure the images are binary
        image_1 = image_1[~nan_mask].astype(bool)
        image_2 = image_2[~nan_mask].astype(bool)
        
        
        return np.sum(image_1 & image_2)


class false_positive_local_computer:
    def __init__(self, name_image_1, name_image_2):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        
    def compute_metrics(self, images) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]

        nan_mask = np.isnan(image_1) | np.isnan(image_2)

        # Ensure the images are binary
        image_1 = image_1[~nan_mask].astype(bool)
        image_2 = image_2[~nan_mask].astype(bool)
          
        return np.sum(image_1 & ~image_2)
    
class false_negative_local_computer:
    def __init__(self, name_image_1, name_image_2):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        
    def compute_metrics(self, images) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]

        nan_mask = np.isnan(image_1) | np.isnan(image_2)

        # Ensure the images are binary
        image_1 = image_1[~nan_mask].astype(bool)
        image_2 = image_2[~nan_mask].astype(bool)
          
        return np.sum(~image_1 & image_2)

class mae_local_computer:
    def __init__(self, name_image_1, name_image_2):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        
    def compute_metrics(self, images) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")
        
        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]
        
        nan_mask = np.isnan(image_1) | np.isnan(image_2)

        image_1 = image_1[~nan_mask]
        image_2 = image_2[~nan_mask]
        
        absolute_differences = np.abs(image_1 - image_2)
        sum = np.sum(absolute_differences)
        nb_value = np.sum(~nan_mask)
        return sum, nb_value

class mae_lower_than_local_computer:
    def __init__(self, name_image_1, name_image_2, threshold):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        self.threshold = threshold
        
    def compute_metrics(self, images) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]
        
        mask = np.isnan(image_1) | np.isnan(image_2) | (image_1 >= self.threshold)

        
        image_1 = image_1[~mask]
        image_2 = image_2[~mask]
        
        absolute_differences = np.abs(image_1 - image_2)
        sum = np.sum(absolute_differences)
        nb_value = np.sum(~mask)
        return sum, nb_value
    
class mae_greater_than_local_computer:
    def __init__(self, name_image_1, name_image_2, threshold):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        self.threshold = threshold
        
    def compute_metrics(self, images) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]
        
        mask = np.isnan(image_1) | np.isnan(image_2) | (image_1 <= self.threshold)
        image_1 = image_1[~mask]
        image_2 = image_2[~mask]
        
        absolute_differences = np.abs(image_1 - image_2)
        sum = np.sum(absolute_differences)
        nb_value = np.sum(~mask)
        return sum, nb_value

class rmse_local_computer:
    def __init__(self, name_image_1, name_image_2):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        
    def compute_metrics(self, images) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]
        
        nan_mask = np.isnan(image_1) | np.isnan(image_2)
        image_1 = image_1[~nan_mask]
        image_2 = image_2[~nan_mask]

        squared_difference = np.square(image_1 - image_2)
        sum = np.sum(squared_difference)
        nb_value = np.sum(~nan_mask)
        return sum, nb_value
    
class bias_local_computer:
    def __init__(self, name_image_1, name_image_2):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        
    def compute_metrics(self, images) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]
        
        nan_mask = np.isnan(image_1) | np.isnan(image_2)
        image_1 = image_1[~nan_mask]
        image_2 = image_2[~nan_mask]
        
        difference = image_1 - image_2
        sum = np.sum(difference)
        nb_value = np.sum(~nan_mask)
        return sum, nb_value
