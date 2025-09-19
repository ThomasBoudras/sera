import numpy as np


class get_metrics_local:
    def __init__(self, metrics_set) :
        self.metrics_set = metrics_set

    def __call__(self, images, row):
        metrics = {}
        for metric_name, metric_computer in self.metrics_set.items():
            metrics[metric_name] =  metric_computer.compute_metrics(images, metrics, row)
        return metrics



class true_positive_local_computer:
    def __init__(self, name_image_1, name_image_2):
        self.name_image_1 = name_image_1
        self.name_image_2 = name_image_2
        
    def compute_metrics(self, images, metrics_previous, row) :
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
        
    def compute_metrics(self, images, metrics_previous, row) :
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
        
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_image_1 not in images) or (self.name_image_2) not in images :
            Exception(f"You must first load {self.name_image_1} and {self.name_image_2}")

        image_1 = images[self.name_image_1]
        image_2 = images[self.name_image_2]

        nan_mask = np.isnan(image_1) | np.isnan(image_2)

        # Ensure the images are binary
        image_1 = image_1[~nan_mask].astype(bool)
        image_2 = image_2[~nan_mask].astype(bool)
          
        return np.sum(~image_1 & image_2)


class precision_local_computer:
    def __init__(self, name_true_positive, name_false_positive):
        self.name_true_positive = name_true_positive
        self.name_false_positive = name_false_positive
        
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_true_positive not in metrics_previous) or (self.name_false_positive) not in metrics_previous :
            Exception(f"You must first load {self.name_true_positive} and {self.name_false_positive}")

        image_true_positive = metrics_previous[self.name_true_positive]
        image_false_positive = metrics_previous[self.name_false_positive]
        
        # if there are no true positive, the precision is nan, we don't want to take it into account, the problem is due to the target mask
        if image_true_positive == 0 :
            return np.nan
        
        precision = image_true_positive / (image_true_positive + image_false_positive)
        return precision


class  recall_local_computer :
    def __init__(self, name_true_positive, name_false_negative):
        self.name_true_positive = name_true_positive
        self.name_false_negative = name_false_negative
        
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_true_positive not in images) or (self.name_false_negative) not in images :
            Exception(f"You must first load {self.name_true_positive} and {self.name_false_negative}")

        image_true_positive = metrics_previous[self.name_true_positive]
        image_false_negative = metrics_previous[self.name_false_negative]
        
        # if there are no true positive, the recall is nan, we don't want to take it into account, the problem is due to the target mask
        if image_true_positive == 0 :
            return np.nan
        
        recall = image_true_positive / (image_true_positive + image_false_negative)
        return recall
    

class f1_score_local_computer:
    def __init__(self, name_true_positive, name_false_negative, name_false_positive):
        self.name_true_positive = name_true_positive
        self.name_false_negative = name_false_negative
        self.name_false_positive = name_false_positive
        
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_true_positive not in images) or (self.name_false_negative) not in images or (self.name_false_positive) not in images :
            Exception(f"You must first load {self.name_true_positive} and {self.name_false_negative} and {self.name_false_positive}")

        image_true_positive = metrics_previous[self.name_true_positive]
        image_false_negative = metrics_previous[self.name_false_negative]
        image_false_positive = metrics_previous[self.name_false_positive]
        
        # if there are no true positive, the f1 score is nan, we don't want to take it into account, the problem is due to the target mask
        if image_true_positive == 0 :
            return np.nan
            
        precision = image_true_positive / (image_true_positive + image_false_positive)
        recall = image_true_positive / (image_true_positive + image_false_negative)

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score


class mean_local_computer:
    def __init__(self, metric_component_name, root=False):
        self.metric_component_name = metric_component_name
        self.root = root
        
    def compute_metrics(self, images, metrics_previous, row) :
        if self.metric_component_name not in metrics_previous :
            Exception(f"You must first load {self.metric_component_name}")
        
        metric_component = metrics_previous[self.metric_component_name]

        # if there are no value, the mean is nan, we don't want to take it into account, the problem is due to the target mask
        if metric_component[1] == 0 :
            return np.nan

        mean = metric_component[0]/metric_component[1]
        if self.root :
            return np.sqrt(mean)
        else :
            return mean
        
        
class mae_component_computer:
    def __init__(self, name_pred, name_target):
        self.name_pred = name_pred
        self.name_target = name_target
        
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
        
        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        nan_mask = np.isnan(image_pred) | np.isnan(image_target)

        image_pred = image_pred[~nan_mask]
        image_target = image_target[~nan_mask]
        
        absolute_differences = np.abs(image_pred - image_target)
        sum = np.sum(absolute_differences)
        nb_value = np.sum(~nan_mask)
        return sum, nb_value


class mae_lower_than_component_computer:
    def __init__(self, name_pred, name_target, threshold):
        self.name_pred = name_pred
        self.name_target = name_target
        self.threshold = threshold

    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")

        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        unvalid_mask = np.isnan(image_pred) | np.isnan(image_target) | (image_pred >= self.threshold)

        
        image_pred = image_pred[~unvalid_mask]
        image_target = image_target[~unvalid_mask]
        
        absolute_differences = np.abs(image_pred - image_target)
        sum = np.sum(absolute_differences)
        nb_value = np.sum(~unvalid_mask)
        return sum, nb_value
  


class mae_greater_than_component_computer:
    def __init__(self, name_pred, name_target, threshold):
        self.name_pred = name_pred
        self.name_target = name_target
        self.threshold = threshold
        
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")

        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        unvalid_mask = np.isnan(image_pred) | np.isnan(image_target) | (image_pred <= self.threshold)
        image_pred = image_pred[~unvalid_mask]
        image_target = image_target[~unvalid_mask]
        
        absolute_differences = np.abs(image_pred - image_target)
        sum = np.sum(absolute_differences)
        nb_value = np.sum(~unvalid_mask)
        return sum, nb_value


class rmse_component_computer:
    def __init__(self, name_pred, name_target):
        self.name_pred = name_pred
        self.name_target = name_target
        
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")

        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        nan_mask = np.isnan(image_pred) | np.isnan(image_target)
        image_pred = image_pred[~nan_mask]
        image_target = image_target[~nan_mask]

        squared_difference = np.square(image_pred - image_target)
        sum = np.sum(squared_difference)
        nb_value = np.sum(~nan_mask)
        return sum, nb_value


class me_component_computer:
    def __init__(self, name_pred, name_target):
        self.name_pred = name_pred
        self.name_target = name_target
           
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")

        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        nan_mask = np.isnan(image_pred) | np.isnan(image_target)
        image_pred = image_pred[~nan_mask]
        image_target = image_target[~nan_mask]
        
        difference = image_pred - image_target
        sum = np.sum(difference)
        nb_value = np.sum(~nan_mask)
        return sum, nb_value


class nmae_component_computer:
    def __init__(self, name_pred, name_target, min_target):
        self.name_pred = name_pred
        self.name_target = name_target
        self.min_target = min_target

    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
            
        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        unvalid_mask = np.isnan(image_pred) | np.isnan(image_target) | (image_target < self.min_target)
        image_pred = image_pred[~unvalid_mask]
        image_target = image_target[~unvalid_mask]
        
        absolute_differences = np.abs(image_pred - image_target)/(image_target + 1)
        sum = np.sum(absolute_differences)
        nb_value = np.sum(~unvalid_mask)
        return sum, nb_value


class treecover_local_computer:
    def __init__(self, name_pred, name_target, threshold):
        self.name_pred = name_pred
        self.name_target = name_target
        self.threshold = threshold

    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
        
        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        nan_mask = np.isnan(image_pred) | np.isnan(image_target)
        image_pred = image_pred[~nan_mask]
        image_target = image_target[~nan_mask]

        mask_pred = image_pred > self.threshold
        mask_target = image_target > self.threshold
        
        intersection = np.sum(mask_pred & mask_target)
        union = np.sum(mask_pred | mask_target)
        
        return intersection, union


class flatten_local_computer:
    def __init__(self, name_pred, name_target):
        self.name_pred = name_pred
        self.name_target = name_target
        
    def compute_metrics(self, images, metrics_previous, row) :
        if self.name_pred not in images or self.name_target not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
        
        image_pred = images[self.name_pred]
        image_target = images[self.name_target]
        
        nan_mask = np.isnan(image_pred) | np.isnan(image_target)
        image_pred = image_pred[~nan_mask]
        image_target = image_target[~nan_mask]
        
        return image_pred.flatten(), image_target.flatten()


class group_by_bins_local_computer:
    def __init__(self, name_pred, name_target, bins, method_metrics):
        self.name_pred = name_pred
        self.name_target = name_target
        self.bins = bins
        self.method_metrics = getattr(self, method_metrics)

    def absolute_error(self, pred, target):
        return np.abs(pred - target)
    
    def squared_error(self, pred, target):
        return np.square(pred - target)
    
    def error(self, pred, target):
        return pred - target
    
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
        
        target = images[self.name_target]
        pred = images[self.name_pred]

        nan_mask = np.isnan(target) | np.isnan(pred)

        target = target[~nan_mask]
        pred = pred[~nan_mask]


        bin_metrics = []
        for i in range(len(self.bins)-1):
            mask_bin = (target >= self.bins[i]) & (target < self.bins[i+1])
            if mask_bin.sum() > 0:
                bin_metric = self.method_metrics(pred[mask_bin], target[mask_bin]).flatten()
            else :
                bin_metric = np.array([])
            bin_metrics.append(bin_metric)

        return bin_metrics


class group_by_date_local_computer:
    """
    Computer that groups metrics by date bins based on a specified date column.
    
    This class processes images to compute metrics for specific date ranges,
    allowing for temporal analysis of predictions versus targets.
    
    Args:
        name_pred (str): Name of the prediction image in the images dictionary
        name_target (str): Name of the target image in the images dictionary
        name_base (str): Base name for the output metrics
        bins_dates (list): List of date bin tuples defining the date ranges
        date_column (str): Name of the column containing date information
        method_metrics (callable): Method to compute metrics between predictions and targets
    """
    def __init__(self, name_pred, name_target, bins_dates, date_column, method_metrics):
        self.name_pred = name_pred
        self.name_target = name_target
        self.bins_dates = bins_dates
        self.date_column = date_column 
        self.method_metrics = getattr(self, method_metrics)

    def absolute_error(self, pred, target):
        return np.abs(pred - target)
    
    def squared_error(self, pred, target):
        return np.square(pred - target)
    
    def error(self, pred, target):
        return pred - target
        
    def compute_metrics(self, images, metrics_previous, row) :
        if (self.name_pred not in images) or (self.name_target) not in images :
            Exception(f"You must first load {self.name_pred} and {self.name_target}")
        
        target = images[self.name_target]
        pred = images[self.name_pred]
        
        nan_mask = np.isnan(target) | np.isnan(pred)
        target = target[~nan_mask]
        pred = pred[~nan_mask]
        
        month = int(row[self.date_column][4:6])

        bin_metrics = []
        for bin in self.bins_dates:
            if (month >= int(bin[0])) and (month <= int(bin[1])):  # check if the month is in the bin
                bin_metrics.append(self.method_metrics(pred, target))
            else : 
                bin_metrics.append(np.array([]))

        return bin_metrics
        