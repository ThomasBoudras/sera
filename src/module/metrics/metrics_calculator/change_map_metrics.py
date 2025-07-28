import torch

class changeMapMetrics :
    #For this metrics, as we want to have a map at the end, we want to have a global metrics and not per image, so we will make the average for each pixel and not the average obtained for each image.
    def __init__(self, threshold_change):
        self.threshold_change =  threshold_change
        self.list_metrics = ["sum_error", "sum_squared_error", "sum_absolute_error", "change_intersection", "change_pred_sum", "change_target_sum", "nb_values"]
        
    def update(self, pred, target, mask, accumulated_metrics) :
        # Initialize metrics if not already present
        for metric in self.list_metrics :
            if not metric in accumulated_metrics :
                accumulated_metrics[metric] = 0
              
        # Apply mask to target and prediction
        masked_target = target[mask].float()
        masked_pred = pred[mask]

        # If there are values, compute metrics
        if len(masked_pred) > 0 :
            # Compute continuous metrics
            accumulated_metrics[f"continuous_intersection"] += (masked_pred * masked_target).sum()  # TP
            accumulated_metrics[f"continuous_pred_sum"] += masked_pred.sum()  # TP + FP
            
            # Compute discrete metrics
            discrete_masked_pred  = (masked_pred >= self.threshold_change).float()  
            accumulated_metrics[f"discrete_intersection"] += (discrete_masked_pred * masked_target).sum()  # TP
            accumulated_metrics[f"discrete_pred_sum"] += discrete_masked_pred.sum()

            # Compute target sum
            accumulated_metrics[f"target_sum"] += masked_target.sum()  # TP + FN
            
            # Compute BCE loss
            log_pred = torch.log(masked_pred)               
            log_1_minus_pred = torch.log(1 - masked_pred)    
            bce_per_element = - (masked_target * log_pred + (1 - masked_target) * log_1_minus_pred)
            accumulated_metrics[f"sum_bce_per_element"] += bce_per_element.sum()

            # Compute number of values
            accumulated_metrics[f"nb_values"] += len(masked_pred) 


    def final_compute(self, accumulated_metrics, final_results):
        # Initialize final results dictionary
        final_results = {}
        continuous_intersection = accumulated_metrics["continuous_intersection"]
        continuous_pred_sum = accumulated_metrics["continuous_pred_sum"]
        discrete_intersection = accumulated_metrics["discrete_intersection"]
        discrete_pred_sum = accumulated_metrics["discrete_pred_sum"]
        target_sum = accumulated_metrics["target_sum"]
        sum_bce_per_element = accumulated_metrics["sum_bce_per_element"]
        nb_values = accumulated_metrics["nb_values"]

        # If there are values, compute metrics
        if nb_values > 0 :
            final_results[f"continuous_recall"] = continuous_intersection / target_sum if target_sum > 0 else torch.nan 
            final_results[f"continuous_precision"] = continuous_intersection/continuous_pred_sum if continuous_pred_sum > 0 else torch.nan 
            final_results[f"continuous_f1_score"] = 2*continuous_intersection/(continuous_pred_sum+target_sum) if continuous_pred_sum+target_sum > 0 else torch.nan
            

            final_results[f"discrete_recall"] = discrete_intersection/target_sum if target_sum > 0 else torch.nan 
            final_results[f"discrete_precision"] = discrete_intersection /discrete_pred_sum if discrete_pred_sum > 0 else torch.nan 
            final_results[f"discrete_f1_score"] = 2*discrete_intersection/(discrete_pred_sum+target_sum) if discrete_pred_sum+target_sum > 0 else torch.nan
            
            final_results["bce_loss"] = sum_bce_per_element / nb_values 

            final_results["nb_values"] = nb_values

        else :
            # If no values, set all metrics to NaN
            final_results[f"continuous_recall"] = torch.nan
            final_results[f"continuous_precision"] = torch.nan
            final_results[f"continuous_f1_score"] = torch.nan
            final_results[f"discrete_recall"] = torch.nan
            final_results[f"discrete_precision"] = torch.nan
            final_results[f"discrete_f1_score"] = torch.nan
            final_results[f"bce_loss"] = torch.nan
            final_results["nb_values"] = 0

