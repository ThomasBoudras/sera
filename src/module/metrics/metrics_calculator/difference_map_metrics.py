import torch

class differenceMapMetrics :
    #For this metrics, as we want to have a map at the end, we want to have a global metrics and not per image, so we will make the average for each pixel and not the average obtained for each image.
    def __init__(self, delta_threshold):
        self.delta_threshold = delta_threshold
        self.list_metrics = ["sum_error", "sum_squared_error", "sum_absolute_error", "change_intersection", "change_pred_sum", "change_target_sum", "nb_values"]
        
    def update(self, pred, target, mask, accumulated_metrics) :
        # Initialize metrics if not already present
        for metric in self.list_metrics :
            if not metric in accumulated_metrics :
                accumulated_metrics[metric] = 0
              
        # Apply mask to target and prediction
        masked_target = target[mask]
        masked_pred = pred[mask]
        if len(masked_pred) > 0 :
            # Compute error metrics
            error = masked_pred - masked_target
            accumulated_metrics[f"sum_error"] += torch.sum(error)      
            accumulated_metrics[f"sum_squared_error"] += torch.sum(error ** 2)
            
            absolute_error = torch.abs(error)
            accumulated_metrics[f"sum_absolute_error"] += torch.sum(absolute_error)
            
            # Compute change metrics
            changes_pred = masked_pred <= self.delta_threshold
            changes_target = masked_target <= self.delta_threshold

            accumulated_metrics[f"change_intersection"] += (changes_pred * changes_target).sum()  # TP
            accumulated_metrics[f"change_pred_sum"] += changes_pred.sum()  # TP + FP
            accumulated_metrics[f"change_target_sum"] += changes_target.sum()  # TP + FN
            accumulated_metrics[f"nb_values"] += len(masked_pred) 

    def final_compute(self, accumulated_metrics, final_results):
        # Initialize final results dictionary
        final_results = {}
        sum_error = accumulated_metrics["sum_error"]
        sum_squared_error = accumulated_metrics["sum_squared_error"]
        sum_absolute_error = accumulated_metrics["sum_absolute_error"]
        change_intersection = accumulated_metrics["change_intersection"]
        change_pred_sum = accumulated_metrics["change_pred_sum"]
        change_target_sum = accumulated_metrics["change_target_sum"]
        nb_values = accumulated_metrics["nb_values"]

        # If there are values, compute metrics
        if nb_values > 0 :
            # ME - Mean Error
            me = sum_error / nb_values
            final_results[f"ME"] = me.to(torch.float32)

            # MAE - Mean Absolute Error
            mae = sum_absolute_error / nb_values
            final_results[f"MAE"] = mae.to(torch.float32)

            # RMSE - Root Mean Square Error
            final_results[f"RMSE"] = torch.sqrt(sum_squared_error / nb_values).to(torch.float32)

            # Recall, Precision and F1 score
            final_results[f"recall"] = (change_intersection / change_target_sum).to(torch.float32) if change_target_sum > 0 else torch.nan 
            final_results[f"precision"] = (change_intersection/change_pred_sum).to(torch.float32) if change_pred_sum > 0 else torch.nan 
            final_results[f"f1_score"] = (2*change_intersection/(change_pred_sum + change_target_sum)).to(torch.float32) if change_pred_sum + change_target_sum > 0 else torch.nan

            final_results["nb_values"] = nb_values.to(torch.float32)
        else :
            # If no values, set all metrics to NaN
            final_results[f"ME"] = torch.nan
            final_results[f"MAE"] = torch.nan
            final_results[f"RMSE"] = torch.nan
            final_results[f"Bias"] = torch.nan
            final_results[f"recall"] = torch.nan
            final_results[f"precision"] = torch.nan
            final_results[f"f1_score"] = torch.nan
            final_results["nb_values"] = 0
    