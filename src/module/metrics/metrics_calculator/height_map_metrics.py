import torch

class heightMapMetrics :
    #For this metrics, as we want to have a map at the end, we want to have a global metrics and not per image, so we will make the average for each pixel and not the average obtained for each image.
    def __init__(self, tree_cover_threshold, min_height_nMAE):
        self.tree_cover_threshold =  tree_cover_threshold
        self.min_height_nMAE = min_height_nMAE
        self.list_metrics = ["sum_error", "sum_squared_error", "sum_absolute_error", "sum_relative_error", "intersection", "union", "nb_values", "nb_values_min_height"]
        
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
            
            # Compute relative error metrics
            mask_min_height = masked_target >= self.min_height_nMAE
            accumulated_metrics[f"sum_relative_error"] += torch.sum((masked_pred[mask_min_height] - masked_target[mask_min_height])/ (1 + masked_target[mask_min_height]))

            # Compute tree cover metrics
            tree_cover_target = target >= self.tree_cover_threshold
            tree_cover_pred = pred >= self.tree_cover_threshold
            accumulated_metrics[f"intersection"] += torch.logical_and(tree_cover_pred, tree_cover_target).sum()
            accumulated_metrics[f"union"] += torch.logical_or(tree_cover_pred, tree_cover_target).sum()

            # Compute number of values for each mask
            accumulated_metrics[f"nb_values"] += mask.sum() 
            accumulated_metrics[f"nb_values_min_height"] += mask_min_height.sum()

    def final_compute(self, accumulated_metrics, final_results):
        # Initialize final results dictionary
        sum_absolute_error = accumulated_metrics["sum_absolute_error"]
        sum_error = accumulated_metrics["sum_error"]
        sum_squared_error = accumulated_metrics["sum_squared_error"]
        sum_relative_error = accumulated_metrics["sum_relative_error"]
        intersection = accumulated_metrics["intersection"]
        union = accumulated_metrics["union"]
        nb_values = accumulated_metrics["nb_values"]
        nb_values_min_height = accumulated_metrics["nb_values_min_height"]

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

            final_results[f"nMAE"] = (sum_relative_error / nb_values_min_height).to(torch.float32)

            # TreeCov - Treecover IoU
            final_results[f"TreeCov"] = (intersection / (union + 1)).to(torch.float32) if union > 0 else torch.nan

            #nb_values - Number of values
            final_results["nb_values"] = nb_values.to(torch.float32)
        else :
            # If no values, set all metrics to NaN
            final_results[f"ME"] = torch.nan
            final_results[f"MAE"] = torch.nan
            final_results[f"RMSE"] = torch.nan
            final_results[f"nMAE"] = torch.nan
            final_results[f"TreeCov"] = torch.nan
            final_results["nb_values"] = 0
 
