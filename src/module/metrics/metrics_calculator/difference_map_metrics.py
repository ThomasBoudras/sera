import torch

class differenceMapMetrics :
    #For this metrics, as we want to have a map at the end, we want to have a global metrics and not per image, so we will make the average for each pixel and not the average obtained for each image.
    def __init__(self, delta_threshold):
        self.delta_threshold = delta_threshold

    def get_required_states(self):
        # On définit le "contrat" : voici les états dont j'ai besoin.
        # Pour chacun, je spécifie comment il doit être réduit en DDP.
        return {
            "sum_error": "sum",
            "sum_squared_error": "sum",
            "sum_absolute_error": "sum",
            "change_intersection": "sum",
            "change_pred_sum": "sum",
            "change_target_sum": "sum",
            "nb_values": "sum",
        }
        
    def batch_update(self, pred, target, mask, states) :
        # Apply mask to target and prediction
        masked_target = target[mask]
        masked_pred = pred[mask]
        
        if len(masked_pred) > 0 :
            # Compute error metrics
            error = masked_pred - masked_target
            states.sum_error += torch.sum(error)   
            states.sum_squared_error += torch.sum(error ** 2)
            states.sum_absolute_error += torch.sum(torch.abs(error))
            
            # Compute change metrics
            changes_pred = masked_pred <= self.delta_threshold
            changes_target = masked_target <= self.delta_threshold

            states.change_intersection += (changes_pred * changes_target).sum()  # TP
            states.change_pred_sum += changes_pred.sum()  # TP + FP
            states.change_target_sum += changes_target.sum()  # TP + FN
            states.nb_values += mask.sum() 

    def epoch_compute(self, states):
        # Initialize final results dictionary
        final_results = {}

        # If there are values, compute metrics
        if states.nb_values > 0 :
            # ME - Mean Error
            final_results["ME"] = (states.sum_error / states.nb_values).to(torch.float32)

            # MAE - Mean Absolute Error
            final_results["MAE"] = (states.sum_absolute_error / states.nb_values).to(torch.float32)

            # RMSE - Root Mean Square Error
            final_results["RMSE"] = torch.sqrt(states.sum_squared_error / states.nb_values).to(torch.float32)

            # Recall, Precision and F1 score
            recall = (states.change_intersection / states.change_target_sum) if states.change_target_sum > 0 else torch.tensor(torch.nan) 
            precision = (states.change_intersection / states.change_pred_sum) if states.change_pred_sum > 0 else torch.tensor(torch.nan) 
            f1_score = (2 * states.change_intersection / (states.change_pred_sum + states.change_target_sum)) if states.change_pred_sum + states.change_target_sum > 0 else torch.tensor(torch.nan)
            
            final_results["recall"] = recall.to(torch.float32)
            final_results["precision"] = precision.to(torch.float32)
            final_results["f1_score"] = f1_score.to(torch.float32)

            final_results["nb_values"] = states.nb_values.to(torch.float32)
        else :
            # If no values, set all metrics to NaN
            final_results["ME"] = torch.tensor(torch.nan)
            final_results["MAE"] = torch.tensor(torch.nan)
            final_results["RMSE"] = torch.tensor(torch.nan)
            final_results["recall"] = torch.tensor(torch.nan)
            final_results["precision"] = torch.tensor(torch.nan)
            final_results["f1_score"] = torch.tensor(torch.nan)
            final_results["nb_values"] = torch.tensor(0)
        
        return final_results 