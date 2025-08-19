import torch

class changeMapMetrics :
    #For this metrics, as we want to have a map at the end, we want to have a global metrics and not per image, so we will make the average for each pixel and not the average obtained for each image.
    def __init__(self, threshold_change):
        self.threshold_change =  threshold_change

    def get_required_states(self):
        return {
            "continuous_intersection": "sum",
            "continuous_pred_sum": "sum",
            "discrete_intersection": "sum",
            "discrete_pred_sum": "sum",
            "target_sum": "sum",
            "sum_bce_per_element": "sum",
            "nb_values": "sum",
        }
        
    def batch_update(self, pred, target, mask, states) :
        # Apply mask to target and prediction
        masked_target = target[mask].float()
        masked_pred = pred[mask]

        # If there are values, compute metrics
        if len(masked_pred) > 0 :
            # Compute continuous metrics
            states.continuous_intersection += (masked_pred * masked_target).sum()  # TP
            states.continuous_pred_sum += masked_pred.sum()  # TP + FP
            
            # Compute discrete metrics
            discrete_masked_pred  = (masked_pred >= self.threshold_change).float()  
            states.discrete_intersection += (discrete_masked_pred * masked_target).sum()  # TP
            states.discrete_pred_sum += discrete_masked_pred.sum()

            # Compute target sum
            states.target_sum += masked_target.sum()  # TP + FN
            
            # Compute BCE loss with epsilon for numerical stability
            log_pred = torch.log(masked_pred + 1e-6)               
            log_1_minus_pred = torch.log(1 - masked_pred + 1e-6)    
            bce_per_element = - (masked_target * log_pred + (1 - masked_target) * log_1_minus_pred)
            states.sum_bce_per_element += bce_per_element.sum()

            # Compute number of values
            states.nb_values += mask.sum() 

    def epoch_compute(self, states):
        # Initialize final results dictionary
        final_results = {}
        
        # If there are values, compute metrics
        if states.nb_values > 0 :
            continuous_recall = (states.continuous_intersection / states.target_sum).to(torch.float32) if states.target_sum > 0 else torch.nan 
            continuous_precision = (states.continuous_intersection/states.continuous_pred_sum).to(torch.float32) if states.continuous_pred_sum > 0 else torch.nan 
            continuous_f1_score = (2*states.continuous_intersection/(states.continuous_pred_sum + states.target_sum)).to(torch.float32) if states.continuous_pred_sum + states.target_sum > 0 else torch.nan
            
            final_results["continuous_recall"] = continuous_recall
            final_results["continuous_precision"] = continuous_precision
            final_results["continuous_f1_score"] = continuous_f1_score

            discrete_recall = (states.discrete_intersection/states.target_sum).to(torch.float32) if states.target_sum > 0 else torch.nan 
            discrete_precision = (states.discrete_intersection /states.discrete_pred_sum).to(torch.float32) if states.discrete_pred_sum > 0 else torch.nan 
            discrete_f1_score = (2*states.discrete_intersection/(states.discrete_pred_sum+states.target_sum)).to(torch.float32) if states.discrete_pred_sum+states.target_sum > 0 else torch.nan

            final_results["discrete_recall"] = discrete_recall
            final_results["discrete_precision"] = discrete_precision
            final_results["discrete_f1_score"] = discrete_f1_score
            
            final_results["bce_loss"] = (states.sum_bce_per_element / states.nb_values).to(torch.float32)

            final_results["nb_values"] = states.nb_values.to(torch.float32)

        else :
            # If no values, set all metrics to NaN
            final_results["continuous_recall"] = torch.nan
            final_results["continuous_precision"] = torch.nan
            final_results["continuous_f1_score"] = torch.nan
            final_results["discrete_recall"] = torch.nan
            final_results["discrete_precision"] = torch.nan
            final_results["discrete_f1_score"] = torch.nan
            final_results["bce_loss"] = torch.nan
            final_results["nb_values"] = 0
        
        return final_results

