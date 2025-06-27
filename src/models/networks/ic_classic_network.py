#End-to-end model with super resolution and subsequent regression
from torch import nn


class ICClassicNetwork(nn.Module):

    def __init__(self, 
                regression_model,
                change_method, 
                use_final_layer = False,
                ):
        super().__init__()
        self.regression_model = regression_model
        self.change_method = change_method
        self.use_final_layer = use_final_layer
        
        if self.use_final_layer:
            self.final_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

    def forward(self, inputs, meta_data):
        separation_inputs = int(inputs.shape[1]/2)  
        separation_meta_data = int(meta_data["dates"].shape[1]/2)

        inputs_1 = inputs[:, :separation_inputs, ...].contiguous()
        meta_data_1 = meta_data.copy()
        meta_data_1["dates"] = meta_data_1["dates"][:,:separation_meta_data].contiguous()

        inputs_2 = inputs[:, separation_inputs:, ...].contiguous()
        meta_data_2 = meta_data.copy()
        meta_data_2["dates"] = meta_data_2["dates"][:,separation_meta_data:].contiguous()

        preds_1 = self.regression_model(inputs_1, meta_data_1)

        preds_2 = self.regression_model(inputs_2, meta_data_2)

        pred = self.change_method(preds_1, preds_2)

        if self.use_final_layer:
            pred = self.final_layer(pred)
        return pred
