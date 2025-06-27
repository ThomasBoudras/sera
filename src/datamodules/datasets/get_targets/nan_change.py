import torch

class get_nan :
    def __call__(self, bounds, row, year_number, transform) :

        return torch.tensor(torch.nan)