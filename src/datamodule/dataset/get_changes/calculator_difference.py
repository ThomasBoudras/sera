import torch

class differenceCalculator:
    def __init__(self, min_difference, max_difference):
        self.min_difference = min_difference
        self.max_difference = max_difference

    def __call__(self, image_t1, image_t2):
        difference  = image_t2 - image_t1
        difference = torch.clip(difference, min=self.min_difference, max=self.max_difference)
        return difference
