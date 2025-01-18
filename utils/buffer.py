# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from copy import deepcopy

def reservoir(num_seen_examples: int, buffer_size: int, current_weights: torch.Tensor, new_weight: float) -> int:
    """
    Reservoir sampling algorithm that considers weights for replacement.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :param current_weights: tensor of current weights in the buffer
    :param new_weight: weight of the new data point
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples  # Directly add if buffer is not full
        
    # Insert the new weight at the start of the current weights
    extended_weights = torch.cat([torch.tensor([1 / (new_weight + 1e-6)], device=current_weights.device), 1 / (current_weights + 1e-6)])

    # Choose a random index based on the extended weights (without normalizing)
    chosen_index = torch.multinomial(extended_weights, 1).item()
    
    # Return the index - 1 to adjust for insertion at the start
    return chosen_index - 1



def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'weights']

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)


    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor, weights: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None, weights=None):
        """
        Adds the data to the memory buffer using the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param weights: tensor containing the weights for each sample
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, weights)

        for i in range(examples.shape[0]):
            # Default weight is 1.0 if not provided
            new_weight = weights[i].item() if weights is not None else 1.0
            # Call the updated reservoir function to get the index for replacement
            index = reservoir(
                self.num_seen_examples,
                self.buffer_size,
                self.weights[:len(self)],  # Only consider valid weights in the buffer
                new_weight
            )

            # Increment the total number of seen examples
            self.num_seen_examples += 1

            # Replace the data at the chosen index if valid
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                self.weights[index] = new_weight
        

    def get_data(self, size: int, transform: transforms = None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items, considering weights.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :param return_index: whether to return the selected indices
        :return: a tuple containing the sampled data and optionally the indices
        """
        if size > len(self):
            size = len(self)
    
        if transform is None:
            transform = lambda x: x
    
        # Use weights to sample indices
        probabilities = self.weights[:len(self)]  # Consider only valid weights
        indices = torch.multinomial(probabilities, size, replacement=False)
    
        # Collect the sampled data
        ret_tuple = (torch.stack([transform(self.examples[idx].cpu()) for idx in indices]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[indices],)
    
        if return_index:
            return (indices,) + ret_tuple
        else:
            return ret_tuple

    def get_data_by_index(self, indexes, transform: transforms=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple


    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def update_weights(self, multiplier: float, cropping_weight: float):
        """
        Updates the weights in the buffer by multiplying them with a given multiplier.
        :param multiplier: The multiplier to scale the weights.
        """
        if hasattr(self, 'weights') and self.weights is not None:
            # Scale the weights by the multiplier
            self.weights *= multiplier
            
            self.weights = torch.minimum(self.weights, torch.tensor(cropping_weight, device=self.weights.device))
            
            
        else:
            print("Weights attribute not initialized or empty. No update performed.")

