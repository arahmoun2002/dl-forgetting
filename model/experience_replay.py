import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import torchvision.transforms.functional as ttf

from utils.buffer import Buffer

def tf_tensor(xs, transforms):
    device = xs.device

    xs = torch.cat([transforms(x).unsqueeze_(0) 
                    for x in xs.cpu()], dim=0)

    return xs.to(device=device)

class ExperienceReplay:
    def __init__(self, net, args):
        super(ExperienceReplay, self).__init__()                
        self.net = net
        self.net_old = None
        self.optim = None        
        
        self.args = args
        self.buffer = Buffer(args.buffer_size, args.device)

    def end_task(self):          
        self.net_old = deepcopy(self.net)
        self.net_old.eval()        
        
    def observe(self, inputs, labels, ratio):
        
        self.optim.zero_grad()
            
        inputs_aug = tf_tensor(inputs, self.args.transform)    
        outputs = self.net(inputs_aug)
        loss = F.cross_entropy(outputs, labels)
        
        # Calculate predicted class and confidence
        probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        pred = torch.max(outputs, 1)[1]  # Get predicted class
        confidences = probs[range(len(labels)), labels]  # Confidence for predicted class

        # Probability of misclassification
        diff_array = (1-confidences) 
        
        if self.net_old is not None:
            if self.args.setting == 'domain_il': 
                augment = None
            else:
                augment = self.args.transform
            
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(int(inputs.size(0) * ratio), transform=augment)
            buf_outputs = self.net(buf_inputs)
            loss += F.cross_entropy(buf_outputs, buf_labels) * ratio    #ratio of buffer sample  that we train on the next task
            
            
        loss.backward()
        self.optim.step()
            
        
        return loss, diff_array
 
    
