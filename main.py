import os, sys
import numpy as np
import argparse
import math
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.experience_replay import ExperienceReplay

from utils.data_utils import get_dataset, get_backbone, filter_classes, get_split
from utils.data_utils import get_transform, transform_resize, progress_bar
from utils.loggers import CsvLogger    

def get_args_parser():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'er')
    parser.add_argument('--root', type = str, default = 'home/arahmoun/dl-forgetting/dataset/data')
    parser.add_argument('--dataset', type = str, default = 'cifar10')
    parser.add_argument('--n_classes', type = int, default = 10)
    parser.add_argument('--n_tasks', type = int, default = 5)
    parser.add_argument('--buffer_size', type = int, default = 200)

    parser.add_argument('--lr', type = float, default = 0.03)
    parser.add_argument('--n_epochs', type = int, default = 20)
    parser.add_argument('--batch_size', type = int, default = 32)
    
    parser.add_argument('--device_id', type = int, default = 0)
    parser.add_argument('--seed', type = int, default = 123)

    parser.add_argument('--initial_weight', type=float, default=1e-3, help="Initial weight for sample differences")
    parser.add_argument('--scale_factor', type=float, default=0.1, help="Factor for scaling differences in loss calculation")
    parser.add_argument('--weight_multiplier', type=float, default=1.2, help="Multiplier for updating buffer weights")
    parser.add_argument('--rehearsal_ratio', type=float, default=1.0, help="Ratio of buffer samples to use during rehearsal")
    parser.add_argument('--cropping_weight', type=float, default=1.0, help="Maximum weight value")
    parser.add_argument('--growth_factor', type=float, default=1.1, help="Maximum weight value")

    return parser

def train(cl_model, train_loader, t, args):
    cl_model.net.train()
    
    cl_model.optim = torch.optim.SGD(cl_model.net.parameters(), lr=args.lr)

    scheduler = None
    if args.dataset == 'cifar10':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(cl_model.optim, [15], gamma=0.1)    
    if args.dataset == 'cifar100':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(cl_model.optim, [35, 45], gamma=0.1)
    if args.dataset == 'tinyimg':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(cl_model.optim, [70, 90], gamma=0.1)

    args.stop_kd = False
    
    # Tensor to store cumulative differences per sample
    num_samples = len(train_loader.dataset)
    sample_differences = torch.full((num_samples,), args.initial_weight, device=args.device)

    for epoch in range(args.n_epochs):    
        epoch_impact = args.growth_factor ** epoch
        for batch_idx, data in enumerate(train_loader):      
            img, label = data
            img, label = img.to(args.device), label.to(args.device)

            # Observe and collect the differences
            loss, diff_array = cl_model.observe(img, label, args.rehearsal_ratio)

            # Track differences per sample
            start_idx = batch_idx * train_loader.batch_size
            end_idx = start_idx + img.size(0)
            
            sample_differences[start_idx:end_idx] += (diff_array * epoch_impact).to(args.device)
            
            progress_bar(batch_idx, len(train_loader), t + 1, epoch + 1, loss.item())

        if scheduler is not None:
            scheduler.step()


    print("Updating Buffer...")
    # Scale the differences using the SCALE_FACTOR
    scaled_differences = torch.exp(args.scale_factor * sample_differences)

    # Normalize the scaled differences to create probabilities
    total_difference = scaled_differences.sum().item()

    

    sample_weights = scaled_differences / total_difference
    
    
    for batch_idx, data in enumerate(train_loader):
        img, label = data
        img, label = img.to(args.device), label.to(args.device)

        # Get weights for the current batch
        start_idx = batch_idx * train_loader.batch_size
        end_idx = start_idx + img.size(0)
        weights = sample_weights[start_idx:end_idx]
        

        # Add data to the buffer
        cl_model.buffer.add_data(img, labels=label, weights=weights)
        cl_model.buffer.update_weights(args.weight_multiplier, args.cropping_weight)
        
    # Debugging buffer weights after the update
    print("Buffer Weights After Update:", cl_model.buffer.weights)
                        
def evaluate(cl_model, test_splits, setting, args):
    cl_model.net.eval()    

    accs, accs_task = [], []    
    for t in range(len(test_splits)):
        metrics = {'correct': [], 'count': []}
        
        if setting == 'class_il':            
            metrics_task = {'correct': [], 'count': []}

        test_loader = DataLoader(test_splits[t], batch_size=args.batch_size, shuffle=False)

        for batch_idx, data in enumerate(test_loader):
            img, label = data
            img, label = img.to(args.device), label.to(args.device)

            with torch.no_grad():   
                predict = cl_model.net(img)    

            pred = torch.max(predict, 1)[1]
            correct = pred.eq(label.data).cpu().sum().numpy()

            metrics['correct'].append(correct)
            metrics['count'].append(label.size(0))
                        
            if setting == 'class_il':            
                predict = filter_classes(predict, t, args.n_classes, args.n_tasks)                
                pred = torch.max(predict, 1)[1]
                correct = pred.eq(label.data).cpu().sum().numpy()

                metrics_task['correct'].append(correct)
                metrics_task['count'].append(label.size(0))
                        
        accs.append(100 * np.sum(metrics['correct']) / np.sum(metrics['count']))        
        if setting == 'class_il':            
            accs_task.append(100 * np.sum(metrics_task['correct']) / np.sum(metrics_task['count']))
    
    return accs, accs_task 

def main(args):    
    if torch.cuda.is_available():
        args.device = torch.device('cuda', args.device_id)
        torch.cuda.manual_seed(args.seed)    
    else:
        args.device = torch.device('cpu')    

                 
    tf_resize = transform_resize(args.dataset)
    
    info = f'model: {args.model}, dataset: {args.dataset}({args.n_tasks} tasks)'    
    print(info)                          

    if args.dataset in ['perm-mnist', 'rot-mnist']:
        setting = 'domain_il'
        
    elif args.dataset in ['cifar10', 'cifar100', 'tinyimg']:
        setting = 'class_il'
        train_set = get_dataset(args.root, args.dataset, train=True, transform=transforms.ToTensor())
        
        tf_test = get_transform(args.dataset, train=False)        
        test_set = get_dataset(args.root, args.dataset, train=False, transform=tf_test)            

    net = get_backbone(args.dataset, args.n_classes).to(args.device)     
    args.setting = setting
    
    cl_model = ExperienceReplay(net, args)     
            
    results, results_task = [], []                
    test_splits = []
    
    for t in range(args.n_tasks):    
        transform = get_transform(args.dataset, train=True)  
        args.transform = transforms.Compose([transforms.ToPILImage(), transform])         
                
        if setting == 'class_il':            
            train_split = get_split(train_set, t, args.n_classes, args.n_tasks)
            test_split = get_split(test_set, t, args.n_classes, args.n_tasks)
            
        elif setting == 'domain_il':                
            # for Domain-IL, use a same transform for both train and test
            train_split = get_dataset(args.root, args.dataset, train=True, transform=transforms.ToTensor())
            test_split = get_dataset(args.root, args.dataset, train=False, transform=transform)               
            
        test_splits.append(test_split)
        train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True)                                
            
        train(cl_model, train_loader, t, args)
        accs = evaluate(cl_model, test_splits, setting, args)
        
        if hasattr(cl_model, 'end_task'):
            cl_model.end_task()
        
        results.append(accs[0])                
        
        if setting == 'class_il':  
            results_task.append(accs[1])        
            print ('\nclass-il: ', [float('{:.02f}'.format(d)) for d in accs[0]], round(np.mean(accs[0]), 2))
            print ('task-il : ', [float('{:.02f}'.format(d)) for d in accs[1]], round(np.mean(accs[1]), 2))
        else:
            print ('\ndomain-il: ', [float('{:.02f}'.format(d)) for d in accs[0]], round(np.mean(accs[0]), 2))
            
#         save_name = os.path.join(save_path, f'{args.aux_dataset}_{t}.pt')
#         torch.save(cl_model.net.state_dict(), save_name)
            
    csv_logger = CsvLogger()    
    csv_logger.write(setting, results, args)
    
    # task_il results when setting is class_il
    if setting == 'class_il':          
        csv_logger.write('task_il', results_task, args)




if __name__ == '__main__':
    args = get_args_parser().parse_args()    
    save_path = os.path.join('ckpt', f'{args.model}_{args.dataset}_{args.n_tasks}')        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    main(args)        
    print()
