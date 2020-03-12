import json
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch.nn.functional as F
import torch
import tqdm
from util.config import *

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, weights, n_epochs=None, fold=None,
          num_classes=None):
    
    """
    training script which include loss calculate, segment loss, class loss, scene loss and sync loss.

    Args:
        args: parameters
        model: load model
        
    Returns:
        train_file_names: train image list
        val_file_names: val image list  
        criterion: loss segment loss
        train_loader: train loader
        valid_loader: val loader
        validation: validation metric funcion
        weights: each class weights
        n_epochs: number of epochs
        fold: validation set id
        
    Raises:
        None
    """ 
    
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)
    root = Path(args.root)
    model_path = root / 'model_{fold}_{net}.pt'.format(fold=fold, net=args.model)
    
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))
    
    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    
    best_iou = -999999
    
    for epoch in range(epoch, n_epochs + 1):
        
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        
        try:
            mean_loss = 0
            
            for i, (inputs, targets, filename) in enumerate(tl):
                
                inputs = cuda(inputs)
                with torch.no_grad():
                    targets = cuda(targets)
                    
                output_seg, output_class, output_scene = model(inputs)
                loss_seg = criterion(output_seg, targets)  
          
                scene_gt = filename[0].split('/')[2]
                if scene_gt in scene_0:
                    scene_gt = [0]
                elif scene_gt in scene_1:
                    scene_gt = [1]
                elif scene_gt in scene_2:
                    scene_gt = [2] 
                scene_gt = torch.cuda.LongTensor(scene_gt)    

                class_gt = [[1 if targets[0, c,:,:].sum()>0 else 0 for c in range(num_classes)]]
                class_gt = torch.cuda.FloatTensor(class_gt)
                
                loss_class = F.binary_cross_entropy(F.sigmoid(output_class), class_gt)

                loss_scene = F.cross_entropy(output_scene, scene_gt)
                
                class_pre = [[1 if (torch.exp(output_seg[0, c,:,:]) > 0.5).sum()>0 else 0 for c in range(num_classes)]]
                class_pre = torch.cuda.FloatTensor(class_pre)
                
                loss_sync = F.binary_cross_entropy(F.sigmoid(output_class), class_pre)
                
                alpha, beta, gama = weights
                loss = loss_seg + alpha * loss_class + beta * loss_sync + gama * loss_scene 
                
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
                    
            write_event(log, step, loss=mean_loss)
            tq.close()
            
            valid_metrics = validation(model, criterion, valid_loader, num_classes)
            valid_iou = valid_metrics[1] 
            
            if valid_iou > best_iou:
                save(epoch + 1)
                best_iou = valid_iou
                print('Save current best weights with {:.5f} IoU with {} epochs'.format(best_iou, epoch + 1))
                         
        except KeyboardInterrupt:
            
            print('Interrupt the training stage')
            return
