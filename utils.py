import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np
import torch.nn.functional as F
import torch
import tqdm


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


train_1=['seq_1','seq_2','seq_4','seq_5','seq_7']
train_2=['seq_3','seq_10','seq_11','seq_12','seq_16']
train_3=['seq_6','seq_9','seq_13','seq_14','seq_15']

def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None,
          num_classes=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{fold}_{net}_{weight}_scene_class.pt'.format(fold=fold, net=args.model, weight=args.loss_weight)
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
    valid_losses = []
    best_iou = 0
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
                outputs, se1, se4, se3 = model(inputs)
                loss = criterion(outputs, targets)  
                
                
                class_gt=filename[0].split('/')[1]
                #print(class_gt)
                #assert(0) 
                if class_gt in train_1:
                    class_gt=[0]
                elif class_gt in train_2:
                    class_gt=[1]
                elif class_gt in train_3:
                    class_gt=[2]
                 
                class_gt = torch.cuda.LongTensor(class_gt)    


                            


#                
                exist_class = [[1 if targets[0, c,:,:].sum()>0 else 0 for c in range(12)]]
                exist_class = torch.cuda.FloatTensor(exist_class)


                se1_loss = F.cross_entropy(se1, class_gt)

                se4_loss = F.binary_cross_entropy(F.sigmoid(se4), exist_class)
                se3_loss = F.binary_cross_entropy(F.sigmoid(se3), exist_class)
 
                
                loss+=   0.1 * se3_loss + 0.1 * se4_loss + 1.0 * se1_loss 
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
#            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics[0]
            valid_iou = valid_metrics[1] 
            if valid_iou>best_iou:
                save(epoch + 1)
                best_iou=valid_iou
                print('save_best')
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            print('done.')
            return
