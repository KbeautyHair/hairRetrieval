import os
import math
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
import json
import shutil
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from dataloader import data_loader
from tqdm import tqdm

accumulated_cols = [27, 29, 36, 38, 40, 42, 50]
cols_list = [ 'basestyle', 'basestyle_type', 'length',
               'front', 'vertical', 'sex', 'horizontal']

def _infer(model, args, data_loader):
    res_fc = None
    res_id = None
    fc_list = []
    label_list = []
    with torch.no_grad():
        for index, (image, label) in tqdm(enumerate(data_loader)):
            if args.cuda:
                image = image['image'].cuda()
            fc = model(image)
            fc = fc.detach().cpu().numpy()
            if args.all_cols:
                before_index = 0
                new_fc = []
                for i, cur_index in enumerate(accumulated_cols):
                    new_fc.append(np.argmax(fc[:, before_index:cur_index], axis = -1))
                    before_index = cur_index
                fc_list.append(fc)
                label_list.append(label.detach().cpu().numpy())
            else:
                fc = np.argmax(fc, axis=-1)
                fc = fc.flatten()
                fc_list.append(fc)
                label_list.append(label.detach().cpu().numpy().flatten())

    if args.all_cols:
        #fc_list = np.array(fc_list)
        fc_list = np.concatenate(np.array(fc_list))

        #label_list = np.array(label_list)
        label_list = np.concatenate(np.array(label_list))
    else:
        fc_list = np.concatenate(np.array(fc_list)).ravel()
        label_list = np.concatenate(np.array(label_list).flatten()).ravel()
    #res_cls = np.argmax(res_fc, axis=1)
    #print('res_cls{}\n{}'.format(res_cls.shape, res_cls))

    return [fc_list, label_list]


def feed_infer(output_file, infer_func):
    prediction_name, prediction_class = infer_func()

    print('write output')
    predictions_str = []
    for index, name in enumerate(prediction_name):
        label = int(prediction_class[index])
        plant = label//21
        disease = label%21
        test_str = name + '\t' + str(plant) + '\t' + str(disease)
        predictions_str.append(test_str) 
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(predictions_str))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')



def validate(model, validate_dataloader, args):
    fc_list, label_list = _infer(model, args, data_loader=validate_dataloader)

    if args.all_cols:
        avg = 0.000001
        print('==========Eval acc score ==========')
        before_index = 0
        i = 0
        for col, cur_index in zip(cols_list,accumulated_cols):
            acc_score = accuracy_score(label_list[:,i], np.argmax(fc_list[:,before_index:cur_index], axis= -1))
            avg += acc_score
            i+=1
            before_index = cur_index
            print('{} : {:.4f}'.format(col,acc_score))
        acc_score = avg / 13
        print('Avg acc : {:.4f}'.format(acc_score))
        print('==========Eval acc score ==========')
    else:
        acc_score = accuracy_score(label_list, fc_list)
        print('Eval acc score : {:.4f}'.format(acc_score))

    return acc_score #auc_score


def test(prediction_file, model, test_dataloader, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=test_dataloader))


def save_model(checkpoint_dir, model, optimizer, scheduler):
    state = {
        'model': model.state_dict()
    }
    torch.save(state, os.path.join(checkpoint_dir + '.pth'))
    print('model saved')


def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')

def make_parser():
    args = argparse.ArgumentParser()

    #exp
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--output_dir", type=str, default='')

    args.add_argument("--train_batch_size", type=int, default=32)
    args.add_argument("--eval_batch_size", type=int, default=32)
    args.add_argument('--num_workers', type=int, default=32)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--amp", type=bool, default=False)

    #dataset
    # args.add_argument("--root", type=str, required = True, default='./data') # dataset direction ex) /tf/notebooks/task_11_dataset
    args.add_argument("--root", type=str, default='') # dataset direction ex) /tf/notebooks/task_11_dataset
    args.add_argument("--image_size", type=int, default=256)
    args.add_argument("--advprop", type=bool, default=False)

    #model
    args.add_argument("--exp_name", type=str, default='eb0_0102')
    args.add_argument("--model_name", type=str, default="efficientnet-b0")
    args.add_argument("--transfer", type=str, default=None)
    args.add_argument("--dropout", type=float, default=0.2)
    args.add_argument("--classes_num_1", type=int, default=1)
    # args.add_argument("--classes_num_2", type=int, default=21)
    args.add_argument("--prediction_file", type=str, default="")


    #hparams
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--num_epochs", type=int, default=150)
    # args.add_argument("--val_same_epoch", type=int, default=20)
    args.add_argument("--weight_decay", type=float, default=1e-5)
    args.add_argument("--criterion", type=str, default="crossentropy")
    args.add_argument("--optim", type=str, default="rangerlars")
    args.add_argument("--scheduler", type=str, default="cosine")
    args.add_argument("--warmup", type=int, default=5)
    args.add_argument("--cutmix_alpha", type=float, default=1)
    args.add_argument("--cutmix_prob", type=float, default=0.5)

    args.add_argument("--mode", type=str, default="train")

    args.add_argument("--gpu_id", type=str, default="0")
    args.add_argument('--focal_loss', action='store_true', default=False)
    args.add_argument('--num_classes', type=int, default = 27)

    args.add_argument('--all_cols', action='store_true', default=False)
    args.add_argument('--only_hair', action='store_true', default=False)
    args.add_argument('--sampler', action='store_true', default=False)
    args.add_argument('--ori_map', action='store_true', default=False)
    args.add_argument('--style_weight', type=float, default=1)

    config = args.parse_args()
    return config


import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def main(args):

    #set_seed(args.seed)
    # args = vars(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # For gpu
    DATASET_PATH = args.root
    torch.manual_seed(args.seed)
    #make dataset
    train_dataloader = data_loader(args=args, phase='train', batch_size=args.train_batch_size)
    validate_dataloader = data_loader(args=args, phase='val', batch_size=args.eval_batch_size)
    test_dataloader = data_loader(args=args, phase='test', batch_size=args.eval_batch_size)

    if args.all_cols :
        args.num_classes = 42+9
    #make model
    if 'efficient' in args.model_name:
        model = MultiheadClassifier(args)
    else:
        import torchvision.models as models
        if '18' in args.model_name:
            model = models.resnet18(pretrained=True)
        elif '34' in args.model_name:
            model = models.resnet34(pretrained=True)
        elif '50' in args.model_name:
            model = models.resnet50(pretrained=True)
        elif '101' in args.model_name:
            model = models.resnet101(pretrained=True)
        elif '50_32x4d' in args.model_name:
            model = models.resnext50_32x4d(pretrained=True)
        elif 'mnasnet' in args.model_name:
            model = models.mnasnet1_3()
            model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=args.num_classes, bias=True)
        elif 'wide50' in args.model_name:
            model = models.wide_resnet50_2(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        elif 'wide101' in args.model_name:
            model = models.wide_resnet101_2(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, args.num_classes)

        if 'resnet' in args.model_name:
            #model.fc.out_features = args.num_classes
            model.fc = nn.Linear(model.fc.in_features, args.num_classes)

        elif 'densenet' in args.model_name:
            model = models.densenet121(pretrained=True)
            model.classifier.out_features = args.num_classes


    model = model.cuda() if args.cuda else model

    if args.transfer is not None:
        model.load_state_dict(torch.load(args.transfer)['model'])


    criterion = torch.nn.CrossEntropyLoss()


    criterion = criterion.cuda() if args.cuda else criterion
    

    optimizer = Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)



    log_dir = os.path.join(args.output_dir, 'logs')
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, args.exp_name)
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    if os.path.exists(checkpoint_dir) == False:
        os.makedirs(checkpoint_dir)

    checkpoint_dir = os.path.join(checkpoint_dir, args.exp_name)
    if os.path.exists(checkpoint_dir) == False:
        os.makedirs(checkpoint_dir)
    if os.path.exists(checkpoint_dir) or os.path.exists(log_dir) :
        
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)
        shutil.rmtree(checkpoint_dir)
        os.mkdir(checkpoint_dir)

    else:
        os.mkdir(log_dir)
        os.mkdir(checkpoint_dir)

    # with open(os.path.join(checkpoint_dir, "config.json"), "w") as json_file:
    #     json.dump(args, json_file)
    # json_file.close()

    #writer = SummaryWriter(log_dir)
    #writer.add_hparams(args)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # set information
    start_time = datetime.datetime.now()
    num_batches = len(train_dataloader)
    
    #check parameter of model
    print("------------------------------------------------------------")
    total_params = sum(p.numel() for p in model.parameters())
    print("num of parameter : ",total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num of trainable_ parameter :",trainable_params)
    print("num batches :",num_batches)
    print("------------------------------------------------------------")
    
    # train
    global_iter = 0
    val_global_iter = 0
    max_val_score = 0
    for epoch in range(args.num_epochs):
        # -------------- train -------------- #
        model.train()
        epoch_loss = []
        
        for iter_, train_data in enumerate(train_dataloader):
            image, label = train_data
            image = image['image']
            label = label.long()
            t1 = time.time()

            # fetch train data
            if args.cuda:
                image = image.cuda()
                label = label.cuda()
                
            #print(label.shape)
            pred_label = model(image)
            if args.all_cols :
                before_index = 0
                loss = 0
                for i, cur_index in enumerate(accumulated_cols):

                    if args.focal_loss:
                        ce_loss = torch.nn.functional.cross_entropy(pred_label[:,before_index:cur_index], label[:,i],
                                                                    reduction='none')  # important to add reduction='none' to keep per-batch-item loss
                        pt = torch.exp(-ce_loss)
                        alpha = 1
                        gamma = 2
                        w = 1.0
                        if i == 0:
                            w = args.style_weight
                        loss += w * (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
                    else :
                        w = 1.0
                        if i == 0:
                            w = args.style_weight
                        loss += w * criterion(pred_label[:,before_index:cur_index], label[:,i])

                    before_index = cur_index
            else :

                if args.focal_loss:
                    ce_loss = torch.nn.functional.cross_entropy(pred_label, label, reduction='none')  # important to add reduction='none' to keep per-batch-item loss
                    pt = torch.exp(-ce_loss)
                    alpha = 1
                    gamma = 2
                    loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # mean over the batch
                else:
                    loss = criterion(pred_label, label)
            
            # loss backward
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            epoch_loss.append(float(loss))
            if iter_ % 100 == 0:
                print('Epoch: {} | Iteration: {} | Running loss: {:1.5f}'.format(epoch, iter_, np.mean(np.array(epoch_loss))))
            
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            t2 = time.time()

            #logging
            #writer.add_scalar('Loss/train_total', loss, global_iter)
            #writer.add_scalar('Speed/train', (t2-t1)/args.train_batch_size , global_iter)

            global_iter+=1

        # scheduler update
        scheduler.step()
       
            
        eval_score = validate(model, validate_dataloader, args)
        if epoch % 2 == 0:
            print('===================== TEST SCORE ==========================')
            test_eval_score = validate(model, test_dataloader, args)
            print('test_eval_score : ',test_eval_score)
            print('============================================================')
        #validate(args.prediction_file, model, validate_dataloader, validate_label_file, args.cuda)
        # save model
        #model_save_dir = os.path.join(checkpoint_dir, f'{(epoch + 1):03}')
        if max_val_score < eval_score :
            max_val_score = eval_score
            max_epoch = epoch
            model_save_dir = os.path.join(checkpoint_dir, "best_epoch"+str(epoch))
            save_model(model_save_dir, model, optimizer, scheduler)
        #break
    print('==================== end ============================')
    print(args.exp_name, 'Max validation ACC : ',max_val_score)
    print('==================== end ============================')

    state = torch.load(model_save_dir+'.pth')
    model.load_state_dict(state['model'])
    eval_score = validate(model, test_dataloader, args)

    print('==================== end ============================')
    print(args.exp_name, 'Max test ACC : ', eval_score, 'max_epoch', max_epoch)
    print('==================== end ============================')



if __name__ == '__main__':
    # mode argument
    tic = time.time()
    args = make_parser()
    main(args)
    toc = time.time()

    print('elapsed time : {} Sec'.format(round(toc - tic, 3)))