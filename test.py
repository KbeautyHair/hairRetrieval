import os
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import argparse
from dataloader import data_loader
from tqdm import tqdm
from datetime import datetime
import datetime as dt

from shutil import copyfile
from tqdm import tqdm
from skimage.measure import label, regionprops

from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import math
import torchvision.transforms as transforms
import torchvision.models as models

import multiprocessing
from sklearn.metrics import average_precision_score

accumulated_cols = [27, 29, 36, 38, 40, 42, 50]
cols_list = [ 'basestyle', 'basestyle_type', 'length',
               'front', 'vertical', 'sex', 'horizontal']


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def _infer(model, args, data_loader, phase='gallery'):
    res_fc = None
    res_id = None
    fc_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for index, (image, label, path) in tqdm(enumerate(data_loader)):
            if args.cuda:
                image = image['image'].cuda()
            fc = model(image)
            fc = fc.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            if index == 0:
                fc_list = fc
                label_list = label
                path_list = np.array(path)
            else:
                fc_list = np.concatenate((fc_list, fc), axis=0)
                label_list = np.concatenate((label_list, label), axis=0)
                path_list = np.concatenate((path_list, np.array(path)), axis=0)

    print(fc_list.shape, label_list.shape, path_list.shape)
    save_path = os.path.join(args.feature_dir, args.exp_name)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    np.save(os.path.join(save_path, phase + '_feature'), fc_list)
    np.save(os.path.join(save_path, phase + '_label'), label_list)
    np.save(os.path.join(save_path, phase + '_path'), path_list)
    print('save_done')
    #return [fc_list, label_list, path_list]

def feature_extract(model, validate_dataloader, args, phase):
    _infer(model, args, data_loader=validate_dataloader, phase = phase)


def get_sim():
    pass
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

    args.add_argument("--train_batch_size", type=int, default=32)
    args.add_argument("--eval_batch_size", type=int, default=32)
    args.add_argument('--num_workers', type=int, default=4)
    args.add_argument("--cuda", type=bool, default=True)

    #dataset
    # args.add_argument("--root", type=str, required = True, default='./data') # dataset direction ex) /tf/notebooks/task_11_dataset
     # dataset direction ex) /tf/notebooks/task_11_dataset
    args.add_argument("--image_size", type=int, default=256)

    #model
    args.add_argument("--exp_name", type=str, default='eb0_0102')
    args.add_argument("--model_name", type=str, default="efficientnet-b0")

    #hparams
    args.add_argument("--mode", type=str, default="test")
    args.add_argument("--gpu_id", type=str, default="0")
    args.add_argument('--focal_loss', action='store_true', default=False)
    args.add_argument('--num_classes', type=int, default = 51)

    args.add_argument('--all_cols', action='store_true', default=False)
    args.add_argument('--only_hair', action='store_true', default=False)
    args.add_argument('--ori_map', action='store_true', default=False)
    args.add_argument("--checkpoint_path", type=str, default=None) # ''
    args.add_argument('--extract_feature', action='store_true', default=False)
    args.add_argument("--feature_dir", type=str, default="")
    args.add_argument("--root_dir", type=str, default="")
    args.add_argument("--data_dir", type=str, default='data/')
    config = args.parse_args()
    config.feature_dir = config.root_dir + 'feature/'
    #config.checkpoint_path = config.root_dir + 'checkpoint/best_epoch6.pth'
    return config

def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)
        print('make path : ', path)


def plt_show(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def cos_matrix_multiplication(matrix, vector):
    """
    Calculating pairwise cosine distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors


def compute_recall_at_k(gt):
    recall_at_k = dict()
    for k in [1, 2, 4, 8]:
        recall_at_k[k] = 0


def evaluation(qi, gallery_feature, query_feature, gallery_label, query_label):
    cols_list = ['basestyle', 'basestyle_type', 'length',
                 'front', 'vertical', 'sex', 'horizontal']
    rel_list = ['basestyle', 'length', 'horizontal', 'sex']
    ap_total = [0 for i in rel_list]

    c = cos_matrix_multiplication(gallery_feature, query_feature[qi])
    refer_index_list = c.argsort()[-100:][::-1]  # top K
    cnt = 0
    for i in range(len(cols_list)):

        if cols_list[i] in rel_list:
            cur_label = []
            for j in refer_index_list:
                if gallery_label[j][i] == query_label[qi][i]:
                    cur_label.append(1)
                else:
                    cur_label.append(0)
            output = list(range(1, 101))[::-1]
            if np.sum(cur_label) == 0:
                ap = 0
            else:
                ap = average_precision_score(cur_label, output)
            ap_total[cnt] += ap
            cnt += 1
    return np.mean(ap_total)

def do_eval(args):
    data_dir = args.data_dir

    gallery_path = os.path.join(args.feature_dir, args.exp_name +'/')
    query_path = os.path.join(args.feature_dir, args.exp_name +'/')

    gallery_feature = np.load(gallery_path + 'gallery_feature.npy')
    gallery_label = np.load(gallery_path + 'gallery_label.npy')
    gallery_path = np.load(gallery_path + 'gallery_path.npy')

    query_feature = np.load(query_path + 'query_feature.npy')
    query_label = np.load(query_path + 'query_label.npy')
    query_path = np.load(query_path + 'query_path.npy')

    len_gallery_feature = len(gallery_feature)
    print('gallery images length : ', len_gallery_feature)

    # calculate mAP@100 for 6816 query images with multiprocessing
    query_index = [i for i in range(len(query_feature))]
    s = time.time()
    #with multiprocessing.Pool(processes=4) as pool:
    #    r = pool.map(evaluation, query_index)  # pool.starmap(evaluation, query_index)
    r = []
    for i in tqdm(range(len(query_index))):
        r.append(evaluation(i, gallery_feature, query_feature, gallery_label, query_label))
    print('total time : {:.4f} sec'.format(time.time() - s))
    print('Total query images length : ', len(query_index))
    print('mAP@100 \t{:.4f}'.format(np.mean(np.array(r))))
    cnt = 0
    for i in np.array(r):
        print('{} : {:.3f}'.format(cnt, i), end=', ' )
        if cnt % 10 == 0 :
            print()
        cnt +=1

def main(args):

    print("timestamp: ", datetime.now(tz=dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # For gpu
    DATASET_PATH = args.data_dir
    torch.manual_seed(args.seed)

    gallery_dataloader = data_loader(args=args, phase='gallery', batch_size=args.eval_batch_size)
    query_dataloader = data_loader(args=args, phase='query', batch_size=args.eval_batch_size)


    if args.all_cols:
        args.num_classes = 42+9
    if 'efficient' in args.model_name:
        model = MultiheadClassifier(args)
    else:
        if '18' in args.model_name:
            model = models.resnet18(pretrained=True)
        elif '34' in args.model_name:
            model = models.resnet34(pretrained=True)
        elif '50' in args.model_name:
            model = models.resnet50(pretrained=True)
        elif '101' in args.model_name:
            model = models.resnet101(pretrained=True)
        if 'resnet' in args.model_name:
            #model.fc.out_features = args.num_classes
            model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    model = model.cuda() if args.cuda else model

    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path)['model'])
    if 'resnet' in args.model_name:
        model.fc = Identity()
    start_time = time.time()

    if args.extract_feature:
        feature_extract(model, gallery_dataloader, args, phase='gallery')
        feature_extract(model, query_dataloader, args, phase ='query')

    print('==================== feature ============================')
    print(args.exp_name, 'extracted features')
    print('==================== feature ============================')
    print('extraction time : {:.4f} sec'.format(time.time() - start_time))

    start_time = time.time()
    print('=================== mAP@100 ============================')
    do_eval(args)
    print('evaluation time : {:.4f} sec'.format(time.time() - start_time))
    print("timestamp: ", datetime.now(tz=dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))
    print('======================================================')

if __name__ == '__main__':
    # mode argument
    tic = time.time()
    args = make_parser()
    main(args)
    toc = time.time()

    print('elapsed time : {} Sec'.format(round(toc - tic, 3)))