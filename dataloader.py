import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import albumentations
import albumentations.pytorch
import pandas as pd


def alb_transform(phase = 'train'):
    if phase == 'train':
        return albumentations.Compose([
            #albumentations.RandomCrop(width=400, height=400),
            albumentations.Resize(224, 224),
            #albumentations.RandomCrop(224, 224),
            albumentations.OneOf([
                albumentations.HorizontalFlip(p=1),
                albumentations.RandomRotate90(p=1),
                #albumentations.Rotate(p=1),
                albumentations.VerticalFlip(p=1)
            ], p=0.7),

            albumentations.OneOf([
                albumentations.GridDistortion(p=1),
                albumentations.OpticalDistortion(p=1)
            ], p=0.5),
            # albumentations.OneOf([

            #     albumentations.MotionBlur(p=1),
            #     albumentations.OpticalDistortion(p=1),
            #     albumentations.GaussNoise(p=1)
            # ], p=1),
            albumentations.RandomBrightnessContrast(p=0.2),
            albumentations.Normalize(),
            albumentations.pytorch.ToTensor()

        ])
    else:
        return albumentations.Compose([
            albumentations.Resize(224, 224),
            albumentations.Normalize(),
            albumentations.pytorch.ToTensor()
        ])



class CustomDataset(data.Dataset):
    def __init__(self, args, phase='train', transform=None):
        self.args = args
        self.root = args.data_dir
        self.phase = phase
        #self.labels = {}

        if phase == 'train':
            cur_df = pd.read_csv(os.path.join(self.root, 'annotation_train.csv'))
        elif phase == 'val':
            cur_df = pd.read_csv(os.path.join(self.root, 'annotation_val.csv'))
        elif phase == 'test' :
            cur_df = pd.read_csv(os.path.join(self.root, 'annotation_test.csv'))
        elif phase == 'gallery':
            cur_df = pd.read_csv(os.path.join(self.root, 'annotation_gallery.csv'))
        else:
            cur_df = pd.read_csv(os.path.join(self.root, 'annotation_query.csv'))

        self.label_list = cur_df.basestyle.tolist()
        self.image_path_list = cur_df.path.tolist()
        self.mapping_label = \
            {'가르마': 0, '남자일반숏': 1, '베이비': 2, '에어': 3, '원랭스': 4, '리프': 5,
             '소프트투블럭댄디': 6, '댄디': 7, '보니': 8, '빌드': 9, '스핀스왈로': 10,
             '쉐도우': 11, '쉼표': 12, '원블럭댄디': 13, '바디': 14, '미스티': 15,
             '여자일반숏': 16, '숏단발': 17, '시스루댄디': 18, '애즈': 19, '기타여자스타일': 20,
             '포마드': 21, '기타남자스타일': 22, '기타레이어드': 23, '보브': 24, '테슬': 25,
             '허쉬': 26, '히피': 27, '루프': 28, '플리츠': 29, '리젠트': 30, }
            # {'바디': 0, '모히칸': 1, '머쉬룸': 2, '포마드': 3, '애즈': 4, '빌드': 5,
            #                   '기타레이어드': 6, '기타스타일': 7, '베이비': 8, '스핀스왈로': 9, '크롭': 10,
            #                   '브이': 11, '댄디': 12, '히피': 13, '쉐도우': 14, '가르마': 15,
            #                   '보브': 16, '기타여자숏컷': 17, '숏단발': 18, '보니': 19, '에어': 20,
            #                   '원랭스': 21, '플리츠': 22, '기타남자숏컷': 23, '리젠트': 24, '허쉬': 25, '리프': 26}

        if args.all_cols :
            # self.basestyle = {'가르마': 0, '기타남자숏컷': 1, '기타레이어드': 2, '기타스타일': 3, '기타여자숏컷': 4, '댄디': 5, '리젠트': 6, '리프': 7,
            #                   '머쉬룸': 8, '모히칸': 9, '바디': 10, '베이비': 11, '보니': 12, '보브': 13, '브이': 14, '빌드': 15, '숏단발': 16,
            #                   '쉐도우': 17, '스핀스왈로': 18, '애즈': 19, '에어': 20, '원랭스': 21, '크롭': 22, '포마드': 23, '플리츠': 24,
            #                   '허쉬': 25, '히피': 26} # 1
            self.basestyle = { '가르마' : 0,'남자일반숏' : 1,'베이비' : 2,'에어' : 3,'원랭스' : 4,'리프' : 5,
                               '소프트투블럭댄디' : 6,'댄디' : 7,'보니' : 8,'빌드' : 9,'스핀스왈로' : 10,
                               '쉐도우' : 11,'쉼표' : 12,'원블럭댄디' : 13,'바디' : 14,'미스티' : 15,
                               '여자일반숏' : 16,'숏단발' : 17,'시스루댄디' : 18,'애즈' : 19,'기타여자스타일' : 20,
                               '포마드' : 21,'기타남자스타일' : 22,'기타레이어드' : 23,'보브' : 24,'테슬' : 25,
                               '허쉬' : 26,'히피' : 27,'루프' : 28,'플리츠' : 29,'리젠트' : 30,}
            self.basestyle_type = {'단': 0, '장': 1} # 2
            #self.length = {'0': 0, '3': 1, '남자': 2, '단발': 3, '여숏': 4, '장발': 5, '중발': 6} # 3
            self.length = { '여숏' : 0,'중발' : 1,'장발' : 2,'남자' : 3,'단발' : 4,}

            self.front = {False: 0, True: 1} # 8
            self.vertical = {'상': 0, '중': 1} # 9
            self.sex = {'남': 0, '여': 1} # 12


            # 13
            self.basestyle_labels = cur_df.basestyle.tolist()
            self.basestyle_type_labels = cur_df['basestyle-type'].tolist()
            self.length_labels = cur_df.length.tolist()
            # self.curl_labels = cur_df.curl.tolist()
            # self.bang_labels = cur_df.bang.tolist()
            # self.loss_labels = cur_df.loss.tolist()
            # self.side_labels = cur_df.side.tolist()
            self.front_labels = cur_df.front.tolist()
            self.vertical_labels = cur_df.vertical.tolist()
            # self.color_labels = cur_df.color.tolist()
            # self.partition_labels = cur_df.partition.tolist()
            self.sex_labels = cur_df.sex.tolist()
            # self.exceptional_labels = cur_df.exceptional.tolist()
            self.horizontal_labels = cur_df.horizontal.tolist()
            self.cols_len = [27, 2, 7, 9, 8, 3, 3, 2, 2, 8, 12, 2, 7]
        self.transform = alb_transform(phase = phase) #transform #get_transform()

    def __getitem__(self, index):
        # train index
        image_path = os.path.join(self.root, self.image_path_list[index])
        img = np.array(Image.open(image_path))
        if self.args.ori_map:
            tt = self.image_path_list[index].split('/')
            ori_map_path = os.path.join(self.root, tt[0], 'mask_ori', tt[2])
            ori_map = np.array(Image.open(ori_map_path))
            img[:,:,-1] = ori_map # += or all channel
        img = self.transform(image=img)


        label = self.mapping_label[self.label_list[index]]
        if self.args.all_cols:
            label = self.make_label(index)

        label = torch.tensor(label)
        # -------------------------------
        return img, label, image_path
        # ------------------------------

    def getdata(self, index):

        # train index
        image_path = os.path.join(self.root, self.image_path_list[index])
        img = np.array(Image.open(image_path))
        if self.args.ori_map:
            tt = self.image_path_list[index].split('/')
            ori_map_path = os.path.join(self.root, tt[0], 'mask_ori', tt[2])
            ori_map = np.array(Image.open(ori_map_path))
            img[:,:,-1] = ori_map # += or all channel
        img = self.transform(image=img)


        label = self.mapping_label[self.label_list[index]]
        if self.args.all_cols:
            label = self.make_label(index)

        label = torch.tensor(label)
        # -------------------------------
        return img, label, image_path
        # -------------------------------


    def __len__(self):
        return len(self.image_path_list)

    def make_label(self, index):
        label = []
        label.append(self.basestyle[self.basestyle_labels[index]])
        label.append(self.basestyle_type[self.basestyle_type_labels[index]])
        label.append(self.length[self.length_labels[index]])
        # label.append(self.curl[self.curl_labels[index]])
        # label.append(self.bang[self.bang_labels[index]])
        # label.append(self.loss[self.loss_labels[index]])
        # label.append(self.side[self.side_labels[index]])
        label.append(self.front[self.front_labels[index]])
        label.append(self.vertical[self.vertical_labels[index]])
        # label.append(self.color[self.color_labels[index]])
        # label.append(self.partition[self.partition_labels[index]])
        label.append(self.sex[self.sex_labels[index]])


        label.append(self.make_horizontal(self.horizontal_labels[index]))
        # label.append(self.exceptional[self.exceptional_labels[index]])
        return label
    def make_horizontal(self, horizontal):

        #cks = [45, 90, 135, 180, 225, 270, 315, 360]
        cks = [90, 180, 270, 360]
        for i in range(len(cks)):
            if horizontal <= cks[i]:
                return i
        return i+1

def data_loader(args, phase='train', batch_size=16, sampler=None):
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False



    dataset = CustomDataset(args, phase)

    if phase == 'train' and args.sampler and args.all_cols == False:
        n_classes = len(dataset.mapping_label)
        # sampler updated
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers = args.num_workers)
    return dataloader


