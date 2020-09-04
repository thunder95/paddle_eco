import os
import numpy as np
import cv2
import sys
import glob
import pickle
from multiprocessing import Pool


# label_dic = np.load('ucf101_label_dir.npy', allow_pickle=True).item()
# print(label_dic)

#标签字典
label_dict = {}
for line in open("/home/aistudio/work/classInd.txt"):   
    ld = line.split(" ")
    label_dict[ld[1].strip()] = ld[0]

#获取train和test的列表
test_list = []
for line in open("/home/aistudio/work/testlist01.txt"):   
    tl = line.split("/")
    test_list.append(tl[-1].replace(".avi", "").strip())

train_list = []
for line in open("/home/aistudio/work/trainlist01.txt"):   
    ld = line.split(" ")
    tl = ld[0].split("/")
    train_list.append(tl[-1].replace(".avi", "").strip())

# print(test_list[:5])
# print(train_list[:5])
# exit()

source_dir = '/home/aistudio/work/UCF-101-jpg/'
target_train_dir = source_dir + 'train01'
target_test_dir = source_dir + 'test01'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)

for key in label_dict:
    each_mulu = key
    print(each_mulu, key) 

    label_dir = os.path.join(source_dir, each_mulu) #一级类别 hair_cut 
    label_mulu = os.listdir(label_dir)
    tag = 1

    cur_len = len(label_mulu)
    split_len = int(cur_len*0.1)
    for each_label_mulu in label_mulu:
        # print("each_label_mulu===> ", each_label_mulu)
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu)) #二级 work/UCF-101-jpg/Haircut/v_Haircut_g16_c01
        image_file.sort()

        image_num = len(image_file)
        frame = []
        vid = each_label_mulu
        for i in range(image_num):
            image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_file[i])
            frame.append(image_path)

        output_pkl = vid + '.pkl'
        if vid in train_list:
            output_pkl = os.path.join(target_train_dir, output_pkl)
        elif vid in test_list:
            output_pkl = os.path.join(target_test_dir, output_pkl)
        else:
            print("erro===>", vid)
            continue
        tag += 1
        f = open(output_pkl, 'wb')
        pickle.dump((vid, label_dict[key], frame), f, -1)
        f.close()
