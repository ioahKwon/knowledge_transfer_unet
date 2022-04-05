## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

##
## Parser 생성하기
parser = argparse.ArgumentParser(description='Train the UNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_dir", default="./data/BSR/BSDS500/data/images", type=str, dest="data_dir")
parser.add_argument("--num_train", default=0, type=int, dest="num_train")
parser.add_argument("--num_val", default=0, type=int, dest="num_val")
parser.add_argument("--num_test", default=0, type=int, dest="num_test")
parser.add_argument("--new_data_dir", default="./data/BSR/BSDS500/data/images", type=str, dest="new_data_dir")

## 트레이닝 파라메터 설정하기

data_dir = args.data_dir
num_train = args.num_train
num_val = args.num_val
num_test = args.num_test
new_data_dir = args.new_data_dir

print("data dir: %s" % data_dir)
print("num_train: %s" % num_train)
print("num_val: %s" % num_val)
print("num_test: %s" % num_test)
print("new data dir: %s" % new_data_dir)

## Train set 만들기

lst_data = os.listdir(data_dir)
lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]

lst_label.sort()
lst_input.sort()

n_offset = 0

dir_save_train = os.path.join(new_data_dir, 'train')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

for i in range(num_train):
    name_label = lst_label[i+n_offset]
    name_input = lst_input[i+n_offset]

    img_label = Image.open(os.path.join(data_dir, name_label)).convert('L')
    img_input = Image.open(os.path.join(data_dir, name_input)).convert('L')

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, lst_label[i+n_offset]), label_)
    np.save(os.path.join(dir_save_train, lst_input[i+n_offset]), input_)

## Val set 만들기
n_offset += num_train

dir_save_val = os.path.join(new_data_dir, 'val')

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

for i in range(num_val):
    name_label = lst_label[i+n_offset]
    name_input = lst_input[i+n_offset]

    img_label = Image.open(os.path.join(data_dir, name_label)).convert('L')
    img_input = Image.open(os.path.join(data_dir, name_input)).convert('L')

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, lst_label[i+n_offset]), label_)
    np.save(os.path.join(dir_save_val, lst_input[i+n_offset]), input_)

## Test set 만들기
n_offset += num_val

dir_save_test = os.path.join(new_data_dir, 'test')

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

for i in range(num_test):
    name_label = lst_label[i+n_offset]
    name_input = lst_input[i+n_offset]

    img_label = Image.open(os.path.join(data_dir, name_label)).convert('L')
    img_input = Image.open(os.path.join(data_dir, name_input)).convert('L')

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, lst_label[i+n_offset]), label_)
    np.save(os.path.join(dir_save_test, lst_input[i+n_offset]), input_)

