import timm
import os 
import robustness
import numpy as np
import torch.nn.functional as F
import torch as ch
import math
from robustness.attacker import AttackerModel
from verification import verify
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm
from openpyxl import load_workbook, Workbook
import pandas as pd
import time
ds = robustness.datasets.DATASETS['imagenet']('/home/tianle/datasets/ImageNet2012')
_, test_loader = ds.make_loaders(workers=8,
                                    batch_size=1,
                                    test_subset = True,
                                    test_subset_size = 100)

# for model_name in ['vit_base_patch16_224']:
for model_name in ['resnetv2_50', 'vit_base_patch16_224', 'mobilenetv2_100', 'efficientnet_b0']:

    print(model_name)


    m = timm.create_model(model_name, pretrained=True)
    model = AttackerModel(m, dataset=ds)
    model = model.cuda()
    model.eval()
    n_test = len(test_loader)

    # result_ada = []
    # time_ada = []
    # res_ada = np.zeros((1,4))
    # result_ho = []
    # time_ho = []
    # res_ho = np.zeros((1,4))
    result_ac = []
    time_ac = []
    res_ac = np.zeros((1,4))

    num_hold = 0
    num_violation = 0
    num_None = 0
    total_num = 0

    iterator = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, (im, label) in iterator:
        im, label = im.cuda(), label.cuda()
        start = time.time()
        result, num = verify(im, label, model, type="AC", tau=0.05, delta=1e-10, sample_limit=5000, bs=1000, kwargs={
            'rot': 30,
            'trans': 0, 
            'scale': 0,
            'hue': 0,
            'satu': 0,
            'bright': 0,
            'cont': 0,
            'gau_size': 11, 
            'gau_sigma': 9,
            'tries':1,
            'use_best': True, 
            'transform_type': 'spatial',
            'attack_type': 'random',
            'do_tqdm': False
            })
        end = time.time()
        time_ac.append(end-start)
        total_num += num
        if result == 0:
            num_violation += 1
            result_ac.append(0)
        elif result == 1:
            num_hold += 1
            result_ac.append(1)
        else:
            num_None += 1
            result_ac.append(2)
    # print(model_name+"_ada:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
    res_ac = np.r_[res_ac, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]

    
    num_hold = 0
    num_violation = 0
    num_None = 0
    total_num = 0

    iterator = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, (im, label) in iterator:
        im, label = im.cuda(), label.cuda()
        start = time.time()
        result, num = verify(im, label, model, type="AC", tau=0.05, delta=1e-10, sample_limit=5000, bs=1000, kwargs={
            'rot': 0,
            'trans': 0.3, 
            'scale': 0,
            'hue': 0,
            'satu': 0,
            'bright': 0,
            'cont': 0,
            'gau_size': 11, 
            'gau_sigma': 9,
            'tries':1,
            'use_best': True, 
            'transform_type': 'spatial',
            'attack_type': 'random',
            'do_tqdm': False
            })
        end = time.time()
        time_ac.append(end-start)
        total_num += num
        if result == 0:
            num_violation += 1
            result_ac.append(0)
        elif result == 1:
            num_hold += 1
            result_ac.append(1)
        else:
            num_None += 1
            result_ac.append(2)
    # print(model_name+"_ada:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
    res_ac = np.r_[res_ac, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]



    num_hold = 0
    num_violation = 0
    num_None = 0
    total_num = 0

    iterator = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, (im, label) in iterator:
        im, label = im.cuda(), label.cuda()
        start = time.time()
        result, num = verify(im, label, model, type="AC", tau=0.05, delta=1e-10, sample_limit=5000, bs=1000, kwargs={
            'rot': 0,
            'trans': 0, 
            'scale': 0.3,
            'hue': 0,
            'satu': 0,
            'bright': 0,
            'cont': 0,
            'gau_size': 11, 
            'gau_sigma': 9,
            'tries':1,
            'use_best': True, 
            'transform_type': 'spatial',
            'attack_type': 'random',
            'do_tqdm': False
            })
        end = time.time()
        time_ac.append(end-start)
        total_num += num
        if result == 0:
            num_violation += 1
            result_ac.append(0)
        elif result == 1:
            num_hold += 1
            result_ac.append(1)
        else:
            num_None += 1
            result_ac.append(2)
    # print(model_name+"_ada:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
    res_ac = np.r_[res_ac, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]



    num_hold = 0
    num_violation = 0
    num_None = 0
    total_num = 0

    iterator = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, (im, label) in iterator:
        im, label = im.cuda(), label.cuda()
        start = time.time()
        result, num = verify(im, label, model, type="AC", tau=0.05, delta=1e-10, sample_limit=5000, bs=1000, kwargs={
            'rot': 0,
            'trans': 0, 
            'scale': 0,
            'hue': math.pi/3,
            'satu': 0,
            'bright': 0,
            'cont': 0,
            'gau_size': 11, 
            'gau_sigma': 9,
            'tries':1,
            'use_best': True, 
            'transform_type': 'color',
            'attack_type': 'random',
            'do_tqdm': False
            })
        end = time.time()
        time_ac.append(end-start)
        total_num += num
        if result == 0:
            num_violation += 1
            result_ac.append(0)
        elif result == 1:
            num_hold += 1
            result_ac.append(1)
        else:
            num_None += 1
            result_ac.append(2)
    # print(model_name+"_ada:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
    res_ac = np.r_[res_ac, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]


    num_hold = 0
    num_violation = 0
    num_None = 0
    total_num = 0

    iterator = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, (im, label) in iterator:
        im, label = im.cuda(), label.cuda()
        start = time.time()
        result, num = verify(im, label, model, type="AC", tau=0.05, delta=1e-10, sample_limit=5000, bs=1000, kwargs={
            'rot': 0,
            'trans': 0, 
            'scale': 0,
            'hue': 0,
            'satu': 0.5,
            'bright': 0,
            'cont': 0,
            'gau_size': 11, 
            'gau_sigma': 9,
            'tries':1,
            'use_best': True, 
            'transform_type': 'color',
            'attack_type': 'random',
            'do_tqdm': False
            })
        end = time.time()
        time_ac.append(end-start)
        total_num += num
        if result == 0:
            num_violation += 1
            result_ac.append(0)
        elif result == 1:
            num_hold += 1
            result_ac.append(1)
        else:
            num_None += 1
            result_ac.append(2)
    # print(model_name+"_ada:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
    res_ac = np.r_[res_ac, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]


    num_hold = 0
    num_violation = 0
    num_None = 0
    total_num = 0

    iterator = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, (im, label) in iterator:
        im, label = im.cuda(), label.cuda()
        start = time.time()
        result, num = verify(im, label, model, type="AC", tau=0.05, delta=1e-10, sample_limit=5000, bs=1000, kwargs={
            'rot': 0,
            'trans': 0, 
            'scale': 0,
            'hue': 0,
            'satu': 0,
            'bright': 0.3,
            'cont': 0.3,
            'gau_size': 11, 
            'gau_sigma': 9,
            'tries':1,
            'use_best': True, 
            'transform_type': 'color',
            'attack_type': 'random',
            'do_tqdm': False
            })
        end = time.time()
        time_ac.append(end-start)
        total_num += num
        if result == 0:
            num_violation += 1
            result_ac.append(0)
        elif result == 1:
            num_hold += 1
            result_ac.append(1)
        else:
            num_None += 1
            result_ac.append(2)
    # print(model_name+"_ada:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
    res_ac = np.r_[res_ac, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]


    num_hold = 0
    num_violation = 0
    num_None = 0
    total_num = 0

    iterator = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, (im, label) in iterator:
        im, label = im.cuda(), label.cuda()
        start = time.time()
        result, num = verify(im, label, model, type="AC", tau=0.05, delta=1e-10, sample_limit=5000, bs=1000, kwargs={
            'rot': 0,
            'trans': 0, 
            'scale': 0,
            'hue': 0,
            'satu': 0,
            'bright': 0,
            'cont': 0,
            'gau_size': 11, 
            'gau_sigma': 9,
            'tries':1,
            'use_best': True, 
            'transform_type': 'blur',
            'attack_type': 'random',
            'do_tqdm': False
            })
        end = time.time()
        time_ac.append(end-start)
        total_num += num
        if result == 0:
            num_violation += 1
            result_ac.append(0)
        elif result == 1:
            num_hold += 1
            result_ac.append(1)
        else:
            num_None += 1
            result_ac.append(2)
    # print(model_name+"_ada:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
    res_ac = np.r_[res_ac, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]

    # df_ada = pd.DataFrame(res_ada)
    # df_ho = pd.DataFrame(res_ho)
    df_ac = pd.DataFrame(res_ac)

    writer = pd.ExcelWriter("./imagenet_result.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace')

    book = load_workbook("./imagenet_result.xlsx")
    writer.book = book
    # df_ada.to_excel(excel_writer=writer, sheet_name=model_name+'ada')
    # df_ho.to_excel(excel_writer=writer, sheet_name=model_name+'ho')
    df_ac.to_excel(excel_writer=writer, sheet_name=model_name+'ac')
    # save file
    writer.save()
    # close writer
    writer.close()

    # np.savez('./'+model_name,time_ac=np.array(time_ac),time_ho=np.array(time_ho), result_ac=np.array(result_ac),result_ho=np.array(result_ho)) 
    np.savez('./'+model_name+'_ac',time_ac=np.array(time_ac), result_ac=np.array(result_ac)) 