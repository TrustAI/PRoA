import os 
import numpy as np
import torch.nn.functional as F
import torch as ch
import time
import math
from robustness.datasets import CIFAR
from verification import verify
from robustness.model_utils import make_and_restore_model


# ds = CIFAR('./path/to/cifar', std = ch.tensor([0.2471, 0.2435, 0.2616]))
ds = CIFAR('./path/to/cifar')
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

from openpyxl import load_workbook, Workbook
import pandas as pd

_, test_loader = ds.make_loaders(workers=2,
                                    batch_size=1,
                                    test_subset = True,
                                    test_subset_size = 500
                                    )
n_test = len(test_loader)

for arch in ['clean.pt.best', 'pat.ckpt.pth', 'pgd_l2.pt.best']:
# for arch in [ 'pgd_l2.pt.best']:
    loca = './models/'+arch

    print(arch)
    
    if arch == 'pat.ckpt.pth':
        model, _ = make_and_restore_model(arch='resnet18',
            dataset=ds)
        model.model.load_state_dict(ch.load(loca))
    else:
        model, _ = make_and_restore_model(arch='resnet18',
            dataset=ds, resume_path=loca, parallel=False)

    model = model.cuda()

    model.eval()  
    # result_ada = []
    # time_ada = []
    # res_ada = np.ones((1,4))
    result_ho = []
    time_ho = []      
    res_ho = np.zeros((1,4))

    for confidence in [1e-4, 1e-15, 1e-30]:

        res_ho = np.r_[res_ho, np.ones((1,4))*confidence]
        num_hold = 0
        num_violation = 0
        num_None = 0
        total_num = 0

        iterator = tqdm(enumerate(test_loader), total=len(test_loader))

        for i, (im, label) in iterator:
            im, label = im.cuda(), label.cuda()
            start = time.time()
            result, num = verify(im, label, model, type="hoeffding", tau=0.05, delta=confidence, sample_limit=50000, bs=1000, kwargs={
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
            # time_ho.append(end-start)
            total_num += num
            if result == 0:
                num_violation += 1
                result_ho.append(0)
            elif result == 1:
                num_hold += 1
                result_ho.append(1)
            else:
                num_None += 1
                result_ho.append(3)
        # print(model_name+"_Hoef:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
        res_ho = np.r_[res_ho, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]


        num_hold = 0
        num_violation = 0
        num_None = 0
        total_num = 0

        iterator = tqdm(enumerate(test_loader), total=len(test_loader))

        for i, (im, label) in iterator:
            im, label = im.cuda(), label.cuda()
            start = time.time()
            result, num = verify(im, label, model, type="hoeffding", tau=0.05, delta=confidence, sample_limit=50000, bs=1000, kwargs={
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
            # time_ho.append(end-start)
            total_num += num
            if result == 0:
                num_violation += 1
                result_ho.append(0)
            elif result == 1:
                num_hold += 1
                result_ho.append(1)
            else:
                num_None += 1
                result_ho.append(3)
        # print(model_name+"_Hoef:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
        res_ho = np.r_[res_ho, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]

        num_hold = 0
        num_violation = 0
        num_None = 0
        total_num = 0

        iterator = tqdm(enumerate(test_loader), total=len(test_loader))

        for i, (im, label) in iterator:
            im, label = im.cuda(), label.cuda()
            start = time.time()
            result, num = verify(im, label, model, type="hoeffding", tau=0.05, delta=confidence, sample_limit=50000, bs=1000, kwargs={
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
            # time_ho.append(end-start)
            total_num += num
            if result == 0:
                num_violation += 1
                result_ho.append(0)
            elif result == 1:
                num_hold += 1
                result_ho.append(1)
            else:
                num_None += 1
                result_ho.append(3)
        # print(model_name+"_Hoef:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
        res_ho = np.r_[res_ho, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]


        num_hold = 0
        num_violation = 0
        num_None = 0
        total_num = 0

        iterator = tqdm(enumerate(test_loader), total=len(test_loader))

        for i, (im, label) in iterator:
            im, label = im.cuda(), label.cuda()
            start = time.time()
            result, num = verify(im, label, model, type="hoeffding", tau=0.05, delta=confidence, sample_limit=50000, bs=1000, kwargs={
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
            # time_ho.append(end-start)
            total_num += num
            if result == 0:
                num_violation += 1
                result_ho.append(0)
            elif result == 1:
                num_hold += 1
                result_ho.append(1)
            else:
                num_None += 1
                result_ho.append(3)
        # print(model_name+"_Hoef:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
        res_ho = np.r_[res_ho, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]


        num_hold = 0
        num_violation = 0
        num_None = 0
        total_num = 0

        iterator = tqdm(enumerate(test_loader), total=len(test_loader))

        for i, (im, label) in iterator:
            im, label = im.cuda(), label.cuda()
            start = time.time()
            result, num = verify(im, label, model, type="hoeffding", tau=0.05, delta=confidence, sample_limit=50000, bs=1000, kwargs={
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
            # time_ho.append(end-start)
            total_num += num
            if result == 0:
                num_violation += 1
                result_ho.append(0)
            elif result == 1:
                num_hold += 1
                result_ho.append(1)
            else:
                num_None += 1
                result_ho.append(3)
        # print(model_name+"_Hoef:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
        res_ho = np.r_[res_ho, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]


        num_hold = 0
        num_violation = 0
        num_None = 0
        total_num = 0

        iterator = tqdm(enumerate(test_loader), total=len(test_loader))

        for i, (im, label) in iterator:
            im, label = im.cuda(), label.cuda()
            start = time.time()
            result, num = verify(im, label, model, type="hoeffding", tau=0.05, delta=confidence, sample_limit=50000, bs=1000, kwargs={
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
            # time_ho.append(end-start)
            total_num += num
            if result == 0:
                num_violation += 1
                result_ho.append(0)
            elif result == 1:
                num_hold += 1
                result_ho.append(1)
            else:
                num_None += 1
                result_ho.append(3)
        # print(model_name+"_Hoef:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
        res_ho = np.r_[res_ho, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]


        num_hold = 0
        num_violation = 0
        num_None = 0
        total_num = 0

        iterator = tqdm(enumerate(test_loader), total=len(test_loader))

        for i, (im, label) in iterator:
            im, label = im.cuda(), label.cuda()
            start = time.time()
            result, num = verify(im, label, model, type="hoeffding", tau=0.05, delta=confidence, sample_limit=50000, bs=1000, kwargs={
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
            # time_ho.append(end-start)
            total_num += num
            if result == 0:
                num_violation += 1
                result_ho.append(0)
            elif result == 1:
                num_hold += 1
                result_ho.append(1)
            else:
                num_None += 1
                result_ho.append(3)
        # print(model_name+"_Hoef:", num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test)
        res_ho = np.r_[res_ho, np.array([[num_hold/n_test, num_violation/n_test, num_None/n_test, total_num/n_test]])]

    df_ho = pd.DataFrame(res_ho)

    writer = pd.ExcelWriter("./cifar_result.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace')

    book = load_workbook("./cifar_result.xlsx")
    writer.book = book
    df_ho.to_excel(excel_writer=writer, sheet_name=arch+'ho'+'confi')

    # save file
    writer.save()
    # close writer
    writer.close()

        # np.savez('./'+arch,time_ada=np.array(time_ada),time_ho=np.array(time_ho), result_ada=np.array(result_ada),result_ho=np.array(result_ho)) 