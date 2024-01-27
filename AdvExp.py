import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from dataloader import Chexpert
from torch.utils.data import DataLoader
import cv2
import time
from sklearn import metrics
from torchmetrics import HammingDistance
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from utils import fairness_metrics, saveImage
from groupDRO import compute_group_avg, compute_robust_loss

def train(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Device setting
    num_workers = args["device"]["num_workers"]
    pin_memory = args["device"]["pin_memory"]

    # Hyperparameters
    pattern_name = args["train"]["pattern_name"]
    step_size = args["train"]["step_size"]
    opt_lr = args["train"]["opt_lr"]
    opt_spurious_lr = args["train"]["opt_spurious_lr"]
    opt_target_lr = args["train"]["opt_target_lr"]
    weight_decay = args["train"]["weight_decay"]
    MI_para = args["train"]["MI_para"]
    target_para = args["train"]["target_para"]
    spurious_para = args["train"]["spurious_para"]
    rev_para = args["train"]["rev_para"]
    epochs = args["train"]["epochs"]
    batch_size = args["train"]["batch_size"]
    save_interval = args["train"]["save_interval"]
    target_classes = args["train"]["target_classes"]
    train_path = args["train"]["train_path"]
    test_path = args["train"]["test_path"]

    # Training metadata
    weight_dir = args["meta"]["weight_dir"]
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    save_image = args["meta"]["saveAllimage"]
    if save_image == True:
        masked_img_dir = args["meta"]["save_image_path"]
        if not os.path.exists(masked_img_dir):
            os.makedirs(masked_img_dir)
    spurious_type = args["meta"]["spurious_type"]
    
    pattern_index_dict = {
        'cardiomegaly': 2,
        'support devices': 13,
        'pneumonia': 7,
        'pleural effusion': 10
    }
    pattern_index = pattern_index_dict[pattern_name]

    Autoencoder = models.Combined_to_oneNet(spurious_weight=False, num_verb=target_classes, spurious_sigmoid=True, grad_reverse=False).to(device)
    spurious_classify = models.spurious_classifier(oc=2, indpt_grad_reverse=True).to(device)

    train_xray = Chexpert(train_path)
    test_xray = Chexpert(test_path)
    opt = optim.Adam(list(Autoencoder.autoencoder.parameters()), lr=opt_lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    opt_spurious = optim.Adam(list(spurious_classify.parameters()), lr=opt_spurious_lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    opt_target = optim.Adam(list(Autoencoder.base_network.parameters()) + list(Autoencoder.finalLayer.parameters()), lr=opt_target_lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    train_loader = DataLoader(train_xray, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    test_loader = DataLoader(test_xray, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    BCE = nn.BCELoss(reduction="none")
    group_weights = torch.ones(1).to(device) / 4

    print("Start Training...")
    
    auc_save_list = torch.zeros(150)

    for epoch in range(0, epochs):
        Autoencoder.train()
        spurious_classify.train()
        running_loss = 0.0
        running_loss_spurious = 0.0
        running_loss_target = 0.0
        running_loss_MI = 0.0
        
        start_time = time.time()
        for i, data in enumerate(train_loader, 0):
            Autoencoder.zero_grad()
            spurious_classify.zero_grad()
            opt.zero_grad()
            opt_spurious.zero_grad()
            opt_target.zero_grad()

            image = data["full_img"].to(device)
            diagnosis = data['diagnosis'].to(device)
            pattern = data['diagnosis'][:, pattern_index].to(device)
            pattern = torch.unsqueeze(pattern, 1)
            spurious = data[spurious_type].to(device)
            _, target_spurious = torch.max(spurious, 1)
            
            spurious_pattern = torch.zeros(spurious.size(0)).to(device)
            for j in range(spurious.size(0)):
                if target_spurious[j] == 0 and pattern[j] == 1:
                    spurious_pattern[j] = 0
                elif target_spurious[j] == 0 and pattern[j] == 0:
                    spurious_pattern[j] = 1
                elif target_spurious[j] == 1 and pattern[j] == 1:
                    spurious_pattern[j] = 2
                else:
                    spurious_pattern[j] = 3

            task_pred, autoencoded_images, mask = Autoencoder(image)
            _, adv_pred = spurious_classify(autoencoded_images)
            
            loss_target = BCE(task_pred, pattern)*target_para
            group_loss, group_count = compute_group_avg(loss_target, spurious_pattern, 4)
            robust_loss_target, new_weights = compute_robust_loss(group_loss, group_count, group_weights, step_size)
            group_weights = new_weights

            loss_MI = torch.mean(torch.sum(adv_pred*torch.log(adv_pred + 1e-6), axis=1))*MI_para
            loss = robust_loss_target + loss_MI
            
            loss.backward()
            opt.step()
            opt_target.step()

            running_loss += loss.detach().item()
            running_loss_target += torch.mean(robust_loss_target).detach().item()
            running_loss_MI += loss_MI.detach().item()
            
            opt.zero_grad()
            opt_spurious.zero_grad()
            opt_target.zero_grad()

            task_pred, autoencoded_images, mask = Autoencoder(image)
            autoencoded_images = models.ReverseLayerF.apply(autoencoded_images, rev_para)
            _, rev_adv_pred = spurious_classify(autoencoded_images)
            _, predict_spurious = torch.max(rev_adv_pred, 1)
            

            loss_spurious = criterion(rev_adv_pred, spurious)*spurious_para
            loss_spurious.backward()
            opt.step()
            opt_spurious.step()

            running_loss += loss_spurious.detach().item()
            running_loss_spurious += loss_spurious.detach().item()

            if i % 100 == 99 or i+1 == len(train_loader):
                print(f"[{epoch+1} epoch: {i+1}/{len(train_loader)}] finished")
                
        print('[%d/%d] loss: %.3f' % (epoch+1, epochs, running_loss / len(train_loader)))
        print("Time elapsed: %.1f"%(time.time()- start_time))
        
        validation_result = validate_fun(Autoencoder, spurious_classify, test_loader, 0, epoch, criterion, BCE, pattern_index, masked_img_dir, spurious_type)
        print(
            "spurious_correct: ", validation_result['spurious_correct'],
            "target_correct: ", validation_result['target_correct'],
            "test_auc: ", validation_result['test_auc'],
            "running_test_loss_spurious: ", validation_result['running_test_loss_spurious'],
            "running_test_loss_target: ", validation_result['running_test_loss_target'], 
            "running_test_loss_MI: ", validation_result['running_test_loss_MI'], 
            "PQD: ", validation_result['PQD'], 
            "DP: ", validation_result['DP'], 
            "EOM:", validation_result['EOM'], 
            "male/black_P_acc:", validation_result['male_P_acc'], 
            "male/black_N_acc:", validation_result['male_N_acc'], 
            "female/white_P_acc:", validation_result['female_P_acc'], 
            "female/white_N_acc:", validation_result['female_N_acc'], 
            "worst_G_acc:", validation_result['worst_G_acc']
            )

        test_auc = validation_result['test_auc']
        
        if epoch % save_interval == save_interval - 1 or test_auc > torch.max(auc_save_list):
            torch.save({
                'autoencoder_state_dict': Autoencoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                }, f"{weight_dir}autoencoder_epoch_{epoch+1}.pt")               
            torch.save({
                'model_state_dict': spurious_classify.state_dict(),
                'optimizer_state_dict': opt_spurious.state_dict(),
                'optimizer_target_state_dict': opt_target.state_dict(),
                }, f"{weight_dir}spurious_classifier_epoch_{epoch+1}.pt")

        auc_save_list[epoch] = test_auc

            
    print('Finished Training')

def val(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = args["device"]["num_workers"]
    pin_memory = args["device"]["pin_memory"]
    pattern_name = args["train"]["pattern_name"]
    pattern_name = args["train"]["pattern_name"]
    batch_size = args["train"]["batch_size"]
    target_classes = args["train"]["target_classes"]
    test_path = args["train"]["test_data_path"]
    pretrain_mask_weight = args["meta"]["pretrain_mask_weight"]
    pretrain_spurious_weight = args['meta']["pretrain_spurious_weight"]
    save_image = args["meta"]["saveAllimage"]
    masked_img_dir = args["meta"]["save_image_path"]
    spurious_type = args["meta"]["exp_type"]
    MI_para = args["train"]["MI_para"]
    target_para = args["train"]["target_para"]
    spurious_para = args["train"]["spurious_para"]

    pattern_index_dict = {
        'cardiomegaly': 2,
        'support devices': 13,
        'pneumonia': 7,
        'pleural effusion': 10
    }
    pattern_index = pattern_index_dict[pattern_name]

    print(f'pretrain_mask_weight: {pretrain_mask_weight}')
    print(f'pretrain_spurious_weight: {pretrain_spurious_weight}')

    print(f'target_classes: {target_classes}')
    print(f'test_path: {test_path}')

    print(f'pattern_index: {pattern_index}')

    Autoencoder = models.Combined_to_oneNet(spurious_weight=False, num_verb=target_classes, spurious_sigmoid=True, grad_reverse=False).to(device)
    spurious_classify = models.spurious_classifier(oc=2, indpt_grad_reverse=True).to(device)
    Autoencoder.load_state_dict(torch.load(pretrain_mask_weight)["autoencoder_state_dict"])
    spurious_classify.load_state_dict(torch.load(pretrain_spurious_weight)["model_state_dict"])
    test_xray = Chexpert(test_path)
    test_loader = DataLoader(test_xray, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    BCE = nn.BCELoss(reduction="none")

    testing_result = validate_fun(Autoencoder, spurious_classify, test_loader, save_image, 0, criterion, BCE, pattern_index, masked_img_dir, spurious_type, MI_para, target_para, spurious_para)
    
    print(
        "spurious_correct: ", testing_result['spurious_correct'],
        "target_correct: ", testing_result['target_correct'],
        "test_auc: ", testing_result['test_auc'],
        "running_test_loss_spurious: ", testing_result['running_test_loss_spurious'],
        "running_test_loss_target: ", testing_result['running_test_loss_target'],
        "running_test_loss_MI: ", testing_result['running_test_loss_MI'],
        "PQD: ", testing_result['PQD'],
        "DP: ", testing_result['DP'],
        "EOM:", testing_result['EOM'],
        "male/black_P_acc:", testing_result['male_P_acc'],
        "male/black_N_acc:", testing_result['male_N_acc'],
        "female/white_P_acc:", testing_result['female_P_acc'],
        "female/white_N_acc:", testing_result['female_N_acc'],
        "worst_G_acc:", testing_result['worst_G_acc'],
    )
    print('Finished Validating')


def validate_fun(Autoencoder, spurious_classify, test_loader, saveallImage, epoch, criterion, BCE, pattern_index, masked_img_dir, spurious_type, spurious_para, target_para, MI_para):
    print('Start validating')
    Autoencoder.eval()
    spurious_classify.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        total = 0
        correct = 0
        spurious_correct = 0
        target_correct = 0
        male_TP = 0
        male_TN = 0
        male_FP = 0
        male_FN = 0

        female_TP = 0
        female_TN = 0
        female_FP = 0
        female_FN = 0

        running_test_loss_spurious = 0.0
        running_test_loss_target = 0.0
        running_test_loss_MI = 0.0

        record_target_pattern = torch.zeros(1).to(device)
        record_predict_pattern = torch.zeros(1).to(device)
        record_male_target_pattern = torch.zeros(1).to(device)
        record_male_predict_pattern = torch.zeros(1).to(device)
        record_female_target_pattern = torch.zeros(1).to(device)
        record_female_predict_pattern = torch.zeros(1).to(device)

        for i, data in enumerate(test_loader, 0):
            test_image = data["full_img"].to(device)
            test_diagnosis = data['diagnosis'].to(device)
            test_pattern = data['diagnosis'][:, pattern_index].to(device)
            test_pattern = torch.unsqueeze(test_pattern, 1)
            test_spurious = data[spurious_type].to(device)
            patient_num = data['patient_num']

            test_task_pred, draw_image, mask = Autoencoder(test_image)
            _, test_adv_pred  = spurious_classify(draw_image)
            test_task_pred_round = torch.round(test_task_pred)

            _, predict_spurious = torch.max(test_adv_pred, 1)
            _, target_spurious = torch.max(test_spurious, 1)

            target_correct += (test_task_pred_round==test_pattern).sum()
            spurious_correct += (predict_spurious==target_spurious).sum()

            subtotal = test_spurious.size(0)
            total += subtotal

            for j in range(test_spurious.size(0)):
                if target_spurious[j] == 0 and test_task_pred_round[j] == 0 and test_pattern[j] == 0:
                    male_TN += 1
                elif target_spurious[j] == 0 and test_task_pred_round[j] == 0 and test_pattern[j] == 1:
                    male_FN += 1
                elif target_spurious[j] == 0 and test_task_pred_round[j] == 1 and test_pattern[j] == 0:
                    male_FP += 1
                elif target_spurious[j] == 0 and test_task_pred_round[j] == 1 and test_pattern[j] == 1:
                    male_TP += 1
                if target_spurious[j] == 1 and test_task_pred_round[j] == 0 and test_pattern[j] == 0:
                    female_TN += 1
                elif target_spurious[j] == 1 and test_task_pred_round[j] == 0 and test_pattern[j] == 1:
                    female_FN += 1
                elif target_spurious[j] == 1 and test_task_pred_round[j] == 1 and test_pattern[j] == 0:
                    female_FP += 1
                elif target_spurious[j] == 1 and test_task_pred_round[j] == 1 and test_pattern[j] == 1:
                    female_TP += 1
            
            test_loss_spurious = criterion(test_adv_pred, target_spurious)*spurious_para
            test_loss_target = BCE(test_task_pred, test_pattern)*target_para
            test_loss_target = torch.sum(test_loss_target) / test_pattern.size(0)
            test_loss_MI = torch.mean(torch.sum(test_adv_pred*torch.log(test_adv_pred + 1e-6), axis=1))*MI_para
            
            test_pattern = torch.squeeze(test_pattern, 1)
            test_task_pred = torch.squeeze(test_task_pred, 1)
            record_target_pattern = torch.cat((record_target_pattern, test_pattern), 0)
            record_predict_pattern = torch.cat((record_predict_pattern, test_task_pred), 0)

            running_test_loss_spurious += test_loss_spurious.detach().item()
            running_test_loss_target += test_loss_target.detach().item()
            running_test_loss_MI += test_loss_MI.detach().item()

            if saveallImage == False:
                if i == 0:
                    saveImage(test_image, draw_image, mask, epoch, patient_num, masked_img_dir)
                    
            elif saveallImage == True:
                saveImage(test_image, draw_image, mask, epoch, patient_num, masked_img_dir)

        
        record_target_pattern = record_target_pattern[1::].cpu().numpy()
        record_predict_pattern = record_predict_pattern[1::].cpu().numpy()

        try:
            test_auc = metrics.roc_auc_score(record_target_pattern, record_predict_pattern)
            
        except ValueError:
            pass

        male_metrics = [male_TP, male_TN, male_FP, male_FN]
        female_metrics = [female_TP, female_TN, female_FP, female_FN]
        
        metrics_result = fairness_metrics(male_metrics, female_metrics)
        
        test_result = {
            "spurious_correct": spurious_correct,
            "target_correct": target_correct,
            "test_auc": test_auc,
            "running_test_loss_spurious": running_test_loss_spurious,
            "running_test_loss_target": running_test_loss_target,
            "running_test_loss_MI": running_test_loss_MI,
            "PQD": metrics_result['PQD'],
            "DP": metrics_result['DP'],
            "EOM": metrics_result['EOM'],
            "male_P_acc": metrics_result['male_P_acc'],
            "male_N_acc": metrics_result['male_N_acc'],
            "female_P_acc": metrics_result['female_P_acc'],
            "female_N_acc": metrics_result['female_N_acc'],
            "worst_G_acc": metrics_result['worst_G_acc'],
            "total": total
        }

        return test_result
    
          
