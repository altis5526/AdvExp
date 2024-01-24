import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import numpy as np
import cv2
import torch.nn.functional as F
from torch import nn

def fairness_metrics(male, female):
    male_TP, male_TN, male_FP, male_FN = male
    female_TP, female_TN, female_FP, female_FN = female
    male_acc = (male_TN + male_TP) / (male_TN + male_FP + male_FN + male_TP)
    female_acc = (female_TN + female_TP) / (female_TN + female_FP + female_FN + female_TP)
    max_acc = max(male_acc, female_acc)
    min_acc = min(male_acc, female_acc)
    PQD = min_acc / max_acc
    
    male_dp_P = (male_FP + male_TP) / (male_TN + male_FP + male_FN + male_TP)
    female_dp_P = (female_FP + female_TP) / (female_TN + female_FP + female_FN + female_TP)
    male_dp_N = (male_FN + male_TN) / (male_TN + male_FP + male_FN + male_TP)
    female_dp_N = (female_FN + female_TN) / (female_TN + female_FP + female_FN + female_TP)
    max_dp_P = max(male_dp_P, female_dp_P)
    min_dp_P = min(male_dp_P, female_dp_P)
    max_dp_N = max(male_dp_N, female_dp_N)
    min_dp_N = min(male_dp_N, female_dp_N)
    try:
        DP = min_dp_P / max_dp_P + min_dp_N / max_dp_N
    except:
        DP = 0
    
    male_EOM_P = male_TP / (male_FN + male_TP)
    female_EOM_P = female_TP / (female_FN + female_TP)
    male_EOM_N = male_TN / (male_FP + male_TN)
    female_EOM_N = female_TN / (female_FP + female_TN)
    max_EOM_P = max(male_EOM_P, female_EOM_P)
    min_EOM_P = min(male_EOM_P, female_EOM_P)
    max_EOM_N = max(male_EOM_N, female_EOM_N)
    min_EOM_N = min(male_EOM_N, female_EOM_N)
    try:
        EOM = min_EOM_P / max_EOM_P + min_EOM_N / max_EOM_N
    except:
        EOM = 0
    
    male_P_acc = male_TP / (male_FN + male_TP)
    male_N_acc = male_TN / (male_TN + male_FP)
    female_P_acc = female_TP / (female_FN + female_TP)
    female_N_acc = female_TN / (female_TN + female_FP)
    worst_G_acc = min(male_P_acc, male_N_acc, female_P_acc, female_N_acc)
    
    output = {'PQD': PQD, 'DP': DP, 'EOM': EOM, 'worst_G_acc': worst_G_acc, 'male_P_acc': male_P_acc, 'male_N_acc': male_N_acc, 'female_P_acc': female_P_acc, 'female_N_acc': female_N_acc}
    
    return output

def saveImage(test_image, draw_image, mask, epoch, patient_num, img_dir):
    for j in range(draw_image.size(0)):
        each_mask = mask[j].cpu().numpy()
        each_mask = np.transpose(each_mask, (1,2,0))
        each_mask = 1 - each_mask
        each_mask = (each_mask - np.min(each_mask)) / (np.max(each_mask) - np.min(each_mask))
        
        each_mask = np.uint8(each_mask*255)
        each_mask = cv2.applyColorMap(each_mask, cv2.COLORMAP_HOT)

        original_img = test_image[j].cpu().numpy()
        original_img = np.transpose(original_img, (1,2,0))
        original_img = np.uint8(original_img*255)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

        output = draw_image[j].cpu().numpy()
        output = np.transpose(output, (1,2,0))
        output = np.uint8(output*255)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        final_output = np.concatenate((original_img, each_mask, output), axis = 1)

        cv2.imwrite(f"{img_dir}epoch_{epoch+1}_final_masked_{patient_num[j]}.jpg", final_output)


#def metrics_bias_align_conflict(target_gender, pattern, loss_target, device):
#    bias_align = torch.zeros(target_gender.size(0), dtype=torch.bool).to(device)
#    bias_conflicting = torch.zeros(target_gender.size(0), dtype=torch.bool).to(device)
#    for j in range(target_gender.size(0)):
#        if target_gender[j] == 0 and pattern[j] == 1:
#            bias_align[j] = True
#        elif target_gender[j] == 1 and pattern[j] == 0:
#            bias_align[j] = True
#        elif target_gender[j] == 1 and pattern[j] == 1:
#            bias_conflicting[j] = True
#        elif target_gender[j] == 0 and pattern[j] == 0:
#            bias_conflicting[j] = True
#    if len(loss_target[bias_align]) != 0:
#        bias_align_loss = torch.mean(loss_target[bias_align])
#    else:
#        bias_align_loss = 0
#    if len(loss_target[bias_conflicting]) != 0:
#        bias_conflicting_loss = torch.mean(loss_target[bias_conflicting])
#    else:
#        bias_conflicting_loss = 0
#        
#    return bias_align_loss, bias_conflicting_loss
#
#def CAM(cam, image_tensor, device):
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#    targets = []
#    targets.append(ClassifierOutputTarget(1))
#    input_tensor = image_tensor.to(device)
#    # output = input_tensor.cpu().numpy()
#    
#    # output = np.transpose(output, (0, 2, 3, 1))
#    
#    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
#    grayscale_cam = np.expand_dims(grayscale_cam, 1)
#    
#    return grayscale_cam
#        
#    
#class ContrastiveLoss(nn.Module):
#    def __init__(self, temperature=0.5, verbose=True):
#        super().__init__()
#        self.register_buffer("temperature", torch.tensor(temperature))
#        self.verbose = verbose
#            
#    def forward(self, emb_i, emb_j, postive_mask, negative_mask):
#        if len([postive_mask]) == 0 or len([negative_mask]) == 0:
#            return 0
#        """
#        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
#        z_i, z_j as per SimCLR paper
#        """
#        z_i = F.normalize(emb_i, dim=1)
#        z_j = F.normalize(emb_j, dim=1)
#
#        representations = torch.cat([z_i, z_j], dim=0)
#        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
#        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
#            
#        def l_ij(i, j):
#            z_i_, z_j_ = representations[i], representations[j]
#            sim_i_j = similarity_matrix[i, j]
#            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
#                
#            numerator = torch.exp(sim_i_j / self.temperature)
#            one_for_not_i = torch.ones((2 * emb_i.size(0))).to(emb_i.device).scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
#            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
#            
#            denominator = torch.sum(
#                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
#            )    
#            if self.verbose: print("Denominator", denominator)
#                
#            loss_ij = -torch.log(numerator / denominator)
#            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
#                
#            return loss_ij.squeeze(0)
#
#        loss = 0.0
#        if postive_mask.size(0) != 1:
#            for k in range(postive_mask.size(0)-1):
#                loss += l_ij(postive_mask[k], postive_mask[k+1]) + l_ij(postive_mask[k+1], postive_mask[k]) + l_ij(postive_mask[k], postive_mask[k])
#        elif postive_mask.size(0) == 1:
#            loss += l_ij(postive_mask[0], postive_mask[0])
#        if negative_mask.size(0) != 1:
#            for k in range(negative_mask.size(0)-1):
#                loss += l_ij(negative_mask[k], negative_mask[k+1]) + l_ij(negative_mask[k+1], negative_mask[k]) + l_ij(negative_mask[k], negative_mask[k])
#        elif negative_mask.size(0) == 1:
#            loss += l_ij(negative_mask[0], negative_mask[0])
#        
#        return 1.0 / (2*emb_i.size(0)) * loss
#    
