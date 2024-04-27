import torch

from data.util.sampling import get_data_loaders_possion
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from train_and_validation.train import train
from train_and_validation.train_with_dp import train_with_dp_agd, train_with_dp_agd_comp
from train_and_validation.validation import validation
from algorithm.CompDP import *


def DPAGD_Comp(train_data, test_data, model, optimizer, privacy_params, batch_size, iteration_times, C_v, sigma_v, index, device):

    minibatch_loader, microbatch_loader = get_data_loaders_possion(minibatch_size=batch_size,microbatch_size=1,iterations=1)

    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    iter = 1
    best_test_acc=0.
    epsilon_list=[]
    test_loss_list=[]
    test_accuracy = 0
    ini_epsilon = optimizer.epsilon

    para_q = 0
    para_y = 0

    if (index == 1):
        para_q, para_y, Var = parameter_tuning(optimizer.epsilon, optimizer.l2_norm_clip, index)
    else:
        para_q, Var = parameter_tuning(optimizer.epsilon, optimizer.l2_norm_clip, index)

    optimizer.custom_noise_list(para_q, index, para_y)

    ep_ini = copy.deepcopy(optimizer.epsilon)
    ep_allocation = 0
    t_ep_rate = privacy_params.t_ep_rate
    t_ep_min = privacy_params.t_ep_min
    t_ep_max = privacy_params.t_ep_max
    T_acc = privacy_params.T_acc

    T_gain = privacy_params.T_gain
    t_lr_rate = privacy_params.t_lr_rate
    t_lr_min = privacy_params.t_lr_min
    t_lr_max = privacy_params.t_lr_max

    Acc_ave_last = 0
    Acc_ave_current = 0

    N = privacy_params.N

    mode = 0

    Acc_ave = 0

    while iter < iteration_times:

        train_dl = minibatch_loader(train_data)  # possion sampling
        for id, (data, target) in enumerate(train_dl):
            optimizer.minibatch_size = len(data)

        if (iter % N) == 0:
            mode = mode + 1
            if mode == 1:
                Acc_ave_current = Acc_ave/N
                Acc_ave = 0
            else:
                Acc_ave_last = copy.deepcopy(Acc_ave_current)
                Acc_ave_current = Acc_ave/N
                Acc_ave = 0

                # Adjust the learning rate
                if Acc_ave_current/Acc_ave_last <= T_gain:
                    optimizer.lr_used = max(optimizer.lr_used*(1-t_lr_rate), t_lr_min)
                else:
                    optimizer.lr_used = min(optimizer.lr_used*(1+t_lr_rate), t_lr_max)

                # Adjust the privacy budget
                if Acc_ave_current <= T_acc:
                    optimizer.epsilon = max(optimizer.epsilon*(1-t_ep_rate), t_ep_min)
                else:
                    if ep_allocation >= 0:
                        optimizer.epsilon = min(optimizer.epsilon*(1+t_ep_rate), t_ep_max)
                    else:
                        optimizer.epsilon = copy.deepcopy(ep_ini)

        if ep_allocation >= 0:
            ep_allocation = ep_allocation + (ep_ini - optimizer.epsilon)
        else:
            optimizer.epsilon = copy.deepcopy(ep_ini)

        train_loss, train_accuracy = train_with_dp_agd_comp(model, train_dl, optimizer, ini_epsilon, C_v, sigma_v, device)
        test_loss, test_accuracy = validation(model, test_dl, device)

        Acc_ave = Acc_ave + test_accuracy

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = iter

        epsilon_list.append(torch.tensor(optimizer.epsilon))
        test_loss_list.append(test_loss)
        print(
            f'iters:{iter},'f'epsilon:{optimizer.epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')
        iter += 1

    print("------finished ------")
    return test_accuracy, iter, best_test_acc, best_iter, model, [epsilon_list, test_loss_list]

