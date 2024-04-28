from data.util.get_data import get_scatter_transform, get_scattered_dataset, get_scattered_loader
from model.CNN import  CIFAR10_CNN_Tanh, MNIST_CNN_Tanh
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.get_MaxSigma_or_MaxSteps import get_max_steps, get_min_sigma
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from privacy_analysis.dp_utils import scatter_normalization, scatter_normalization_direct
from utils.dp_optimizer import DPSGD_Optimizer, DPAdam_Optimizer
import torch

from train_and_validation.train_with_dp import train_with_dp_sgd_comp
from train_and_validation.validation import validation
import copy
import numpy as np

from data.util.sampling import  get_data_loaders_possion

from data.util.dividing_validation_data import dividing_validation_set, dividing_validation_set_for_IMDB
import os
from algorithm.CompDP import *


def DPSUR_Comp(dataset_name, train_dataset, test_data, model, privacy_params, batch_size, lr, momentum, delta, iteration_times, C_t, epsilon, use_scattering, input_norm, bn_noise_multiplier, num_groups, bs_valid, C_v, beta, sigma_v, index, device):

    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    if dataset_name != 'IMDB':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

        if use_scattering:
            scattering, K, _ = get_scatter_transform(dataset_name)
            scattering.to(device)
        else:
            scattering = None
            K = 3 if len(train_dataset.data.shape) == 4 else 1

        if input_norm == "BN":
            save_dir = f"bn_stats/{dataset_name}"
            os.makedirs(save_dir, exist_ok=True)
            bn_stats = scatter_normalization_direct(train_loader,
                                                    scattering,
                                                    K,
                                                    device,
                                                    len(train_dataset),
                                                    len(train_dataset),
                                                    noise_multiplier=bn_noise_multiplier,
                                                    save_dir=save_dir)
            model = CNNS[dataset_name](K, input_norm="BN", bn_stats=bn_stats, size=None)
        else:
            model = CNNS[dataset_name](K, input_norm=input_norm, num_groups=num_groups, size=None)


        model.to(device)
        train_data = get_scattered_dataset(train_loader, scattering, device, len(train_dataset))
        test_dl = get_scattered_loader(test_dl, scattering, device)

        optimizer = DPSGD_Optimizer(
            lr=lr,
            l2_norm_clip=C_t,
            epsilon=epsilon,
            delta=delta,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            device=device,
            lr_used=lr,
            momentum=momentum,
        )
    else:
        optimizer = DPAdam_Optimizer(
            lr=lr,
            l2_norm_clip=C_t,
            epsilon=epsilon,
            delta=delta,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            device=device,
            lr_used=lr,
        )
    minibatch_loader_for_train, microbatch_loader = get_data_loaders_possion(minibatch_size=batch_size, microbatch_size=1, iterations=1)
    minibatch_loader_for_valid, microbatch_loader = get_data_loaders_possion(minibatch_size=bs_valid, microbatch_size=1, iterations=1)

    last_valid_loss = 100000.0
    last_accept_test_acc = 0.
    last_model = model
    t = 1
    iter = 1
    best_iter = 1
    best_test_acc = 0.
    epsilon_list = []
    test_loss_list = []

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
        if dataset_name=='IMDB':
            train_dl = minibatch_loader_for_train(train_dataset)
            valid_dl = minibatch_loader_for_valid(train_dataset)
            for id, (data, target) in enumerate(train_dl):
                optimizer.minibatch_size = len(data)

        else:
            # training =========#
            train_dl = minibatch_loader_for_train(train_data)
            valid_dl = minibatch_loader_for_valid(train_data)
            # ==================#
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

        train_loss, train_accuracy = train_with_dp_sgd_comp(model, train_dl, optimizer, device)

        valid_loss, valid_accuracy = validation(model, valid_dl, device)

        test_loss, test_accuracy = validation(model, test_dl, device)

        Acc_ave = Acc_ave + test_accuracy

        deltaE=valid_loss - last_valid_loss
        # deltaE=torch.tensor(deltaE).cpu()
        # deltaE = torch.tensor(deltaE, device=device)
        # deltaE = deltaE.to(device)
        print("Delta E:",deltaE)

        # deltaE= np.clip(deltaE, -C_v, C_v)
        deltaE = torch.clamp(deltaE, -C_v, C_v)

        deltaE_after_dp =2*C_v*sigma_v*np.random.normal(0,1)+deltaE

        print("Delta E after dp:",deltaE_after_dp)

        if deltaE_after_dp < beta*C_v:
            last_valid_loss = valid_loss
            last_model = copy.deepcopy(model)
            t = t + 1
            print("accept updates，the number of updates t：", format(t))
            last_accept_test_acc = test_accuracy

            if last_accept_test_acc > best_test_acc:
                best_test_acc = last_accept_test_acc
                best_iter = t

            epsilon_list.append(torch.tensor(epsilon))
            test_loss_list.append(test_loss)

        else:
            print("reject updates")
            model.load_state_dict(last_model.state_dict(), strict=True)

        print(f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')

        iter+=1

    print("------ finished ------")
    return last_accept_test_acc, t, best_test_acc, best_iter, last_model, [epsilon_list, test_loss_list]

CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}