from data.util.get_data import get_scatter_transform, get_scattered_dataset, get_scattered_loader
from model.CNN import CIFAR10_CNN_Tanh, MNIST_CNN_Tanh

from privacy_analysis.dp_utils import scatter_normalization, scatter_normalization_direct
from train_and_validation.train import train
from train_and_validation.train_with_dp import train_with_dp_sgd, train_with_dp_sgd_comp
from utils.dp_optimizer import  DPSGD_Optimizer
import torch

from train_and_validation.validation import validation

from data.util.sampling import get_data_loaders_possion

from algorithm.CompDP import *
import os
import math


def DPSGD_HF_Comp(dataset_name, train_data, test_data, privacy_params, batch_size, lr, momentum, epsilon, delta, iteration_times, C_t, use_scattering, input_norm, bn_noise_multiplier, num_groups, index, device):

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    if use_scattering:
        scattering, K, _ = get_scatter_transform(dataset_name)
        scattering.to(device)
    else:
        scattering = None
        K = 3 if len(train_data.data.shape) == 4 else 1

    noise_multiplier_tmp =(math.sqrt(2 * math.log(1.25 / delta))) / epsilon

    if input_norm == "BN":
        # compute noisy data statistics or load from disk if pre-computed
        save_dir = f"bn_stats/{dataset_name}"
        os.makedirs(save_dir, exist_ok=True)
        bn_stats = scatter_normalization_direct(train_loader,
                                                   scattering,
                                                   K,
                                                   device,
                                                   len(train_data),
                                                   len(train_data),
                                                   noise_multiplier=bn_noise_multiplier,
                                                   save_dir=save_dir)
        model = CNNS[dataset_name](K, input_norm="BN", bn_stats=bn_stats, size=None)

    else:
        model = CNNS[dataset_name](K, input_norm=input_norm, num_groups=num_groups, size=None)

    model.to(device)

    train_data_scattered = get_scattered_dataset(train_loader, scattering, device, len(train_data))
    test_loader = get_scattered_loader(test_loader, scattering, device)

    minibatch_loader, microbatch_loader = get_data_loaders_possion(minibatch_size=batch_size, microbatch_size=1, iterations=1)

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

    iter = 1
    epsilon_list=[]
    test_loss_list=[]
    best_test_acc=0.

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
        train_dl = minibatch_loader(train_data_scattered)
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

        test_loss, test_accuracy = validation(model, test_loader, device)

        Acc_ave = Acc_ave + test_accuracy

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = iter

        print(f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')
        iter+=1
        epsilon_list.append(torch.tensor(epsilon))
        test_loss_list.append(test_loss)

    print("------ finished ------")
    return test_accuracy, iter, best_test_acc, best_iter, model, [epsilon_list, test_loss_list]

CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}