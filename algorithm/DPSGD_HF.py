from data.util.get_data import get_scatter_transform, get_scattered_dataset, get_scattered_loader
from model.CNN import CIFAR10_CNN_Tanh, MNIST_CNN_Tanh

from privacy_analysis.dp_utils import scatter_normalization, scatter_normalization_direct
from train_and_validation.train import train
from train_and_validation.train_with_dp import train_with_dp_sgd
from utils.dp_optimizer import  DPSGD_Optimizer
import torch

from train_and_validation.validation import validation

from data.util.sampling import get_data_loaders_possion

# from algorithm.CompDP import *
import os
import math


def DPSGD_HF(dataset_name, train_data, test_data, batch_size, lr, momentum, epsilon, delta, iteration_times, C_t, use_scattering, input_norm, bn_noise_multiplier, num_groups, device):

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

    while iter < iteration_times:

        train_dl = minibatch_loader(train_data_scattered)
        for id, (data, target) in enumerate(train_dl):
            optimizer.minibatch_size = len(data)

        train_loss, train_accuracy = train_with_dp_sgd(model, train_dl, optimizer, device)

        test_loss, test_accuracy = validation(model, test_loader,device)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_iter = iter

        print(f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')
        iter+=1
        epsilon_list.append(torch.tensor(epsilon))
        test_loss_list.append(test_loss)

    print("------ finished ------")
    return test_accuracy,iter,best_test_acc,best_iter,model,[epsilon_list,test_loss_list]

CNNS = {
    "CIFAR-10": CIFAR10_CNN_Tanh,
    "FMNIST": MNIST_CNN_Tanh,
    "MNIST": MNIST_CNN_Tanh,
}