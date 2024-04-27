import torch

from data.util.sampling import get_data_loaders_possion
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from train_and_validation.train import train
from train_and_validation.train_with_dp import train_with_dp_agd
from train_and_validation.validation import validation
# from algorithm.CompDP import *


def DPAGD(train_data, test_data, model, optimizer, batch_size, iteration_times, C_v, sigma_v, device):

    minibatch_loader, microbatch_loader = get_data_loaders_possion(minibatch_size=batch_size,microbatch_size=1,iterations=1)

    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    iter = 1
    best_test_acc=0.
    epsilon_list=[]
    test_loss_list=[]
    test_accuracy = 0
    ini_epsilon = optimizer.epsilon

    while iter < iteration_times:

        train_dl = minibatch_loader(train_data)  # possion sampling
        for id, (data, target) in enumerate(train_dl):
            optimizer.minibatch_size = len(data)

        train_loss, train_accuracy = train_with_dp_agd(model, train_dl, optimizer, ini_epsilon, C_v, sigma_v, device)
        test_loss, test_accuracy = validation(model, test_dl, device)
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

