import argparse
import copy
import os
import pickle
from datetime import time, datetime

import pandas as pd
import torch

from data.util.get_data import get_data

from model.get_model import get_model
from utils.dp_optimizer import get_dp_optimizer

from algorithm.DPSGD import DPSGD
from algorithm.DPAGD import DPAGD
from algorithm.DPSGD_TS import DPSGD_TS
from algorithm.DPSGD_HF import DPSGD_HF
from algorithm.DPSGD_Comp import DPSGD_Comp
from algorithm.DPAGD_Comp import DPAGD_Comp
from algorithm.DPSGD_TS_Comp import DPSGD_TS_Comp
from algorithm.DPSGD_HF_Comp import DPSGD_HF_Comp
from algorithm.DPSUR import DPSUR
from algorithm.DPSUR_Comp import DPSUR_Comp
from privacy_parameters.privacy_params import Privacy_param

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="DPSGD",choices=['DPSGD', 'DPSGD-Comp', 'DPAGD', 'DPAGD-Comp', 'DPSGD-TS', 'DPSGD-TS-Comp', 'DPSGD-HF', 'DPSGD-HF-Comp', 'DPSUR', 'DPSUR-Comp'])
    parser.add_argument('--dataset_name', type=str, default="MNIST",choices=['MNIST', 'FMNIST', 'CIFAR-10', 'IMDB'])
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--use_scattering', action="store_true")
    parser.add_argument('--input_norm', default=None, choices=["GroupNorm", "BN"])
    parser.add_argument('--bn_noise_multiplier', type=float, default=8)
    parser.add_argument('--num_groups', type=int, default=27)

    # C_t is the clipping threshold value
    parser.add_argument('--C_t', type=float, default=0.5)

    # Privacy parameters
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=10e-6)

    # PM_index refers to privacy mechanism index
    parser.add_argument('--PM_index', type=int, default=2)

    # iteration_times refers to the total iteration times
    parser.add_argument('--iteration_times', type=int, default=3000)

    parser.add_argument('--MIA', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=256)

    # Noisy applied for DPAGD in choosing learning rate (for function noisyMax)
    parser.add_argument('--sigma_v', type=float, default=1.0)
    parser.add_argument('--C_v', type=float, default=0.001)
    parser.add_argument('--bs_valid', type=int, default=256)
    parser.add_argument('--beta', type=float, default=-1.0)

    parser.add_argument('--T_acc', type=float, default=90.0)
    parser.add_argument('--t_ep_rate', type=float, default=0.01)
    parser.add_argument('--t_ep_min', type=float, default=1.0)
    parser.add_argument('--t_ep_max', type=float, default=1.0)

    parser.add_argument('--T_gain', type=float, default=1.01)
    parser.add_argument('--t_lr_rate', type=float, default=0.1)
    parser.add_argument('--t_lr_min', type=float, default=0.1)
    parser.add_argument('--t_lr_max', type=float, default=2.0)
    parser.add_argument('--N', type=int, default=10)

    # Load initial parameters
    args = parser.parse_args()

    algorithm = args.algorithm
    dataset_name = args.dataset_name
    lr = args.lr
    momentum = args.momentum

    use_scattering = args.use_scattering
    input_norm = args.input_norm
    bn_noise_multiplier = args.bn_noise_multiplier
    num_groups = args.num_groups

    C_t = args.C_t

    epsilon = args.epsilon
    delta = args.delta
    PM_index = args.PM_index

    iteration_times = args.iteration_times

    MIA = args.MIA
    device = args.device
    batch_size = args.batch_size

    sigma_v = args.sigma_v
    C_v = args.C_v
    bs_valid = args.bs_valid
    beta = args.beta

    T_acc = args.T_acc
    t_ep_rate = args.t_ep_rate
    t_ep_min = args.t_ep_min
    t_ep_max = args.t_ep_max

    T_gain = args.T_gain
    t_lr_rate = args.t_lr_rate
    t_lr_min = args.t_lr_min
    t_lr_max = args.t_lr_max

    N = args.N


    if MIA:
        print("MIA still in coding")

    else:
        train_data, test_data, dataset = get_data(dataset_name, augment=False)
        model = get_model(algorithm, dataset_name, device)
        optimizer = get_dp_optimizer(dataset_name, algorithm, lr, lr, momentum, C_t, epsilon, delta, batch_size, model, device)

        if algorithm == 'DPSGD':
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPSGD(train_data, test_data, model, optimizer, batch_size, iteration_times, device)
        elif algorithm == 'DPAGD':
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPAGD(train_data, test_data, model, optimizer, batch_size, iteration_times, C_v, sigma_v, device)
        elif algorithm == 'DPSGD-TS':
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPSGD_TS(train_data, test_data, model, optimizer, batch_size, iteration_times, device)
        elif algorithm == 'DPSGD-HF':
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPSGD_HF(dataset_name, train_data, test_data, batch_size, lr, momentum, epsilon, delta, iteration_times, C_t, use_scattering, input_norm, bn_noise_multiplier, num_groups, device)
        elif algorithm == "DPSUR":
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPSUR(dataset_name, train_data, test_data, model, batch_size, lr, momentum, delta, iteration_times, C_t, epsilon, use_scattering, input_norm, bn_noise_multiplier, num_groups, bs_valid, C_v, beta, sigma_v, device)
        elif algorithm == 'DPSGD-Comp':
            privacy_params = Privacy_param(T_acc, t_ep_rate, t_ep_min, t_ep_max, T_gain, t_lr_rate, t_lr_min, t_lr_max, N)
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPSGD_Comp(train_data, test_data, model, optimizer, privacy_params, batch_size, iteration_times, PM_index, device)
        elif algorithm == 'DPAGD-Comp':
            privacy_params = Privacy_param(T_acc, t_ep_rate, t_ep_min, t_ep_max, T_gain, t_lr_rate, t_lr_min, t_lr_max, N)
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPAGD_Comp(train_data, test_data, model, optimizer, privacy_params, batch_size, iteration_times, C_v, sigma_v, PM_index, device)
        elif algorithm == 'DPSGD-TS-Comp':
            privacy_params = Privacy_param(T_acc, t_ep_rate, t_ep_min, t_ep_max, T_gain, t_lr_rate, t_lr_min, t_lr_max, N)
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPSGD_TS_Comp(train_data, test_data, model, optimizer, privacy_params, batch_size, iteration_times, PM_index, device)
        elif algorithm == 'DPSGD-HF-Comp':
            privacy_params = Privacy_param(T_acc, t_ep_rate, t_ep_min, t_ep_max, T_gain, t_lr_rate, t_lr_min, t_lr_max, N)
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPSGD_HF_Comp(dataset_name, train_data, test_data, privacy_params, batch_size, lr, momentum, epsilon, delta, iteration_times, C_t, use_scattering, input_norm, bn_noise_multiplier, num_groups, PM_index, device)
        elif algorithm == "DPSUR-Comp":
            privacy_params = Privacy_param(T_acc, t_ep_rate, t_ep_min, t_ep_max, T_gain, t_lr_rate, t_lr_min, t_lr_max, N)
            test_acc, last_iter, best_acc, best_iter, trained_model, iter_list = DPSUR_Comp(dataset_name, train_data, test_data, model, privacy_params, batch_size, lr, momentum, delta, iteration_times, C_t, epsilon, use_scattering, input_norm, bn_noise_multiplier, num_groups, bs_valid, C_v, beta, sigma_v, PM_index, device)

        else:
            raise ValueError("this algorithm is not exist")

        # Finish training ...........
        if MIA:
            print("MIA still in coding")

        else:
            File_Path_Csv = os.getcwd() + f"/result/Without_MIA/{algorithm}/{dataset_name}/{epsilon}//"
            if not os.path.exists(File_Path_Csv):
                os.makedirs(File_Path_Csv)
            result_path = f'{File_Path_Csv}/{str(epsilon)}_{str(lr)}_{str(batch_size)}.csv'
            pd.DataFrame([best_acc, int(best_iter), test_acc, int(last_iter)]).to_csv(result_path, index=False, header=False)
            torch.save(iter_list, f"{File_Path_Csv}/iterList.pth")


if __name__=="__main__":

    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    main()
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("start time: ", start_time)
    print("end time: ", end_time)