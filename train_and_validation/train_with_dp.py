import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from privacy_analysis.RDP.compute_rdp import compute_rdp
from train_and_validation.validation import validation, validation_per_sample
import numpy as np

from utils.NoisyMax import NoisyMax

from math import sqrt

def train_with_dp_sgd(model, train_loader, optimizer,device):
    model.train()
    train_loss = 0.0
    train_acc=0.
    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)  #改为负数似然损失函数了，后面记得要改回来

            loss.backward()
            optimizer.microbatch_step()
        optimizer.step_dp_sgd()

    return train_loss, train_acc

def train_with_dp_sgd2(model, train_loader, optimizer,device):
    model.train()
    train_loss = 0.0
    train_acc=0.
    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)  #改为负数似然损失函数了，后面记得要改回来

            loss.backward()
            optimizer.microbatch_step()

        # 在执行参数更新之前打印学习率
        current_lr = optimizer.param_groups[0]['lr']  # 假设所有参数组使用相同的学习率
        # print(f"Current learning rate: {current_lr}")

        optimizer.step_dp_sgd()

    return train_loss, train_acc

def train_with_dp_sgd_comp(model, train_loader, optimizer,device):
    model.train()
    train_loss = 0.0
    train_acc=0.
    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)  #改为负数似然损失函数了，后面记得要改回来

            loss.backward()
            optimizer.microbatch_step()
        optimizer.step_dp_sgd_comp(new_lr=optimizer.lr_used)

    return train_loss, train_acc

def train_with_dp_sgd_comp2(model, train_loader, optimizer,device):
    model.train()
    train_loss = 0.0
    train_acc=0.
    for id,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()
        for iid,(X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):

            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape)==2:
                output=torch.squeeze(output,0)
            loss = F.cross_entropy(output, y_microbatch)  #改为负数似然损失函数了，后面记得要改回来

            loss.backward()
            optimizer.microbatch_step()

        # 在执行参数更新之前打印学习率
        current_lr = optimizer.param_groups[0]['lr']  # 假设所有参数组使用相同的学习率
        print(f"Current learning rate: {current_lr}")

        optimizer.step_dp_sgd_comp(new_lr=optimizer.lr_used)

    return train_loss, train_acc

def select_learning_rate(model, original_gradients, train_loader, device, learning_rates, C_v, sigma_v, n):
    losses = []
    for lr in learning_rates:
        # 模拟更新参数
        apply_gradients(model, original_gradients, lr)
        test_loss = validation_per_sample(model, train_loader, device, C_v)
        losses.append(test_loss)
        # 还原参数
        apply_gradients(model, original_gradients, -lr)

    # 使用NoisyMax选择最小损失的学习率
    min_index = NoisyMax(losses, sigma_v, C_v, n, device)
    return learning_rates[min_index], min_index

def apply_gradients(model, gradients, lr):
    # 模拟更新参数
    with torch.no_grad():
        for param, grad in zip(model.parameters(), gradients):
            param -= lr * grad


def train_with_dp_agd(model, train_loader, optimizer, ini_epsilon, C_v, sigma_v, device):
    # print("train with adg2")
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    C_t = optimizer.l2_norm_clip
    noise_multiplier = optimizer.gaussian_noise_multiplier(optimizer.epsilon, optimizer.delta)
    lr_used = optimizer.lr_used

    optimizer.adjust_clipping_and_noise_and_lr(noise_multiplier, C_t, lr_used)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        original_gradients = [param.grad.clone() for param in model.parameters()]
        optimizer.step_agd_no_update_grad()

        # 学习率选择
        learning_rate = np.linspace(0.1, 2, 10)
        min_lr, min_index = select_learning_rate(model, original_gradients, train_loader, device, learning_rate, C_v, sigma_v, len(target))

        if min_index > 0:
            gamma = 1.01
            epsilon_used = min(optimizer.epsilon * gamma, ini_epsilon * 1.05)
            optimizer.epsilon = epsilon_used

        # 恢复原始梯度并更新参数
        for param, grad in zip(model.parameters(), original_gradients):
            param.grad = grad
        optimizer.lr_used = min_lr
        train_loss, train_acc = train_with_dp_sgd2(model, train_loader, optimizer, device)

    return train_loss, train_acc



def train_with_dp_agd2(model, train_loader, optimizer, ini_epsilon, C_v, sigma_v, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    C_t = optimizer.l2_norm_clip
    noise_multiplier = optimizer.gaussian_noise_multiplier(optimizer.epsilon, optimizer.delta)
    lr_used = optimizer.lr_used

    # print("C_t =", C_t)
    # print("noise_multiplier =", noise_multiplier)
    # print("lr_used =", lr_used)

    optimizer.adjust_clipping_and_noise_and_lr(noise_multiplier, C_t, lr_used)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step_agd_no_update_grad()  # 在这一步，优化器应用了调整后的裁剪和噪声

        # Gradient update
        # 获取原参数和裁剪的梯度值,这个是为了后面可能重加噪用的

        model_parameters = model.parameters()
        gradients = [param.grad.clone() for param in model_parameters]

        model_parameters_dict = model.state_dict()
        # optimizer.step_agd_update_with_new_lr(new_lr=0.2)

        #开始更新学习率了：
        learning_rate = np.linspace(0.1, 0.5, 10)  # 学习率从0.1-2.0分成20份

        loss = []
        for i, lr in enumerate(learning_rate):
            # 更新参数
            with torch.no_grad():

                for param, gradient in zip(model_parameters_dict.values(), gradients):
                    param -= lr * gradient

                test_loss = validation_per_sample(model, train_loader, device, C_v)
                loss.append(test_loss)

                for param, gradient in zip(model_parameters_dict.values(), gradients):
                    param += lr * gradient


        # 找最小值
        min_index = NoisyMax(loss, sigma_v, C_v, len(target), device)

        if min_index>0:
            # 拿到使得这次loss最小的梯度值
            lr_used = learning_rate[min_index]
            # print("lr_used =", lr_used)
            optimizer.lr_used = lr_used
        else:
            # 如果是0最佳的，那么需要进行隐私预算加大，即多分配隐私预算，然后sigma变小
            # 对g进行重加噪，用小的sigma进行重加噪，然后隐私资源消耗
            gamma = 1.01
            epsilon_used = min(optimizer.epsilon * gamma, ini_epsilon*1.05)

            # print("epsilon_used:", epsilon_used)
            # print("ini_epsilon:", ini_epsilon)
            lr_used = learning_rate[min_index]
            optimizer.lr_used = lr_used
            optimizer.epsilon = epsilon_used

        optimizer.step_agd_update_with_new_lr(new_lr=lr_used)

    return train_loss, train_acc

def train_with_dp_agd_comp(model, train_loader, optimizer, ini_epsilon, C_v, sigma_v, device):
    # print("train with adg2")
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    C_t = optimizer.l2_norm_clip
    noise_multiplier = optimizer.gaussian_noise_multiplier(optimizer.epsilon, optimizer.delta)
    lr_used = optimizer.lr_used

    optimizer.adjust_clipping_and_noise_and_lr(noise_multiplier, C_t, lr_used)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        original_gradients = [param.grad.clone() for param in model.parameters()]
        optimizer.step_agd_no_update_grad()

        # 学习率选择
        learning_rate = np.linspace(0.05, 0.1, 5)
        min_lr, min_index = select_learning_rate(model, original_gradients, train_loader, device, learning_rate, C_v, sigma_v, len(target))

        if min_index > 0:
            gamma = 1.01
            epsilon_used = min(optimizer.epsilon * gamma, ini_epsilon * 1.05)
            optimizer.epsilon = epsilon_used

        # 恢复原始梯度并更新参数
        for param, grad in zip(model.parameters(), original_gradients):
            param.grad = grad
        optimizer.lr_used = min_lr

        train_loss, train_acc = train_with_dp_sgd_comp2(model, train_loader, optimizer, device)

        # train_loss, train_acc = train_with_dp_sgd2(model, train_loader, optimizer, device)

    return train_loss, train_acc


def train_with_dp_agd_comp2(model, train_loader, optimizer, ini_epsilon, C_v, sigma_v, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    C_t = optimizer.l2_norm_clip
    noise_multiplier = optimizer.gaussian_noise_multiplier(optimizer.epsilon, optimizer.delta)
    lr_used = optimizer.lr_used

    optimizer.adjust_clipping_and_noise_and_lr(noise_multiplier, C_t, lr_used)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step_agd_no_update_grad_comp()  # 在这一步，优化器应用了调整后的裁剪和噪声

        # Gradient update
        # 获取原参数和裁剪的梯度值,这个是为了后面可能重加噪用的
        model_parameters_clipped = model.parameters()   #获取原参数
        gradients_clipped = [param.grad.clone() for param in model_parameters_clipped]

        model_parameters = model.parameters()
        gradients = [param.grad.clone() for param in model_parameters]

        model_parameters_dict = model.state_dict()

        #开始更新学习率了：
        learning_rate = np.linspace(0.1, 0.5, 10)  # 学习率从0.1-2.0分成20份

        loss = []
        for i, lr in enumerate(learning_rate):
            # 更新参数
            with torch.no_grad():

                for param, gradient in zip(model_parameters_dict.values(), gradients):
                    param -= lr * gradient

                test_loss = validation_per_sample(model, train_loader, device, C_v)
                loss.append(test_loss)

                for param, gradient in zip(model_parameters_dict.values(), gradients):
                    param += lr * gradient


        # 找最小值
        min_index = NoisyMax(loss, sigma_v, C_v, len(target), device)

        if min_index>0:
            # 拿到使得这次loss最小的梯度值
            lr_used = learning_rate[min_index]
            optimizer.lr_used = lr_used
        else:
            # 如果是0最佳的，那么需要进行隐私预算加大，即多分配隐私预算，然后sigma变小
            # 对g进行重加噪，用小的sigma进行重加噪，然后隐私资源消耗
            gamma = 1.01
            epsilon_used = min(optimizer.epsilon * gamma, ini_epsilon*1.05)

            lr_used = learning_rate[min_index]
            optimizer.lr_used = lr_used
            optimizer.epsilon = epsilon_used

        optimizer.step_agd_update_with_new_lr(new_lr=lr_used)

    return train_loss, train_acc
