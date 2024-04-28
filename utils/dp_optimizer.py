
import numpy as np
import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop
# from algorithm.CompDP import *
import math
from algorithm.CompDP import *
import random

def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, epsilon, delta, minibatch_size, microbatch_size, lr_used, device, *args, **kwargs):

            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.epsilon = epsilon
            self.delta = delta
            self.minibatch_size = minibatch_size
            self.microbatch_size = microbatch_size
            self.device = device
            self.lr_used = lr_used


            for id,group in enumerate(self.param_groups):
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def gaussian_noise_multiplier(self, epsilon, delta):
            sigma = (math.sqrt(2 * math.log(1.25 / delta))) / epsilon
            return sigma

        def adjust_clipping_and_noise_and_lr(self, noise_multiplier, clipping_threshold, new_lr):
            """
            Adjusts the noise multiplier and clipping threshold for all parameter groups.
            """
            for group in self.param_groups:
                group['noise_multiplier'] = noise_multiplier
                group['l2_norm_clip'] = clipping_threshold
                group['lr'] = new_lr

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()


        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.

            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm+ 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            return total_norm


        def zero_accum_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step_dp_sgd(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:

                        param.grad.data = accum_grad.clone()

                        noise_multiplier = self.gaussian_noise_multiplier(self.epsilon, self.delta)

                        param.grad.data.add_(self.l2_norm_clip * noise_multiplier * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            super(DPOptimizerClass, self).step(*args, **kwargs)

        def custom_noise_list(self, para_q, index, para_y=0.5):
            # 调用perturbation_multiple来一次性生成所有噪声值
            noise_list_tmp = perturbation_Q_multiple_list(0, para_q, self.epsilon, self.l2_norm_clip, index, para_y)
            self.noise_list = noise_list_tmp

        def composite_noise(self, shape, *args, **kwargs):

            # 使用np.prod来计算形状的所有维度上的元素总数
            volume = np.prod(shape)

            random.shuffle(self.noise_list)
            noise_list = []
            for i in range(volume):
                noise_list.append(self.noise_list[i % len(self.noise_list)])

            # 确保噪声列表长度与张量的元素总数匹配
            if len(noise_list) != volume:
                raise ValueError(
                    f"The length of noise_list ({len(noise_list)}) does not match the total number of elements in the shape ({volume}).")

            # 将噪声列表转换为张量，并调整为原始张量的形状
            noise_tensor = torch.tensor(noise_list, device=self.device).view(shape)

            return noise_tensor

        # def step_dp_sgd_comp(self, *args, **kwargs):
        #     for group in self.param_groups:
        #         for param, accum_grad in zip(group['params'], group['accum_grads']):
        #             if param.requires_grad:
        #
        #                 # 将累积的、已裁剪的梯度复制到param.grad.data
        #                 param.grad.data = accum_grad.clone()
        #
        #                 # 使用自定义的噪声函数生成噪声并添加
        #                 noise = self.composite_noise(param.grad.data.shape)
        #                 param.grad.data.add_(noise)
        #
        #                 param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
        #
        #     super(DPOptimizerClass, self).step(*args, **kwargs)


        def step_dp_sgd_comp(self, new_lr=None, *args, **kwargs):
            if new_lr is not None:
                for group in self.param_groups:
                    group['lr'] = new_lr

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:

                        # 将累积的、已裁剪的梯度复制到param.grad.data
                        param.grad.data = accum_grad.clone()

                        # 使用自定义的噪声函数生成噪声并添加
                        noise = self.composite_noise(param.grad.data.shape)
                        param.grad.data.add_(noise)

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            super(DPOptimizerClass, self).step(*args, **kwargs)



        def step_agd_no_update_grad(self, closure=None):
            """
            Performs a single optimization step (parameter update).
            """
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                l2_norm_clip = group['l2_norm_clip']
                noise_multiplier = group['noise_multiplier']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    param_norm = grad.norm(2)
                    clip_coef = l2_norm_clip / (param_norm + 1e-6)
                    clip_coef = min(clip_coef, 1.0)
                    grad = grad * clip_coef

                    # Add composite noise for differential privacy
                    noise = torch.randn_like(grad) * l2_norm_clip * noise_multiplier
                    grad = grad + noise / self.minibatch_size
                    p.grad.data = grad

        def step_agd_no_update_grad_comp(self, closure=None):
            """
            Performs a single optimization step (parameter update).
            """
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                l2_norm_clip = group['l2_norm_clip']
                noise_multiplier = group['noise_multiplier']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    param_norm = grad.norm(2)
                    clip_coef = l2_norm_clip / (param_norm + 1e-6)
                    clip_coef = min(clip_coef, 1.0)
                    grad = grad * clip_coef

                    # Add noise for differential privacy
                    noise = self.composite_noise(p.grad.data.shape)
                    grad = grad + noise / self.minibatch_size
                    p.grad.data = grad

        def step_agd_update_with_new_lr(self, new_lr, closure=None):
            """Performs a single optimization step (parameter update)."""
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                group['lr'] = new_lr
                for p in group['params']:
                    if p.grad is None:
                        continue
                    # 使用已经裁剪并加噪的梯度直接更新参数
                    # p.data.add_(-group['lr'], p.grad.data)
                    p.data.add_(p.grad.data, alpha=-group['lr'])

            return loss


    return DPOptimizerClass

DPAdam_Optimizer = make_optimizer_class(Adam)
DPAdagrad_Optimizer = make_optimizer_class(Adagrad)
DPSGD_Optimizer = make_optimizer_class(SGD)
DPRMSprop_Optimizer = make_optimizer_class(RMSprop)


def get_dp_optimizer(dataset_name, algortithm, lr, lr_used, momentum, C_t, epsilon, delta, batch_size, model, device):
    print("algorithm =", algortithm)
    if dataset_name=='IMDB' and (algortithm != 'DPAGD' or algortithm != 'DPAGD-Comp'):
        optimizer = DPAdam_Optimizer(
            lr=lr,
            l2_norm_clip=C_t,
            epsilon=epsilon,
            delta=delta,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            device=device,
            lr_used=lr_used,
        )
        print("Optimizer-with-no-momentum")
    else:
        optimizer = DPSGD_Optimizer(
            lr=lr,
            l2_norm_clip=C_t,
            epsilon=epsilon,
            delta=delta,
            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            device=device,
            lr_used=lr_used,
            momentum=momentum,
        )
        print("Optimizer-with-momentum")
    return optimizer
