from sympy import *
import numpy as np
import math
import random
import copy


def obtain_k(L, t, index, y=0.7):
    if (index == 1):
        k_value = 2 * t / (sqrt(pi) * (2 * y * erf(L * t) - erf(L * t) + erf(2 * L * t)))
        return float(k_value)
    if (index == 2):
        k_value = t * (-sqrt(pi) * (erf(L * t) + erf(2 * L * t)) * exp(L ** 2 * t ** 2) + sqrt(
            16 * sqrt(pi) * L * t * exp(L ** 2 * t ** 2) * erf(L * t) + pi * exp(L ** 2 * t ** 2) * erf(
                L * t) ** 2 + 2 * pi * exp(L ** 2 * t ** 2) * erf(L * t) * erf(2 * L * t) + pi * exp(
                L ** 2 * t ** 2) * erf(2 * L * t) ** 2 - 16 * exp(L ** 2 * t ** 2) + 16) * exp(L ** 2 * t ** 2 / 2)) / (
                          4 * (sqrt(pi) * L * t * exp(L ** 2 * t ** 2) * erf(L * t) - exp(L ** 2 * t ** 2) + 1))
        return float(k_value)
    if (index == 3):
        k_value = (-sqrt(pi) * t ** 2 * (erf(L * t) + erf(2 * L * t)) * exp(L ** 2 * t ** 2) + t ** 1.5 * sqrt(
            16 * sqrt(pi) * L ** 2 * t ** 2 * exp(L ** 2 * t ** 2) * erf(L * t) + 16 * L * t + pi * t * exp(
                L ** 2 * t ** 2) * erf(L * t) ** 2 + 2 * pi * t * exp(L ** 2 * t ** 2) * erf(L * t) * erf(
                2 * L * t) + pi * t * exp(L ** 2 * t ** 2) * erf(2 * L * t) ** 2 - 8 * sqrt(pi) * exp(
                L ** 2 * t ** 2) * erf(L * t)) * exp(L ** 2 * t ** 2 / 2)) / (
                          4 * sqrt(pi) * L ** 2 * t ** 2 * exp(L ** 2 * t ** 2) * erf(L * t) + 4 * L * t - 2 * sqrt(
                      pi) * exp(L ** 2 * t ** 2) * erf(L * t))
        return float(k_value)
    print("k_value errors")
    return -1


def check_constraints(q, epsilon, sensitivity, index, y=0.7):
    # Check parameter
    t = q * epsilon / sensitivity
    L = sensitivity

    k = obtain_k(L, t, index, y)

    # Check general parameters
    if (L <= 0) or (q <= 0) or (y < 0) or (y > 1):
        return -1
    if (k <= 0) or (k > 1):
        # print("k_value exceeded, k = ",k)
        return -2

    # Check constraints for every index
    if (index == 1):
        if (epsilon < (3 * L ** 2 * t ** 2 + ln(y))):
            return -3
        if (epsilon < L ** 2 * t ** 2):
            return -4

    if (index == 2):
        if (k > 2 * L * t ** 2):
            return -3
        if (epsilon < 3 * L ** 2 * t ** 2):
            return -4
        if (epsilon < L ** 2 * t ** 2):
            return -5

    if (index == 3):
        if (k > t ** 2):
            return -3
        if (epsilon < 3 * L ** 2 * t ** 2):
            return -4
        if (epsilon < (ln(k * L ** 2 + 1) + L ** 2 * t ** 2)):
            return -5

    return 0


def obtain_Ex(q, epsilon, sensitivity, index, y=0.7):
    t = q * epsilon / sensitivity
    L = sensitivity
    k = obtain_k(L, t, index, y)

    if (index == 1):
        Ex = k * (exp(3 * L ** 2 * t ** 2) - 1) * exp(-4 * L ** 2 * t ** 2) / (2 * t ** 2)
        return float(Ex)
    if (index == 2):
        Ex = k * (exp(3 * L ** 2 * t ** 2) - 1) * exp(-4 * L ** 2 * t ** 2) / (2 * t ** 2)
        return float(Ex)
    if (index == 3):
        Ex = k * (exp(3 * L ** 2 * t ** 2) - 1) * exp(-4 * L ** 2 * t ** 2) / (2 * t ** 2)
        return float(Ex)
    print("Ex errors")
    return -10000


def obtain_Var(q, epsilon, sensitivity, index, y=0.7):
    t = q * epsilon / sensitivity
    L = sensitivity
    k = obtain_k(L, t, index, y)

    if (index == 1):
        Var = k * (-k * (1 - exp(3 * L ** 2 * t ** 2)) ** 2 + t * (1 - 2 * y) * (
                2 * L * t - sqrt(pi) * exp(L ** 2 * t ** 2) * erf(L * t)) * exp(7 * L ** 2 * t ** 2) - t * (
                           4 * L * t - sqrt(pi) * exp(4 * L ** 2 * t ** 2) * erf(2 * L * t)) * exp(
            4 * L ** 2 * t ** 2)) * exp(-8 * L ** 2 * t ** 2) / (4 * t ** 4)
        return float(Var)
    if (index == 2):
        Var = sqrt(pi) * L * k ** 2 * erf(L * t) / (2 * t ** 3) - L * k * exp(-L ** 2 * t ** 2) / (
                2 * t ** 2) - L * k * exp(-4 * L ** 2 * t ** 2) / t ** 2 - k ** 2 * (
                      1 - exp(3 * L ** 2 * t ** 2)) ** 2 * exp(-8 * L ** 2 * t ** 2) / (
                      4 * t ** 4) - k ** 2 / t ** 4 + k ** 2 * exp(-L ** 2 * t ** 2) / t ** 4 + sqrt(pi) * k * erf(
            L * t) / (4 * t ** 3) + sqrt(pi) * k * erf(2 * L * t) / (4 * t ** 3)
        return float(Var)
    if (index == 3):
        Var = sqrt(pi) * L ** 2 * k ** 2 * erf(L * t) / (2 * t ** 3) + 3 * L * k ** 2 * exp(-L ** 2 * t ** 2) / (
                2 * t ** 4) - L * k * exp(-L ** 2 * t ** 2) / (2 * t ** 2) - L * k * exp(
            -4 * L ** 2 * t ** 2) / t ** 2 - k ** 2 * (1 - exp(3 * L ** 2 * t ** 2)) ** 2 * exp(
            -8 * L ** 2 * t ** 2) / (4 * t ** 4) - 3 * sqrt(pi) * k ** 2 * erf(L * t) / (4 * t ** 5) + sqrt(
            pi) * k * erf(L * t) / (4 * t ** 3) + sqrt(pi) * k * erf(2 * L * t) / (4 * t ** 3)
        return float(Var)
    print("Var errors")
    return -10000


def parameter_tuning(epsilon, sensitivity, index, y=0.7):
    q_ini = 0.01
    step = 0.01

    if (index == 1):
        Var_final = 10000
        q_final = 0
        q = copy.deepcopy(q_ini)

        while (q < 5):
            if (check_constraints(q, epsilon, sensitivity, index, y) == 0):
                Var_tmp = obtain_Var(q, epsilon, sensitivity, index, y)
                if Var_tmp < Var_final:
                    Var_final = copy.deepcopy(Var_tmp)
                    q_final = copy.deepcopy(q)
            q = q + step

        return q_final, y, Var_final

    else:
        Var_final = 10000
        q_final = 0
        q = copy.deepcopy(q_ini)

        while (q < 5):
            if (check_constraints(q, epsilon, sensitivity, index) == 0):
                Var_tmp = obtain_Var(q, epsilon, sensitivity, index)
                if Var_tmp < Var_final:
                    Var_final = copy.deepcopy(Var_tmp)
                    q_final = copy.deepcopy(q)
            q = q + step

        return q_final, Var_final


def float3f(num):
    return float(format(num, '.3f'))


def float2f(num):
    return float(format(num, '.2f'))


def probability_fun(x, k, t, epsilon, sensitivity, index, y=0.7):
    L = sensitivity

    if (index == 1):
        P = 0
        if (x >= -L) and (x < L):
            P = y * k * exp(-t ** 2 * x ** 2)
        if (x >= L) and (x <= 2 * L):
            P = k * exp(-t ** 2 * x ** 2)
        return float3f(P)

    if (index == 2):
        P = 0
        if (x >= -L) and (x < 0):
            P = k * exp(-t ** 2 * x ** 2) * (k * x + k * L + 1)
        if (x >= 0) and (x < L):
            P = k * exp(-t ** 2 * x ** 2) * (-k * x + k * L + 1)
        if (x >= L) and (x <= 2 * L):
            P = k * exp(-t ** 2 * x ** 2)
        return float3f(P)

    if (index == 3):
        P = 0
        if (x >= -L) and (x < L):
            P = k * exp(-t ** 2 * x ** 2) * (-k * x ** 2 + k * L ** 2 + 1)
        if (x >= L) and (x <= 2 * L):
            P = k * exp(-t ** 2 * x ** 2)
        return float3f(P)

    print("PDF errors")
    print("x = ", x)
    print("-L =", -L)
    return -1


def generate_noise_optimized(q, epsilon, sensitivity, index, y=0.7):
    L = sensitivity

    divid = 1000
    step = 3 * L / divid
    x_count = -L
    X_axis = []
    P_axis = []
    Perturbed_list = []

    if (index == 1):
        while (x_count <= 2 * L):
            t = q * epsilon / L
            k = obtain_k(L, t, index, y)
            P_x = probability_fun(x_count, k, t, epsilon, sensitivity, index, y)
            P_axis.append(P_x)
            X_axis.append(x_count)
            x_count = x_count + step

        for i in range(len(X_axis)):
            rp = P_axis[i]
            rp = int(rp * 1000)
            for j in range(rp):
                Perturbed_list.append(X_axis[i])

        random.shuffle(Perturbed_list)
        return Perturbed_list

    else:
        while (x_count <= 2 * L):
            t = q * epsilon / L
            k = obtain_k(L, t, index)
            P_x = probability_fun(x_count, k, t, epsilon, sensitivity, index)
            P_axis.append(P_x)
            X_axis.append(x_count)
            x_count = x_count + step

        for i in range(len(X_axis)):
            rp = P_axis[i]
            rp = int(rp * 1000)
            for j in range(rp):
                Perturbed_list.append(X_axis[i])

        random.shuffle(Perturbed_list)
        return Perturbed_list


def perturbation_Q_multiple(fd, q, epsilon, sensitivity, index, volume, y=0.7):
    perturbed_result = []
    noise_array = generate_noise_optimized(q, epsilon, sensitivity, index, y)
    bias = obtain_Ex(q, epsilon, sensitivity, index, y)
    for i in range(volume):
        perturbed_result.append(fd + noise_array[i] - bias)
    return perturbed_result


def perturbation_Q_multiple_list(fd, q, epsilon, sensitivity, index, y=0.7):
    perturbed_result = []
    noise_array = generate_noise_optimized(q, epsilon, sensitivity, index, y)
    bias = obtain_Ex(q, epsilon, sensitivity, index, y)
    for i in range(len(noise_array)):
        perturbed_result.append(fd + noise_array[i] - bias)
    return perturbed_result


def gaussian_noise(epsilon, sensitivity, delta):
    """
    Generate Gaussian noise according to the Differential Privacy Gaussian Mechanism.

    :param epsilon: Privacy budget epsilon (>0)
    :param sensitivity: The sensitivity of the query/function
    :param delta: The delta parameter of differential privacy (0 < delta < 1)
    :return: A noise value drawn from the Gaussian distribution with calculated standard deviation
    """
    # Calculate the standard deviation of the Gaussian noise
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    # Generate and return the Gaussian noise
    noise = np.random.normal(0, sigma)
    return noise


def gaussian_noise_multiple(epsilon, sensitivity, delta, volume):
    noise = []
    for i in range(volume):
        noise.append(gaussian_noise(epsilon, sensitivity, delta))

    return noise

