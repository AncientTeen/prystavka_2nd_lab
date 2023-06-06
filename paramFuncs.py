import numpy as np
from funcs import *


def average(data):
    avr = 0
    for i in range(len(data)):
        avr += data[i]
    avr = round((avr / len(data)), 4)
    return avr


def average_sq(data):
    avr_sq = 0
    for i in range(len(data)):
        avr_sq += data[i] ** 2
    avr_sq = round((avr_sq / len(data)), 4)
    return avr_sq


def truncatedAverage(data):
    alf = 0.3
    k = round(len(data) * alf)

    trncAvr = 0
    for i in range(k, len(data) - k):
        trncAvr += data[i]
    trncAvr = round(trncAvr / (len(data) - 2 * k), 4)
    return trncAvr


def medium(data):
    if len(data) % 2 == 0:
        md = round(((data[int(len(data) / 2)] + data[int((len(data) / 2) - 1)]) / 2), 4)

    else:
        md = round((data[len(data) // 2]), 4)
    return md


def mediumWalsh(data):
    sumData = []

    for i in range(len(data) - 1):
        for j in range(len(data) - 1):
            x = (data[i] + data[j + 1]) / 2
            sumData.append(x)

    sumData = shellSort(sumData, len(sumData))
    mdWlsh = medium(sumData)
    return mdWlsh


def mediumAbsMiss(data, md):
    n = len(data)
    arr = []
    for i in range(n):
        x = abs(data[i] - md)
        arr.append(x)

    arr = shellSort(arr, len(arr))
    mdAbsMss = round(1.483 * medium(arr), 4)

    return mdAbsMss


def averageSq(data, avr):
    avrSq = 0

    for i in range(len(data)):
        avrSq += (data[i] - avr) ** 2
    avrSq = round((avrSq / (len(data) - 1)) ** (1 / 2), 4)

    return avrSq


def assymCoef(data, avr):
    shftSq = 0

    for i in range(len(data)):
        shftSq += data[i] ** 2 - avr ** 2
    shftSq = round((shftSq / (len(data))) ** (1 / 2), 4)

    sftAssmCf = 0

    for i in range(len(data)):
        sftAssmCf += (data[i] - avr) ** 3

    sftAssmCf = sftAssmCf / (len(data) * (shftSq ** 3))

    assmCf = round(((((len(data) * (len(data) - 1)) ** (1 / 2)) * sftAssmCf) / (len(data) - 2)), 4)

    return assmCf


def excessCoef(data, avr):
    shftSq = 0

    for i in range(len(data)):
        shftSq += data[i] ** 2 - avr ** 2
    shftSq = round((shftSq / (len(data))) ** (1 / 2), 4)

    shftExCf = 0
    for i in range(len(data)):
        shftExCf += (data[i] - avr) ** 4
    shftExCf = shftExCf / (len(data) * (shftSq ** 4))

    exCf = round(
        (((len(data) ** 2 - 1) / ((len(data) - 2) * (len(data) - 3))) * ((shftExCf - 3) + (6 / (len(data) + 1)))), 3)

    return exCf


def contrExcessCoef(exCf):
    cntrExCf = round((1 / ((abs(exCf)) ** (1 / 2))), 4)
    return cntrExCf


def pirsonCoef(avrSq, avr):
    if avr < 0.0001 and avr > -0.0001:
        return None
    elif avr == 0:
        return None

    prsCf = round((avrSq / avr), 4)
    return prsCf


def nonParamCoefVar(mdAbsMss, md):
    return round(mdAbsMss / md, 4)


def confInterAvr(data):
    freed = len(data)
    t = 0
    if freed < 120:
        if freed == 69:
            t = 1.995
        elif freed == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)
    sq = averageSq(data, avr)

    inf = round(avr - t * sq / (len(data) ** (1 / 2)), 4)
    sup = round(avr + t * sq / (len(data) ** (1 / 2)), 4)

    return inf, sup


def confInterSqAvr(data):
    n = len(data)

    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)
    sq = averageSq(data, avr)

    inf = round(sq - t * sq * (2 / (n - 1)) ** (1 / 4), 4)
    sup = round(sq + t * sq * (2 / (n - 1)) ** (1 / 4), 4)

    return inf, sup


def confInterAssym(data):
    n = len(data)
    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)
    assymCof = assymCoef(data, avr)

    inf = round(assymCof - t * (6 * (n - 2) / ((n + 1) * (n + 3))) ** (1 / 2), 4)
    sup = round(assymCof + t * (6 * (n - 2) / ((n + 1) * (n + 3))) ** (1 / 2), 4)

    return inf, sup


def confInterExcess(data):
    n = len(data)
    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)
    exCf = excessCoef(data, avr)

    inf = round(exCf - t * (24 * n * (n - 1) ** 2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))) ** (1 / 2), 4)
    sup = round(exCf + t * (24 * n * (n - 1) ** 2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))) ** (1 / 2), 4)

    return inf, sup


def confInterContrEx(data):
    n = len(data)
    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)

    shftSq = 0

    for i in range(n):
        shftSq += data[i] ** 2 - avr ** 2
    shftSq = round((shftSq / n) ** (1 / 2), 4)

    shftExCf = 0
    for i in range(n):
        shftExCf += (data[i] - avr) ** 4
    shftExCf = shftExCf / (n * (shftSq ** 4))

    exCf = excessCoef(data, avr)
    cntrExCf = contrExcessCoef(exCf)

    # inf = round(cntrExCf - t * (((abs(shftExCf) / (29 * n)) ** (1 / 2)) * (abs((shftExCf ** 2) - 1) ** (3 / 4))),
    #             4)
    # sup = round(cntrExCf + t * (((abs(shftExCf) / (29 * n)) ** (1 / 2)) * (abs((shftExCf ** 2) - 1) ** (3 / 4))),
    #             4)

    # inf = round(cntrExCf - t * ((abs(shftExCf) / (29 * n)) * ((abs(shftExCf ** 2 - 1)) ** (3 / 4))) ** (1 / 2), 4)
    # sup = round(cntrExCf + t * ((abs(shftExCf) / (29 * n)) * ((abs(shftExCf ** 2 - 1)) ** (3 / 4))) ** (1 / 2), 4)

    inf = round(cntrExCf - t * ((abs(shftExCf) / (29 * n)) ** (1 / 2)) * ((abs(shftExCf ** 2 - 1)) ** (3 / 4)), 4)
    sup = round(cntrExCf + t * ((abs(shftExCf) / (29 * n)) ** (1 / 2)) * ((abs(shftExCf ** 2 - 1)) ** (3 / 4)), 4)

    return inf, sup


def confInterVariation(data):
    n = len(data)
    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)
    avrSq = averageSq(data, avr)
    prsCf = pirsonCoef(avrSq, avr)
    if prsCf is None or prsCf < -10 or prsCf > 10:
        return None

    inf = round(prsCf - t * prsCf * (((1 + 2 * prsCf) / (2 * n)) ** (1 / 2)), 4)
    sup = round(prsCf + t * prsCf * (((1 + 2 * prsCf) / (2 * n)) ** (1 / 2)), 4)

    return inf, sup



