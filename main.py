from tkinter.ttk import Notebook

from paramFuncs import *

from tkinter import *
from tkinter import filedialog as fd, simpledialog
import csv
import seaborn as sns
import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

cls = None
array = None

def classes():
    global cls
    user_input = simpledialog.askstring("Classes", "Введіть кількість класів")

    if user_input != "":
        cls = int(user_input)
    createHist(arr)
    create_distribution_function(arr)


def openFile():
    root.filename = fd.askopenfilename(initialdir="/", title="Select file", filetypes=[('All Files', '*.*'),
                                                                                       ('Python Files', '*.py'),
                                                                                       ('Text Document', '*.txt'),
                                                                                       ('CSV files', "*.csv")])
    global array
    if root.filename.split('.')[1] == 'txt':
        array = np.loadtxt(root.filename, delimiter=",", dtype='float')
        array = shellSort(array, len(array))
    elif root.filename.split('.')[1] == 'DAT':
        array = np.loadtxt(root.filename, dtype='float')
        print(array)
        arr_buff = []
        for i in range(len(array)):
            for j in range(len(array[i])):
                arr_buff.append(array[i][j])
        array = arr_buff
        array = np.asarray(array)
        array = shellSort(array, len(array))
    elif root.filename.split('.')[1] == 'csv':
        array = np.loadtxt(root.filename, dtype='float')
        arr_buff = []
        for i in range(len(array)):
            for j in range(len(array[i])):
                arr_buff.append(array[i][j])
        array = arr_buff

        array = shellSort(array, len(array))

    createHist(array)
    create_distribution_function(array)
    outputData(array)


"""models params"""


def expParams():
    user_input1 = simpledialog.askstring("N", "Введіть розмірність вибірки")
    user_input2 = simpledialog.askstring("Lambd", "Введіть параметр \u03BB")
    n = 0
    lambd = 0
    if user_input1 != "":
        n = int(user_input1)
    if user_input1 != "":
        lambd = float(user_input2)

    sampleGenExp(n, lambd)


def uniParams():
    user_input1 = simpledialog.askstring("N", "Введіть розмірність вибірки")
    user_input2 = simpledialog.askstring("a", "Введіть параметр a")
    user_input3 = simpledialog.askstring("b", "Введіть параметр b")
    n = 0
    a = 0
    b = 0
    if user_input1 != "":
        n = int(user_input1)
    if user_input2 != "":
        a = float(user_input2)
    if user_input3 != "":
        b = float(user_input3)

    sampleGenUni(n, a, b)

def weibParams():
    user_input1 = simpledialog.askstring("N", "Введіть розмірність вибірки")
    user_input2 = simpledialog.askstring("\u03B1", "Введіть параметр \u03B1")
    user_input3 = simpledialog.askstring("\u03B2", "Введіть параметр \u03B2")
    n = 0
    alf = 0
    beta = 0
    if user_input1 != "":
        n = int(user_input1)
    if user_input2 != "":
        alf = float(user_input2)
    if user_input3 != "":
        beta = float(user_input3)

    sampleGenWeib(n, alf, beta)


"""sample generation"""


def sampleGenExp(n, lambd):
    alfa = np.random.default_rng().uniform(0, 1, n)
    sample = (1 / lambd) * np.log(1 / (1 - alfa))
    # global arr
    arr = shellSort(sample, n)

    avr = average(arr)
    lambd_teta = 1 / avr
    lambd_sq = (lambd ** 2) / n

    t = 0
    l = 0
    if int(n) < 60:
        t = 1.16
        l = 0.3
    elif int(n) >= 60 and int(n) <= 200:
        t = 1.345
        l = 0.1
    elif int(n) >= 200 and int(n) <= 400:
        t = 1.44
        l = 0.075
    elif int(n) >= 400 and int(n) <= 1000:
        t = 1.645
        l = 0.05
    elif int(n) > 1000:
        t = 2.33
        l = 0.01


    tTest = (lambd - lambd_teta) / lambd_sq
    T3.delete('1.0', END)
    T3.insert(END, 'Характеристика\t\t\t' + 'INF\t\t' + 'Значення\t\t' + 'SUP\t\t' + 'SKV\n')
    T3.insert(END, 'Т-статистика: \t\t\t\t\t' + str(round(tTest, 4)) + '\n')
    T3.insert(END, '\u03BB: \t\t\t' + str(round(lambd - lambd_sq * t, 4)) + '\t\t' + str(
        round(lambd, 4)) + '\t\t' + str(round(lambd + lambd_sq * t, 4)) + '\t\t' + str(
        round(lambd_sq, 4)) + '\n')
    T3.insert(END, f'Квантиль розподілу з рівнем значущості {l} : \t' + str(t) + '\n')



    createHist(arr)
    create_distribution_function(arr)
    outputData(arr)


def sampleGenUni(n, a, b):
    alfa = np.random.default_rng().uniform(0, 1, n)
    sample = alfa / (b - a) + a

    arr = shellSort(sample, n)
    t = 0
    if int(n) < 60:
        t = 1.16
    elif int(n) >= 60 and int(n) <= 200:
        t = 1.345
    elif int(n) >= 200 and int(n) <= 400:
        t = 1.44
    elif int(n) >= 400 and int(n) <= 1000:
        t = 1.645
    elif int(n) > 1000:
        t = 2.33



    dfH11 = 1 + 3 * ((a + b) / (b - a))
    dfH12 = -(3 / (b - a))
    dfH21 = 1 - 3 * ((a + b) / (b - a))
    dfH22 = 3 / (b - a)

    Dx = ((b - a) ** 2) / (12 * n)
    covX_xx = ((a + b) * ((b - a) ** 2)) / (12 * n)
    Dx_sq = ((b - a) ** 4 + 15 * (a + b) ** 2 * (b - a) ** 2) / (180 * n)

    disp_a = dfH11 ** 2 * Dx + dfH12 ** 2 * Dx_sq + 2 * dfH11 * dfH12 * covX_xx
    disp_b = dfH21 ** 2 * Dx + dfH22 ** 2 * Dx_sq + 2 * dfH21 * dfH22 * covX_xx

    T3.delete('1.0', END)
    T3.insert(END, 'Характеристика\t\t\t' + 'INF\t\t' + 'Значення\t\t' + 'SUP\t\t' + 'SKV\n')
    T3.insert(END, 'a: \t\t\t' + str(round(a - disp_a * t, 4)) + '\t\t' + str(
        round(a, 4)) + '\t\t' + str(round(a + disp_a * t, 4)) + '\t\t' + str(
        round(disp_a, 4)) + '\n')
    T3.insert(END, 'b: \t\t\t' + str(round(b - disp_b * t, 4)) + '\t\t' + str(
        round(b, 4)) + '\t\t' + str(round(b + disp_b * t, 4)) + '\t\t' + str(
        round(disp_b, 4)) + '\n')
    T3.insert(END, 'Квантиль розподілу з рівнем значущості 5% : \t' + str(t) + '\n')

    createHist(arr)
    create_distribution_function(arr)
    outputData(arr)


def sampleGenWeib(n, alf, beta):
    alfa = np.random.default_rng().uniform(0, 1, n)
    sample = (alf * np.log(1 / (1 - alfa))) ** (1 / beta)
    arr = shellSort(sample, n)
    s_y = np.arange(1, n + 1) / n

    t = 0
    if int(n) < 60:
        t = 1.16
    elif int(n) >= 60 and int(n) <= 200:
        t = 1.345
    elif int(n) >= 200 and int(n) <= 400:
        t = 1.44
    elif int(n) >= 400 and int(n) <= 1000:
        t = 1.645
    elif int(n) > 1000:
        t = 2.33

    a11 = n - 1

    a12 = 0

    for i in range(n - 1):
        a12 += np.log(arr[i])

    a22 = 0

    for i in range(n - 1):
        a22 += np.log(arr[i]) ** 2

    b1 = 0

    for i in range(n - 1):
        b1 += np.log(np.log(1 / (1 - s_y[i])))

    b2 = 0

    for i in range(n - 1):
        b2 += np.log(arr[i]) * np.log(np.log(1 / (1 - s_y[i])))

    a_matr = [[a11, a12],

              [a12, a22]]

    b_matr = [b1, b2]

    a_matr_inv = funcReversMatr(a_matr, 2)

    cof_matr = np.dot(a_matr_inv, b_matr)




    snd_mom = centre_mom(arr, 2)
    trd_mom = centre_mom(arr, 3)
    frth_mom = centre_mom(arr, 4)

    asym = trd_mom / (snd_mom ** (3 / 2))
    exes = (frth_mom / (snd_mom ** 2)) - 3

    s_zal = 0
    for i in range(n - 1):
        s_zal += (np.log(np.log(1 / (1 - s_y[i])) - cof_matr[0] - beta * np.log(arr[i]))) ** 2
    s_zal = s_zal / (n - 3)

    disp_A = (a22 * s_zal) / (a11 * a22 - a12 * a12)
    disp_beta = (a11 * s_zal) / (a11 * a22 - a12 * a12)

    disp_alf = np.exp(-2 * cof_matr[0]) * disp_A


    T3.delete('1.0', END)

    T3.insert(END, 'Характеристика\t\t\t' + 'INF\t\t' + 'Значення\t\t' + 'SUP\t\t' + 'SKV\n')
    T3.insert(END, '\u03B1: \t\t\t' + str(round(alf - np.sqrt(disp_alf) * t, 4)) + '\t\t' + str(
        round(alf, 4)) + '\t\t' + str(round(alf + np.sqrt(disp_alf) * t, 4)) + '\t\t' + str(
        round(np.sqrt(disp_alf), 4)) + '\n')
    T3.insert(END, '\u03B2: \t\t\t' + str(round(beta - np.sqrt(disp_beta) * t, 4)) + '\t\t' + str(
        round(beta, 4)) + '\t\t' + str(round(beta + np.sqrt(disp_beta) * t, 4)) + '\t\t' + str(
        round(np.sqrt(disp_beta), 4)) + '\n')
    T3.insert(END, 'Квантиль розподілу з рівнем значущості 5% : \t' + str(t) + '\n')






    createHist(arr)
    create_distribution_function(arr)
    outputData(arr)


def experiment():
    user_input1 = simpledialog.askstring("N", "Введіть розмірність вибірки")
    user_input2 = simpledialog.askstring("Lambd", "Введіть параметр \u03BB")
    user_input3 = simpledialog.askstring("Count", "Введіть кількість експериментів")
    user_input4 = simpledialog.askstring("Filepath", "Назва файлу для збереження")
    if user_input1 != "":
        n = int(user_input1)
    if user_input1 != "":
        lambd = float(user_input2)
    if user_input3 != "":
        count = int(user_input3)
    if user_input4 != "":
        filepath = str(user_input4)

    mass = ''
    exp = ''
    for i in range(count):
        alfa = np.random.default_rng().uniform(0, 1, n)
        sample = (1 / lambd) * np.log(1 / (1 - alfa))
        sample = shellSort(sample, n)
        avr = average(sample)
        lambd_teta = 1 / avr
        lambd_sq = np.sqrt((lambd ** 2) / n)
        tTest = (lambd - lambd_teta) / lambd_sq
        mass += str(round(tTest, 4)) + '\t' + str(round(lambd_teta, 4)) + '\n'

    f = open(f"{filepath}", "w")
    f.write(mass)

def openExperiment():
    root.filename = fd.askopenfilename(initialdir="/", title="Select file", filetypes=[('All Files', '*.*'),
                                                                                       ('Python Files', '*.py'),
                                                                                       ('Text Document', '*.txt'),
                                                                                       ('CSV files', "*.csv")])


    array = []
    exp = []
    if root.filename.split('.')[1] == 'txt':
        file = np.loadtxt(root.filename, delimiter="\t", dtype='float')
        for i in range(len(file)):
            array.append(file[i][0])
            exp.append((file[i][1]))
        array = shellSort(array, len(array))
        exp = shellSort(exp, len(exp))
    elif root.filename.split('.')[1] == 'DAT':
        array = np.loadtxt(root.filename, dtype='float')
        print(array)
        arr_buff = []
        for i in range(len(array)):
            for j in range(len(array[i])):
                arr_buff.append(array[i][j])
        array = arr_buff
        array = np.asarray(array)
        array = shellSort(array, len(array))
    elif root.filename.split('.')[1] == 'csv':
        array = np.loadtxt(root.filename, dtype='float')
        arr_buff = []
        for i in range(len(array)):
            for j in range(len(array[i])):
                arr_buff.append(array[i][j])
        array = arr_buff

        array = shellSort(array, len(array))

    file = root.filename.split('/')[-1]
    n = file.split('.')[0]

    t = 0
    l = 0
    if int(n) < 60:
        t = 1.16
        l = 0.3
    elif int(n) >= 60 and int(n) <= 200:
        t = 1.345
        l = 0.1
    elif int(n) >= 200 and int(n) <= 400:
        t = 1.44
        l = 0.075
    elif int(n) >= 400 and int(n) <= 1000:
        t = 1.645
        l = 0.05
    elif int(n) > 1000:
        t = 2.33
        l = 0.01



    check = 0
    for i in range(len(array)):
        if abs(array[i]) <= t:
            check += 1

    m = average(array)
    sq = averageSq(array, m)

    lambd_avr = average(exp)
    lambd_sq = averageSq(exp, lambd_avr)

    createHist(array)
    create_distribution_function(array)
    outputData(array)

    T4.insert(END, f'{500}\t' + f'{n}\t' + f'{m}\t' + f'{sq}\t' + f'{lambd_avr}\t' + f'{lambd_sq}\t' + f'{check}\t' + f'{l}\n')







def createHist(arr=None, distrb=None, kolmogor=None, xi_val=None):
    if array is not None and arr is None:
        arr = array

    arr = np.asarray(arr)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)

    if cls:
        b = cls
    else:
        if len(arr) < 100:
            b = round((len(arr) ** (1 / 2)))
            if b % 2 == 0:
                b -= 1
        else:
            b = round((len(arr) ** (1 / 3)))
            if b % 2 == 0:
                b -= 1

    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.xlabel('Варіанти')
    plt.ylabel('Частоти')

    plt.title('Відносні частоти')

    plt.hist(arr, bins=b, edgecolor="black", color='blue', weights=np.ones_like(arr) / len(arr))

    # T3.delete('1.0', END)

    """model reproducing"""

    n = len(arr)
    avr = average(arr)
    avr_sq = average_sq(arr)

    t = 0
    if (n - 1) < 120:
        if (n - 1) == 69:
            t = 1.995
        elif (n - 1) == 24:
            t = 2.06
    else:
        t = 1.96

    if distrb == 'Експоненціальний':
        a_n = np.histogram(arr, bins=b)
        nm = max(a_n[0]) / n
        print(nm)

        lambd = 1 / avr

        y = (lambd * np.exp(-lambd * arr))

        y = y * (nm / max(y))

        print(y)

        mat_sp = 1 / lambd
        disp = 1 / (lambd ** 2)
        asym = 2
        exes = 6
        disp_lambd = (lambd ** 2) / n

        # T3.insert(END, 'Характеристики експоненціального розподілу:\n')
        #
        # T3.insert(END, 'Мат. сподівання: \t\t\t\t\t' + str(round(mat_sp, 4)) + '\n')
        # T3.insert(END, 'Дисперсія: \t\t\t\t\t' + str(round(disp, 4)) + '\n')
        # T3.insert(END, 'Коефіцієнт асиметрії: \t\t\t\t\t' + str(round(asym, 4)) + '\n')
        # T3.insert(END, 'Коефіцієнт ексцесу: \t\t\t\t\t' + str(round(exes, 4)) + '\n')
        #
        # T3.insert(END, 'Характеристика\t\t\t' + 'INF\t\t' + 'Значення\t\t' + 'SUP\t\t' + 'SKV\n')
        # T3.insert(END, '\u03BB: \t\t\t' + str(round(lambd - np.sqrt(disp_lambd) * t, 4)) + '\t\t' + str(
        #     round(lambd, 4)) + '\t\t' + str(round(lambd + np.sqrt(disp_lambd) * t, 4)) + '\t\t' + str(
        #     round(np.sqrt(disp_lambd), 4)) + '\n')

        if kolmogor:
            znach_l1 = 0
            if n > 100:
                znach_l1 = 0.05
            elif n < 30:
                znach_l1 = 0.3
            else:
                znach_l1 = 0.175

            znach_l2 = 0
            if n == 25:
                znach_l2 = 11.1

            elif n == 75:
                znach_l2 = 14.1
            elif n == 200:
                znach_l2 = 11.1
            elif n == 500:
                znach_l2 = 14.1
            else:
                znach_l2 = 12.5

            T3.insert(END, 'Критичний рівень значущості для критерію Колмогорова: ' + str(znach_l1) + '\n')
            T3.insert(END, 'Значення ймовірності узгодження Колмогорова дорівнює: ' + str(
                round(kolmogor, 4)) + ' , отже модель нормального розподілу адекватна' + '\n')
            T3.insert(END, 'Критичний рівень значущості для критерію Пірсона: ' + str(znach_l2) + '\n')
            T3.insert(END, 'Значення критерію Пірсона дорівнює: ' + str(round(xi_val, 4)) + '\n')

        plt.plot(arr, y, color='red')


    elif distrb == 'Арксінуса':
        a_n = np.histogram(arr, bins=b)
        nm1 = min(a_n[0]) / n

        nm2 = abs(max(arr, key=abs))
        nm3 = a_n[1][1] - a_n[1][0]
        a = np.sqrt(2) * np.sqrt(avr_sq - (avr ** 2))
        print(a)
        a = a + abs(a - nm2) + 0.001
        print(a)

        y = 1 / (np.pi * (np.sqrt(a ** 2 - arr ** 2)))
        y = y * (nm1 / min(y))

        mat_sp = 0
        disp = (a ** 2) / 2
        asym = 0
        exes = -1.5
        disp_a = (a ** 4) / (8 * n)
        #
        # T3.insert(END, 'Характеристики розподілу арксінуса:\n')
        #
        # T3.insert(END, 'Мат. сподівання: \t\t\t\t\t' + str(round(mat_sp, 4)) + '\n')
        # T3.insert(END, 'Дисперсія: \t\t\t\t\t' + str(disp) + '\n')
        # T3.insert(END, 'Коефіцієнт асиметрії: \t\t\t\t\t' + str(round(asym, 4)) + '\n')
        # T3.insert(END, 'Коефіцієнт ексцесу: \t\t\t\t\t' + str(round(exes, 4)) + '\n')
        #
        # T3.insert(END, 'Характеристика\t\t\t' + 'INF\t\t' + 'Значення\t\t' + 'SUP\t\t' + 'SKV\n')
        # T3.insert(END, 'a: \t\t\t' + str(round(a - np.sqrt(disp_a) * t, 4)) + '\t\t' + str(round(a, 4)) + '\t\t' + str(
        #     round(a + np.sqrt(disp_a) * t, 4)) + '\t\t' + str(round(np.sqrt(disp_a), 4)) + '\n')

        if kolmogor:
            znach_l1 = 0
            if n > 100:
                znach_l1 = 0.05
            elif n < 30:
                znach_l1 = 0.3
            else:
                znach_l1 = 0.175

            znach_l2 = 0
            if n == 25:
                znach_l2 = 11.1

            elif n == 75:
                znach_l2 = 14.1
            elif n == 200:
                znach_l2 = 11.1
            elif n == 500:
                znach_l2 = 14.1
            else:
                znach_l2 = 12.5

            # T3.insert(END, 'Критичний рівень значущості для критерію Колмогорова: ' + str(znach_l1) + '\n')
            # T3.insert(END, 'Значення ймовірності узгодження Колмогорова дорівнює: ' + str(
            #     round(kolmogor, 4)) + ' , отже модель розподілу арксінуса адекватна' + '\n')
            # T3.insert(END, 'Критичний рівень значущості для критерію Пірсона: ' + str(znach_l2) + '\n')
            # T3.insert(END, 'Значення критерію Пірсона дорівнює: ' + str(round(xi_val, 4)) + '\n')
        plt.ylim(0, (max(a_n[0]) / n) * 1.5)

        plt.plot(arr, y, color='red')


    elif distrb == 'Нормальний':
        a = np.histogram(arr, bins=b)
        nm = max(a[0]) / n
        print(nm)
        m = avr
        sq = (n / (n - 1)) * (np.sqrt(avr_sq - avr ** 2))

        y = (np.exp(-((arr - m) ** 2) / (2 * (sq ** 2)))) / (sq * np.sqrt(2 * np.pi))

        y = y * (nm / max(y))

        mat_sp = m
        disp = sq ** 2
        asym = 0
        exes = 0
        shift_exes = 3

        disp_m = (sq ** 2) / n
        disp_sq = (sq ** 2) / (2 * n)
        cov = 0
        #
        # T3.insert(END, 'Характеристики нормального розподілу:\n')
        #
        # T3.insert(END, 'Мат. сподівання: \t\t\t\t\t' + str(round(mat_sp, 4)) + '\n')
        # T3.insert(END, 'Дисперсія: \t\t\t\t\t' + str(round(disp, 4)) + '\n')
        # T3.insert(END, 'Коефіцієнт асиметрії: \t\t\t\t\t' + str(round(asym, 4)) + '\n')
        # T3.insert(END, 'Коефіцієнт незсунений ексцесу: \t\t\t\t\t' + str(round(exes, 4)) + '\n')
        # T3.insert(END, 'Коефіцієнт зсунений ексцесу: \t\t\t\t\t' + str(round(shift_exes, 4)) + '\n')
        #
        # T3.insert(END, 'Характеристика\t\t\t' + 'INF\t\t' + 'Значення\t\t' + 'SUP\t\t' + 'SKV\n')
        # T3.insert(END, 'm: \t\t\t' + str(round(m - np.sqrt(disp_m) * t, 4)) + '\t\t' + str(round(m, 4)) + '\t\t' + str(
        #     round(m + np.sqrt(disp_m) * t, 4)) + '\t\t' + str(round(np.sqrt(disp_m), 4)) + '\n')
        # T3.insert(END,
        #           '\u03C3: \t\t\t' + str(round(sq - np.sqrt(disp_sq) * t, 4)) + '\t\t' + str(round(sq, 4)) + '\t\t' + str(
        #               round(sq + np.sqrt(disp_sq) * t, 4)) + '\t\t' + str(round(np.sqrt(disp_sq), 4)) + '\n')

        if kolmogor:
            znach_l1 = 0
            if n > 100:
                znach_l1 = 0.05
            elif n < 30:
                znach_l1 = 0.3
            else:
                znach_l1 = 0.175

            znach_l2 = 0
            if n == 25:
                znach_l2 = 11.1

            elif n == 75:
                znach_l2 = 14.1
            elif n == 200:
                znach_l2 = 11.1
            elif n == 500:
                znach_l2 = 14.1
            else:
                znach_l2 = 12.5

            # T3.insert(END, 'Критичний рівень значущості для критерію Колмогорова: ' + str(znach_l1) + '\n')
            # T3.insert(END, 'Значення ймовірності узгодження Колмогорова дорівнює: ' + str(
            #     round(kolmogor, 4)) + ' , отже модель нормального розподілу адекватна' + '\n')
            # T3.insert(END, 'Критичний рівень значущості для критерію Пірсона: ' + str(znach_l2) + '\n')
            # T3.insert(END, 'Значення критерію Пірсона дорівнює: ' + str(round(xi_val, 4)) + '\n')

        plt.plot(arr, y, color='red')



    elif distrb == 'Вейбула':
        a_n = np.histogram(arr, bins=b)
        nm = max(a_n[0]) / n
        s_y = np.arange(1, n + 1) / n

        a11 = n - 1

        a12 = 0

        for i in range(n - 1):
            a12 += np.log(arr[i])

        a22 = 0

        for i in range(n - 1):
            a22 += np.log(arr[i]) ** 2

        b1 = 0

        for i in range(n - 1):
            b1 += np.log(np.log(1 / (1 - s_y[i])))

        b2 = 0

        for i in range(n - 1):
            b2 += np.log(arr[i]) * np.log(np.log(1 / (1 - s_y[i])))

        a_matr = [[a11, a12],

                  [a12, a22]]

        b_matr = [b1, b2]

        a_matr_inv = funcReversMatr(a_matr, 2)

        cof_matr = np.dot(a_matr_inv, b_matr)

        alf = np.exp(-cof_matr[0])

        beta = cof_matr[1]

        y = (beta / alf) * (arr ** (beta - 1)) * np.exp(-(arr ** beta) / alf)

        y = y * (nm / max(y))

        mat_sp = alf ** (2 / beta) * gammaFunc(1 + (1 / beta))
        disp = alf ** (2 / beta) * (gammaFunc(1 + (2 / beta)) - gammaFunc(1 + (1 / beta)) ** 2)

        snd_mom = centre_mom(arr, 2)
        trd_mom = centre_mom(arr, 3)
        frth_mom = centre_mom(arr, 4)

        asym = trd_mom / (snd_mom ** (3 / 2))
        exes = (frth_mom / (snd_mom ** 2)) - 3

        s_zal = 0
        for i in range(n - 1):
            s_zal += (np.log(np.log(1 / (1 - s_y[i])) - cof_matr[0] - beta * np.log(arr[i]))) ** 2
        s_zal = s_zal / (n - 3)

        disp_A = (a22 * s_zal) / (a11 * a22 - a12 * a12)
        disp_beta = (a11 * s_zal) / (a11 * a22 - a12 * a12)
        cov1 = -(a12 * s_zal) / (a11 * a22 - a12 * a12)

        disp_alf = np.exp(-2 * cof_matr[0]) * disp_A
        cov2 = -np.exp(cof_matr[0]) * cov1

        # T3.insert(END, 'Характеристики розподілу Вейбулла:\n')
        #
        # T3.insert(END, 'Мат. сподівання: \t\t\t\t\t' + str(round(mat_sp, 4)) + '\n')
        # T3.insert(END, 'Дисперсія: \t\t\t\t\t' + str(round(disp, 4)) + '\n')
        # T3.insert(END, 'Коефіцієнт асиметрії: \t\t\t\t\t' + str(round(asym, 4)) + '\n')
        # T3.insert(END, 'Коефіцієнт незсунений ексцесу: \t\t\t\t\t' + str(round(exes, 4)) + '\n')
        #
        # T3.insert(END, 'Характеристика\t\t\t' + 'INF\t\t' + 'Значення\t\t' + 'SUP\t\t' + 'SKV\n')
        # T3.insert(END, '\u03B1: \t\t\t' + str(round(alf - np.sqrt(disp_alf) * t, 4)) + '\t\t' + str(
        #     round(alf, 4)) + '\t\t' + str(round(alf + np.sqrt(disp_alf) * t, 4)) + '\t\t' + str(
        #     round(np.sqrt(disp_alf), 4)) + '\n')
        # T3.insert(END, '\u03B2: \t\t\t' + str(round(beta - np.sqrt(disp_beta) * t, 4)) + '\t\t' + str(
        #     round(beta, 4)) + '\t\t' + str(round(beta + np.sqrt(disp_beta) * t, 4)) + '\t\t' + str(
        #     round(np.sqrt(disp_beta), 4)) + '\n')

        if kolmogor:
            znach_l1 = 0
            if n > 100:
                znach_l1 = 0.05
            elif n < 30:
                znach_l1 = 0.3
            else:
                znach_l1 = 0.175

            znach_l2 = 0
            if n == 25:
                znach_l2 = 11.1

            elif n == 75:
                znach_l2 = 14.1
            elif n == 200:
                znach_l2 = 11.1
            elif n == 500:
                znach_l2 = 14.1
            else:
                znach_l2 = 12.5

            # T3.insert(END, 'Критичний рівень значущості для критерію Колмогорова: ' + str(znach_l1) + '\n')
            # T3.insert(END, 'Значення ймовірності узгодження Колмогорова дорівнює: ' + str(
            #     round(kolmogor, 4)) + ' , отже модель розподілу Вейбулла адекватна' + '\n')
            # T3.insert(END, 'Критичний рівень значущості для критерію Пірсона: ' + str(znach_l2) + '\n')
            # T3.insert(END, 'Значення критерію Пірсона дорівнює: ' + str(round(xi_val, 4)) + '\n')
        plt.plot(arr, y, color='red')




    elif distrb == 'Рівномірний':
        a_n = np.histogram(arr, bins=b)
        nm = sum(a_n[0]) / (n * b)

        a = avr - np.sqrt(3 * (avr_sq - avr ** 2))
        b = avr + np.sqrt(3 * (avr_sq - avr ** 2))

        y = [0 for i in range(n)]

        i = 0
        while i < n:
            y[i] = 1 / (b - a)
            i += 1
        y = np.asarray(y)
        y = y * (nm / max(y))

        mat_sp = (a + b) / 2
        disp = ((b - a) ** 2) / 12

        asym = 0
        exes = -1.2

        dfH11 = 1 + 3 * ((a + b) / (b - a))
        dfH12 = -(3 / (b - a))
        dfH21 = 1 - 3 * ((a + b) / (b - a))
        dfH22 = 3 / (b - a)

        Dx = ((b - a) ** 2) / (12 * n)
        covX_xx = ((a + b) * ((b - a) ** 2)) / (12 * n)
        Dx_sq = ((b - a) ** 4 + 15 * (a + b) ** 2 * (b - a) ** 2) / (180 * n)

        disp_a = dfH11 ** 2 * Dx + dfH12 ** 2 * Dx_sq + 2 * dfH11 * dfH12 * covX_xx
        disp_b = dfH21 ** 2 * Dx + dfH22 ** 2 * Dx_sq + 2 * dfH21 * dfH22 * covX_xx
        cov_a_b = dfH11 * dfH21 * Dx + dfH12 * dfH22 * Dx_sq + (dfH11 * dfH22 + dfH12 * dfH21) * covX_xx

        # T3.insert(END, 'Характеристики рівномірного розподілу:\n')
        #
        # T3.insert(END, 'Мат. сподівання: \t\t\t\t\t' + str(round(mat_sp, 4)) + '\n')
        # T3.insert(END, 'Дисперсія: \t\t\t\t\t' + str(round(disp, 4)) + '\n')
        # T3.insert(END, 'Коефіцієнт асиметрії: \t\t\t\t\t' + str(round(asym, 4)) + '\n')
        # T3.insert(END, 'Коефіцієнт незсунений ексцесу: \t\t\t\t\t' + str(round(exes, 4)) + '\n')
        #
        # T3.insert(END, 'D{a}: \t\t\t\t\t' + str(disp_a) + '\n')
        # T3.insert(END, 'D{b}: \t\t\t\t\t' + str(disp_b) + '\n')
        #
        # T3.insert(END, 'a: \t\t\t' + str(round(a - np.sqrt(disp_a) * t, 4)) + '\t\t' + str(
        #     round(a, 4)) + '\t\t' + str(round(a + np.sqrt(disp_a) * t, 4)) + '\t\t' + str(
        #     round(np.sqrt(disp_a), 4)) + '\n')
        # T3.insert(END, 'b: \t\t\t' + str(round(b - np.sqrt(disp_b) * t, 4)) + '\t\t' + str(
        #     round(b, 4)) + '\t\t' + str(round(b + np.sqrt(disp_b) * t, 4)) + '\t\t' + str(
        #     round(np.sqrt(disp_b), 4)) + '\n')

        if kolmogor:
            znach_l1 = 0
            if n > 100:
                znach_l1 = 0.05
            elif n < 30:
                znach_l1 = 0.3
            else:
                znach_l1 = 0.175

            znach_l2 = 0
            if n == 25:
                znach_l2 = 11.1

            elif n == 75:
                znach_l2 = 14.1
            elif n == 200:
                znach_l2 = 11.1
            elif n == 500:
                znach_l2 = 14.1
            else:
                znach_l2 = 12.5
            #
            # T3.insert(END, 'Критичний рівень значущості для критерію Колмогорова: ' + str(znach_l1) + '\n')
            # T3.insert(END, 'Значення ймовірності узгодження Колмогорова дорівнює: ' + str(
            #     round(kolmogor, 4)) + ' , отже модель рівномірного розподілу адекватна' + '\n')
            # T3.insert(END, 'Критичний рівень значущості для критерію Пірсона: ' + str(znach_l2) + '\n')
            # T3.insert(END, 'Значення критерію Пірсона дорівнює: ' + str(round(xi_val, 4)) + '\n')

        plt.plot(arr, y, color='red')

    hist = FigureCanvasTkAgg(fig, master=root)
    hist.get_tk_widget().grid(row=0, column=0)
    toolbar = NavigationToolbar2Tk(hist, root, pack_toolbar=False)
    toolbar.update()
    toolbar.grid(row=1, column=0)


def create_distribution_function(arr=None, distrb=None):
    # global arr

    if array is not None:
        arr = array

    arr = np.asarray(arr)

    fig, ax = plt.subplots(figsize=(5, 4))

    plt.grid(color='grey', linestyle='--', linewidth=0.5)


    n = len(arr)

    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    if cls:
        b = cls
    else:
        if n < 100:
            b = round((n ** (1 / 2)))
        else:
            b = round((n ** (1 / 3)))

    s_y = np.arange(1, n + 1) / n
    ax.scatter(x=arr, y=s_y, s=5)
    sns.histplot(arr, element="step", fill=False,
                 cumulative=True, stat="density", common_norm=False, bins=b, color='red')

    """model reproducing"""

    n = len(arr)
    avr = average(arr)
    avr_sq = average_sq(arr)

    if distrb == 'Експоненціальний':

        lambd = 1 / avr
        conf_inter = 0
        for i in range(n):
            conf_inter += (lambd * np.exp(-lambd * arr[i])) ** 2 * (
                    (arr[i] ** 2 * np.exp(-2 * lambd * arr[i]) * lambd ** 2) / n)

        y = exp_distr(lambd, arr)
        y_up = exp_up(lambd, arr)
        y_low = exp_low(lambd, arr)

        plt.plot(arr, y, label='Теоретична функція розподілу')
        plt.plot(arr, y_up, label='Верхня межа')
        plt.plot(arr, y_low, label='Нижня межа')

        plt.legend()


    elif distrb == 'Арксінуса':

        nm2 = abs(max(arr, key=abs))
        a = np.sqrt(2) * np.sqrt(avr_sq - (avr ** 2))
        a = a + abs(a - nm2) + 0.001
        disp_a = (a ** 4) / (8 * n)

        arr_a = np.clip(arr / a, -1, 1)

        y = (1 / 2) + np.arcsin(arr_a) / np.pi

        plt.ylim(0, 1.1)

        plt.plot(arr, y, label='Теоретична функція розподілу', color='red')

        plt.legend()


    elif distrb == 'Нормальний':

        m = avr
        sq = (n / (n - 1)) * (np.sqrt(avr_sq - avr ** 2))

        y = [0 for i in range(n)]
        y_up = [0 for i in range(n)]
        y_low = [0 for i in range(n)]

        disp_m = (sq ** 2) / n
        disp_sq = (sq ** 2) / (2 * n)
        df_m = -np.exp(-(arr - m) ** 2 / (2 * sq ** 2)) / (sq * np.sqrt(2 * np.pi))
        df_sq = (-(arr - m) * np.exp(-(arr - m) ** 2 / (2 * sq ** 2))) / (sq ** 2 * np.sqrt(2 * np.pi))

        dF = (-np.exp(-(arr - m) ** 2 / (2 * sq ** 2)) / (sq * np.sqrt(2 * np.pi))) ** 2 * disp_m + (
                (-(arr - m) * np.exp(-(arr - m) ** 2 / (2 * sq ** 2))) / (
                sq ** 2 * np.sqrt(2 * np.pi))) ** 2 * disp_sq

        for i in range(n):
            y[i] = norm_distr(m, sq, arr[i])
            y_up[i] = norm_distr(m, sq, arr[i]) + np.sqrt(((-np.exp(-(arr[i] - m) ** 2 / (2 * sq ** 2)) / (
                    sq * np.sqrt(2 * np.pi))) ** 2 * disp_m + ((-(arr[i] - m) * np.exp(
                -(arr[i] - m) ** 2 / (2 * sq ** 2))) / (sq ** 2 * np.sqrt(2 * np.pi))) ** 2 * disp_sq)) * 1.96

            y_low[i] = norm_distr(m, sq, arr[i]) - np.sqrt(((-np.exp(-(arr[i] - m) ** 2 / (2 * sq ** 2)) / (
                    sq * np.sqrt(2 * np.pi))) ** 2 * disp_m + ((-(arr[i] - m) * np.exp(
                -(arr[i] - m) ** 2 / (2 * sq ** 2))) / (sq ** 2 * np.sqrt(2 * np.pi))) ** 2 * disp_sq)) * 1.96

        print(y)

        plt.plot(arr, y, label='Теоретична функція розподілу')
        plt.plot(arr, y_up, label='Верхня межа')
        plt.plot(arr, y_low, label='Нижня межа')

        plt.legend()

    elif distrb == 'Вейбула':

        a_n = np.histogram(arr, bins=b)
        nm = max(a_n[0]) / n
        s_y = np.arange(1, n + 1) / n

        a11 = n - 1

        a12 = 0

        for i in range(n - 1):
            a12 += np.log(arr[i])

        a22 = 0

        for i in range(n - 1):
            a22 += np.log(arr[i]) ** 2

        b1 = 0

        for i in range(n - 1):
            b1 += np.log(np.log(1 / (1 - s_y[i])))

        b2 = 0

        for i in range(n - 1):
            b2 += np.log(arr[i]) * np.log(np.log(1 / (1 - s_y[i])))

        a_matr = [[a11, a12],

                  [a12, a22]]

        b_matr = [b1, b2]

        a_matr_inv = funcReversMatr(a_matr, 2)

        cof_matr = np.dot(a_matr_inv, b_matr)

        alf = np.exp(-cof_matr[0])

        beta = cof_matr[1]

        s_zal = 0
        for i in range(n - 1):
            s_zal += (np.log(np.log(1 / (1 - s_y[i])) - cof_matr[0] - beta * np.log(arr[i]))) ** 2
        s_zal = s_zal / (n - 3)

        disp_A = (a22 * s_zal) / (a11 * a22 - a12 * a12)
        disp_beta = (a11 * s_zal) / (a11 * a22 - a12 * a12)
        cov1 = -(a12 * s_zal) / (a11 * a22 - a12 * a12)

        disp_alf = np.exp(-2 * cof_matr[0]) * disp_A
        cov2 = -np.exp(cof_matr[0]) * cov1

        y = weib_distr(alf, beta, arr)

        # y_up = weib_distr(alf, beta, arr) + conf_inter
        # y_low = weib_distr(alf, beta, arr) - conf_inter

        # dF = (((-arr ** beta) / (alf ** 2)) * np.exp((((-arr ** beta) / alf)))) ** 2 * disp_alf + (
        #             ((arr ** beta) / alf) * np.log(arr) * np.exp((((-arr ** beta) / alf)))) ** 2 * disp_beta + 2 * (
        #         ((((-arr ** beta) / (alf ** 2)) * np.exp((((-arr ** beta) / alf)))) * ((arr ** beta) / alf) * np.log(arr) * np.exp((((-arr ** beta) / alf)))) * cov2)

        y_up = 1 - np.exp(-(arr ** beta) / alf) + np.sqrt(
            (((-arr ** beta) / (alf ** 2)) * np.exp((((-arr ** beta) / alf)))) ** 2 * disp_alf + (
                    ((arr ** beta) / alf) * np.log(arr) * np.exp((((-arr ** beta) / alf)))) ** 2 * disp_beta + 2 * (
                    ((((-arr ** beta) / (alf ** 2)) * np.exp((((-arr ** beta) / alf)))) * (
                            (arr ** beta) / alf) * np.log(arr) * np.exp((((-arr ** beta) / alf)))) * cov2)) * 1.96
        y_low = 1 - np.exp(-(arr ** beta) / alf) - np.sqrt(
            (((-arr ** beta) / (alf ** 2)) * np.exp((((-arr ** beta) / alf)))) ** 2 * disp_alf + (
                    ((arr ** beta) / alf) * np.log(arr) * np.exp((((-arr ** beta) / alf)))) ** 2 * disp_beta + 2 * (
                    ((((-arr ** beta) / (alf ** 2)) * np.exp((((-arr ** beta) / alf)))) * (
                            (arr ** beta) / alf) * np.log(arr) * np.exp((((-arr ** beta) / alf)))) * cov2)) * 1.96

        plt.plot(arr, y, label='Теоретична функція розподілу')
        plt.plot(arr, y_up, label='Верхня межа')
        plt.plot(arr, y_low, label='Нижня межа')

        plt.legend()


    elif distrb == 'Рівномірний':

        a = avr - np.sqrt(3 * (avr_sq - avr ** 2))
        b = avr + np.sqrt(3 * (avr_sq - avr ** 2))

        dfH11 = 1 + 3 * ((a + b) / (b - a))
        dfH12 = -(3 / (b - a))
        dfH21 = 1 - 3 * ((a + b) / (b - a))
        dfH22 = (3 / (b - a))

        Dx = ((b - a) ** 2) / (12 * n)
        covX_xx = ((a + b) * ((b - a) ** 2)) / (12 * n)
        Dx_sq = ((b - a) ** 4 + 15 * (a + b) ** 2 * (b - a) ** 2) / (180 * n)

        disp_a = dfH11 ** 2 * Dx + dfH12 ** 2 * Dx_sq + 2 * dfH11 * dfH12 * covX_xx
        disp_b = dfH21 ** 2 * Dx + dfH22 ** 2 * Dx_sq + 2 * dfH21 * dfH22 * covX_xx
        cov_a_b = dfH11 * dfH21 * Dx + dfH12 * dfH22 * Dx_sq + (dfH11 * dfH22 + dfH12 * dfH21) * covX_xx

        y = [0 for i in range(n)]

        i = 0

        while i < n:
            if arr[i] >= a and arr[i] < b:
                y[i] = uni_distr(a, b, arr[i])
            elif arr[i] >= b:
                y[i] = 1
            i += 1

        y_up = [0 for j in range(n)]
        y_low = [0 for j in range(n)]
        for k in range(n):
            y_up[k] = y[k] + np.sqrt(
                ((arr[k] - b) ** 2 / ((b - a) ** 4)) * disp_a + ((arr[k] - a) ** 2 / ((b - a) ** 4)) * disp_b - 2 * (
                        ((arr[k] - a) * (arr[k] - b)) / ((b - a) ** 4)) * cov_a_b) * 1.96
            y_low[k] = y[k] - np.sqrt(
                ((arr[k] - b) ** 2 / ((b - a) ** 4)) * disp_a + ((arr[k] - a) ** 2 / ((b - a) ** 4)) * disp_b - 2 * (
                        ((arr[k] - a) * (arr[k] - b)) / ((b - a) ** 4)) * cov_a_b) * 1.96
        plt.plot(arr, y, label='Теоретична функція розподілу')
        plt.plot(arr, y_up, label='Верхня межа')
        plt.plot(arr, y_low, label='Нижня межа')

        plt.legend()

    plt.title('Функція розподілу')
    plt.xlabel('')
    plt.ylabel('')
    distr_func = FigureCanvasTkAgg(fig, master=root)
    distr_func.get_tk_widget().grid(row=0, column=2, columnspan=2)
    toolbar = NavigationToolbar2Tk(distr_func, root, pack_toolbar=False)
    toolbar.update()
    toolbar.grid(row=1, column=2, columnspan=2)


def outputData(arr):
    T1.delete('1.0', END)
    T2.delete('1.0', END)
    # tabControl = Notebook(root)

    # tab1 = Frame(tabControl)
    # tab2 = Frame(tabControl)
    # tabControl.add(tab1, text='Об\'єкти')
    # tabControl.add(tab2, text='Протокол')
    # tabControl.grid(row=4, column=0)
    #
    # # for c in range(3): tab1.columnconfigure(index=c, weight=1)
    # # for r in range(4): tab1.rowconfigure(index=r, weight=1)
    #
    # T1 = Text(master=tab1, height=10, width=100)
    #
    # T1.grid(row=11, column=0)
    #
    T1.insert(END, arr)
    #
    # T2 = Text(master=tab2, height=10, width=100)
    #
    # T2.grid(row=11, column=0)

    n = len(arr)
    T2.insert(END, 'Характеристика\t\t\t' + 'INF\t\t' + 'Значення\t\t' + 'SUP\t\t' + 'SKV\n')

    avr = average(arr)
    sq = averageSq(arr, avr)
    avrIntr = confInterAvr(arr)
    T2.insert(END,
              'Середнє значення: \t\t\t' + str(avrIntr[0]) + '\t\t' + str(avr) + '\t\t' + str(avrIntr[1]) + '\t\t' +
              str(round(sq / (len(arr) ** (1 / 2)), 4)) + '\n')

    md = medium(arr)
    T2.insert(END, 'Медіана: \t\t\t' + str(md) + '\t\t' + str(md) + '\t\t' + str(md) + '\t\t' + '0.0000\n')

    avrSq = averageSq(arr, avr)
    sqIntr = confInterSqAvr(arr)
    T2.insert(END,
              'Сер. квадратичне: \t\t\t' + str(sqIntr[0]) + '\t\t' + str(avrSq) + '\t\t' + str(sqIntr[1]) + '\t\t' +
              str(round(sq * (2 / (n - 1)) ** (1 / 4), 4)) + '\n')

    assmCf = assymCoef(arr, avr)
    assmIntrCof = confInterAssym(arr)
    T2.insert(END, 'Коефіцієнт асиметрії: \t\t\t' + str(assmIntrCof[0]) + '\t\t' + str(assmCf) + '\t\t' + str(
        assmIntrCof[1]) + '\t\t' +
              str(round((6 * (n - 2) / ((n + 1) * (n + 3))) ** (1 / 2), 4)) + '\n')

    exCf = excessCoef(arr, avr)
    exIntrCof = confInterExcess(arr)
    T2.insert(END, 'Коефіцієнт ексцесу: \t\t\t' + str(exIntrCof[0]) + '\t\t' + str(exCf) + '\t\t' + str(
        exIntrCof[1]) + '\t\t' +
              str(round((24 * n * (n - 1) ** 2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))) ** (1 / 2), 4)) + '\n')

    shftSq = 0

    for i in range(n):
        shftSq += arr[i] ** 2 - avr ** 2
    shftSq = round((shftSq / n) ** (1 / 2), 4)

    shftExCf = 0
    for i in range(n):
        shftExCf += (arr[i] - avr) ** 4
    shftExCf = shftExCf / (n * (shftSq ** 4))

    cntrExCf = contrExcessCoef(exCf)
    cExIntrCof = confInterContrEx(arr)

    T2.insert(END, 'Коефіцієнт контрексцесу: \t\t\t' + str(cExIntrCof[0]) + '\t\t' + str(cntrExCf) + '\t\t' + str(
        cExIntrCof[1]) + '\t\t' +
              str(round(((abs(shftExCf) / (29 * n)) ** (1 / 2)) * ((abs(shftExCf ** 2 - 1)) ** (3 / 4)), 4)) + '\n')

    prsCf = pirsonCoef(avrSq, avr)
    variatIntrCof = confInterVariation(arr)
    if prsCf is None or prsCf < 10 or prsCf > 10:
        T2.insert(END, 'Коефіцієнт Варіації: \t\t\t\t\t' + str(prsCf) + '\n')
    else:
        T2.insert(END, 'Коефіцієнт Варіації: \t\t\t' + str(variatIntrCof[0]) + '\t\t' + str(prsCf) + '\t\t' + str(
            variatIntrCof[1]) + '\t\t' +
                  str(round(prsCf * (((1 + 2 * prsCf) / (2 * n)) ** (1 / 2)), 4)) + '\n')

    trcnAvr = truncatedAverage(arr)
    T2.insert(END, 'Усічене середнє: \t\t\t\t\t' + str(trcnAvr) + '\n')

    mdWlsh = mediumWalsh(arr)
    T2.insert(END, 'Медіана Уолша: \t\t\t\t\t' + str(mdWlsh) + '\n')

    mdAbsMss = mediumAbsMiss(arr, md)
    T2.insert(END, 'Медіана абс. відхилень: \t\t\t\t\t' + str(mdAbsMss) + '\n')

    nonParamCfVar = nonParamCoefVar(mdAbsMss, md)
    T2.insert(END, 'Непарам. коеф. варіації: \t\t\t\t\t' + str(nonParamCfVar) + '\n')

    T2.insert(END, '-----------------------------\n')
    T2.insert(END, 'Квантилі : \n')
    T2.insert(END, '0.05: \t' + str(round(np.quantile(arr, 0.05), 3)) + '\n')
    T2.insert(END, '0.1: \t' + str(round(np.quantile(arr, 0.1), 3)) + '\n')
    T2.insert(END, '0.25: \t' + str(round(np.quantile(arr, 0.25), 3)) + '\n')
    T2.insert(END, '0.5: \t' + str(round(np.quantile(arr, 0.5), 3)) + '\n')
    T2.insert(END, '0.75: \t' + str(round(np.quantile(arr, 0.75), 3)) + '\n')
    T2.insert(END, '0.9: \t' + str(round(np.quantile(arr, 0.9), 3)) + '\n')
    T2.insert(END, '0.95: \t' + str(round(np.quantile(arr, 0.95), 3)) + '\n')


def norm():
    createHist(distrb="Нормальний")
    create_distribution_function(distrb="Нормальний")


def exp():
    createHist(distrb="Експоненціальний")
    create_distribution_function(distrb="Експоненціальний")


def ravn():
    createHist(distrb="Рівномірний")
    create_distribution_function(distrb="Рівномірний")


def veib():
    createHist(distrb="Вейбула")
    create_distribution_function(distrb="Вейбула")


def arcsin():
    createHist(distrb="Арксінуса")
    create_distribution_function(distrb="Арксінуса")


def univers():
    n = len(arr)
    if n < 100:
        b = round((n ** (1 / 2)))
    else:
        b = round((n ** (1 / 3)))

    avr = average(arr)
    avr_sq = average_sq(arr)
    conf_inter = 1.36 / np.sqrt(n)
    a_n = np.histogram(arr, bins=b)
    nm = max(a_n[0]) / n
    s_y = np.arange(1, n + 1) / n

    """exponential distribution check"""
    arr_exp = np.asarray(arr)

    lambd = 1 / avr
    y = 1 - np.exp(-lambd * arr_exp)

    dPlus = 0
    for i in range(n):
        if dPlus < abs(s_y[i] - y[i]):
            dPlus = abs(s_y[i] - y[i])

    dMinus = 0
    for i in range(1, n):
        if dMinus < abs(s_y[i] - y[i - 1]):
            dMinus = abs(s_y[i] - y[i - 1])

    zet = np.sqrt(n) * max(dPlus, dMinus)

    pZet_exp = 1 - K_zet(zet, n)

    """arcsine distribution check"""

    arr_max = arr[0]
    arr_min = arr[-1]

    y = -(2 * np.arcsin(np.sqrt((arr - arr_min) / (arr_max - arr_min)))) / np.pi + 1

    dPlus = 0
    for i in range(n):
        if dPlus < abs(s_y[i] - y[i]):
            dPlus = abs(s_y[i] - y[i])

    dMinus = 0
    for i in range(1, n):
        if dMinus < abs(s_y[i] - y[i - 1]):
            dMinus = abs(s_y[i] - y[i - 1])

    zet = np.sqrt(n) * max(dPlus, dMinus)

    pZet_arc = 1 - K_zet(zet, n)

    """normal distribution check"""
    m = avr
    sq = (n / (n - 1)) * (np.sqrt(avr_sq - avr ** 2))

    y = [0 for i in range(n)]
    for i in range(n):
        y[i] = 0.5 * (1 + erf(((array[i] - m) / (np.sqrt(2) * sq))))

    dPlus = 0
    for i in range(n):
        if dPlus < abs(s_y[i] - y[i]):
            dPlus = abs(s_y[i] - y[i])

    dMinus = 0
    for i in range(1, n):
        if dMinus < abs(s_y[i] - y[i - 1]):
            dMinus = abs(s_y[i] - y[i - 1])

    zet = np.sqrt(n) * max(dPlus, dMinus)

    pZet_norm = 1 - K_zet(zet, n)

    """uniform distribution check"""
    a = avr - ((3 * (avr_sq - avr ** 2)) ** (1 / 2))
    b = avr + ((3 * (avr_sq - avr ** 2)) ** (1 / 2))

    y = [0 for i in range(n)]

    i = 0

    while i < n:
        if array[i] >= a and array[i] < b:
            y[i] = (array[i] - a) / (b - a)
        elif array[i] >= b:
            y[i] = 1
        i += 1

    dPlus = 0
    for i in range(n):
        if dPlus < abs(s_y[i] - y[i]):
            dPlus = abs(s_y[i] - y[i])

    dMinus = 0
    for i in range(1, n):
        if dMinus < abs(s_y[i] - y[i - 1]):
            dMinus = abs(s_y[i] - y[i - 1])

    zet = np.sqrt(n) * max(dPlus, dMinus)

    pZet_uni = 1 - K_zet(zet, n)

    """weibull distribution check"""
    a11 = n - 1

    a12 = 0

    for i in range(n - 1):
        a12 += np.log(array[i])

    a22 = 0

    for i in range(n - 1):
        a22 += np.log(array[i]) ** 2

    b1 = 0

    for i in range(n - 1):
        b1 += np.log(np.log(1 / (1 - s_y[i])))

    b2 = 0

    for i in range(n - 1):
        b2 += np.log(array[i]) * np.log(np.log(1 / (1 - s_y[i])))

    a_matr = [[a11, a12],

              [a12, a22]]

    b_matr = [b1, b2]

    a_matr_inv = funcReversMatr(a_matr, 2)

    cof_matr = np.dot(a_matr_inv, b_matr)

    alf = np.exp(-cof_matr[0])

    beta = cof_matr[1]

    y = 1 - np.exp(-(array ** beta) / alf)

    dPlus = 0
    for i in range(n):
        if dPlus < abs(s_y[i] - y[i]):
            dPlus = abs(s_y[i] - y[i])

    dMinus = 0
    for i in range(1, n):
        if dMinus < abs(s_y[i] - y[i - 1]):
            dMinus = abs(s_y[i] - y[i - 1])

    zet = np.sqrt(n) * max(dPlus, dMinus)

    pZet_weib = 1 - K_zet(zet, n)

    print(pZet_weib, ' weib')
    print(pZet_arc, ' arc')
    print(pZet_norm, ' norm')
    print(pZet_uni, ' uni')
    print(pZet_exp, ' exp')

    max_distr = max(pZet_weib, pZet_arc, pZet_norm, pZet_uni, pZet_exp)

    xi_pirs = pirson()

    if max_distr == pZet_weib:
        createHist(distrb="Вейбула", kolmogor=pZet_weib, xi_val=xi_pirs[4])
        create_distribution_function("Вейбула")

    elif max_distr == pZet_arc:
        createHist(distrb="Арксінуса", kolmogor=pZet_arc, xi_val=xi_pirs[1])
        create_distribution_function("Арксінуса")

    elif max_distr == pZet_norm:
        createHist(distrb="Нормальний", kolmogor=pZet_norm, xi_val=xi_pirs[3])
        create_distribution_function("Нормальний")

    elif max_distr == pZet_uni:
        createHist(distrb="Рівномірний", kolmogor=pZet_uni, xi_val=xi_pirs[0])
        create_distribution_function("Рівномірний")

    elif max_distr == pZet_exp:
        createHist(distrb="Експоненціальний", kolmogor=pZet_exp, xi_val=xi_pirs[2])
        create_distribution_function(distrb="Експоненціальний")


def pirson():
    n = len(arr)
    if n < 100:
        b = round((n ** (1 / 2)))
    else:
        b = round((n ** (1 / 3)))

    hst = np.histogram(arr, bins=b)

    avr = average(arr)
    avr_sq = average_sq(arr)
    s_y = np.arange(1, n + 1) / n

    """exponential distribution check"""
    lambd = 1 / avr

    xi_exp = 0

    for i in range(b):
        xi_exp += ((((hst[0][i] - n * (
                exp_distr(lambd, hst[1][i + 1]) - exp_distr(lambd, hst[1][i])))) ** 2) / (n * (
                exp_distr(lambd, hst[1][i + 1]) - exp_distr(lambd, hst[1][i]))))

    """arcsine distribution check"""

    arr_max = hst[1][0]
    arr_min = hst[1][-1]

    xi_arc = 0

    for i in range(b):
        #
        xi_arc += (((hst[0][i] - n * (
                (-(2 * np.arcsin(np.sqrt((hst[1][i + 1] - arr_min) / (arr_max - arr_min)))) / np.pi + 1) - (
                -(2 * np.arcsin(np.sqrt((hst[1][i] - arr_min) / (arr_max - arr_min)))) / np.pi + 1))) ** 2) /
                   (n * ((-(2 * np.arcsin(np.sqrt((hst[1][i + 1] - arr_min) / (arr_max - arr_min)))) / np.pi + 1) - (
                           -(2 * np.arcsin(np.sqrt((hst[1][i] - arr_min) / (arr_max - arr_min)))) / np.pi + 1))))

    """normal distribution check"""
    m = avr
    sq = (n / (n - 1)) * (np.sqrt(avr_sq - avr ** 2))
    xi_norm = 0

    for i in range(b):
        xi_norm += (((hst[0][i] - n * (norm_distr(m, sq, hst[1][i + 1]) - (norm_distr(m, sq, hst[1][i])))) ** 2) /
                    (n * (norm_distr(m, sq, hst[1][i + 1]) - (norm_distr(m, sq, hst[1][i])))))

    """uniform distribution check"""
    a = avr - ((3 * (avr_sq - avr ** 2)) ** (1 / 2))
    b_d = avr + ((3 * (avr_sq - avr ** 2)) ** (1 / 2))

    xi_uni = 0

    for i in range(b):
        xi_uni += (((hst[0][i] - n * (uni_distr(a, b_d, hst[1][i + 1]) - uni_distr(a, b_d, hst[1][i]))) ** 2) /
                   (n * (uni_distr(a, b_d, hst[1][i + 1]) - uni_distr(a, b_d, hst[1][i]))))

    """weibull distribution check"""
    a11 = n - 1

    a12 = 0

    for i in range(n - 1):
        a12 += np.log(array[i])

    a22 = 0

    for i in range(n - 1):
        a22 += np.log(array[i]) ** 2

    b1 = 0

    for i in range(n - 1):
        b1 += np.log(np.log(1 / (1 - s_y[i])))

    b2 = 0

    for i in range(n - 1):
        b2 += np.log(array[i]) * np.log(np.log(1 / (1 - s_y[i])))

    a_matr = [[a11, a12],

              [a12, a22]]

    b_matr = [b1, b2]

    a_matr_inv = funcReversMatr(a_matr, 2)

    cof_matr = np.dot(a_matr_inv, b_matr)

    alf = np.exp(-cof_matr[0])

    beta = cof_matr[1]

    # y = 1 - np.exp(-(array ** beta) / alf)

    xi_weib = 0

    for i in range(b):
        xi_weib += (((hst[0][i] - n * (weib_distr(alf, beta, hst[1][i + 1]) - weib_distr(alf, beta, hst[1][i]))) ** 2) /
                    (n * (weib_distr(alf, beta, hst[1][i + 1]) - weib_distr(alf, beta, hst[1][i]))))

    print(xi_uni, 'uni')
    print(xi_arc, 'arc')
    print(xi_exp, 'exp')
    print(xi_norm, 'norm')
    print(xi_weib, 'weib')

    xi_mass = [xi_uni, xi_arc, xi_exp, xi_norm, xi_weib]
    return xi_mass


root = Tk()
root.geometry("1400x800")

label = Label(root)
label.grid(row=0, column=0)

fig, ax = plt.subplots(figsize=(5, 4))
voidPlace1 = FigureCanvasTkAgg(fig, master=root)
voidPlace1.get_tk_widget().grid(row=0, column=0)
voidPlace2 = FigureCanvasTkAgg(fig, master=root)
voidPlace2.get_tk_widget().grid(row=0, column=2)

tabControl = Notebook(root)

tab1 = Frame(tabControl)
tab2 = Frame(tabControl)
tab3 = Frame(tabControl)
tab4 = Frame(tabControl)


tabControl.add(tab1, text='Об\'єкти')
tabControl.add(tab2, text='Протокол')
tabControl.add(tab3, text='Оцінки параметрів')
tabControl.add(tab4, text='Відтворення вибірок')


tabControl.grid(row=50, column=0)

# for c in range(3): tab1.columnconfigure(index=c, weight=1)
# for r in range(4): tab1.rowconfigure(index=r, weight=1)

T1 = Text(master=tab1, height=10, width=100)
T1.grid(row=50, column=0)

T2 = Text(master=tab2, height=10, width=100)
T2.grid(row=50, column=0)


T3 = Text(master=tab3, height=10, width=100)
T3.grid(row=50, column=0)

T4 = Text(master=tab4, height=10, width=100)
T4.grid(row=50, column=0)
T4.insert(END, 'Count\t' + 'N\t' + 'M\t' + 'Sq\t' + '\u03BB\t' + 'sq(\u03BB)\t' + 't-test\t' + 'level\n')



"""window menu"""
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Відкрити файл", command=openFile)
filemenu.add_command(label="Класи", command=classes)

sampleMod_menu = Menu(menubar, tearoff=0)
sampleMod_menu.add_command(label="Експоненціальний", command=expParams)
sampleMod_menu.add_command(label="Рівномірний", command=uniParams)
sampleMod_menu.add_command(label="Вейбула", command=weibParams)

models_reproduction = Menu(menubar, tearoff=0)
models_reproduction.add_command(label="Нормальний", command=norm)
models_reproduction.add_command(label="Експоненціальний", command=exp)
models_reproduction.add_command(label="Рівномірний", command=ravn)
models_reproduction.add_command(label="Вейбула", command=veib)
models_reproduction.add_command(label="Арксінуса", command=arcsin)
models_reproduction.add_command(label="Універсальний", command=univers)


experim = Menu(menubar, tearoff=0)
experim.add_command(label="Створення експерименту", command=experiment)
experim.add_command(label="Відкрити файл", command=openExperiment)




menubar.add_cascade(label="Меню", menu=filemenu)
filemenu.add_cascade(label="Відтворення моделей", menu=models_reproduction)
filemenu.add_cascade(label="Mоделювання вибірок", menu=sampleMod_menu)
filemenu.add_cascade(label="Експеримент", menu=experim)




filemenu.add_command(label="Вийти", command=root.quit)

root.config(menu=menubar)

root.configure(bg="gray")

root.mainloop()
