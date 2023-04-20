""" С клавиатуры вводятся два числа K и N. Квадратная матрица А(N,N), состоящая из 4-х равных по размерам подматриц,
B, C, D, E заполняется случайным образом целыми числами в интервале [-10,10]. Для тестирования использовать не случайное
заполнение, а целенаправленное.
Вид матрицы А:
D	Е
С	В
Для простоты все индексы в подматрицах относительные.
По сформированной матрице F (или ее частям) необходимо вывести не менее 3 разных графика.
Программа должна использовать функции библиотек numpy  и mathplotlib
Вариант 4:
Формируется матрица F следующим образом: скопировать в нее А и если в Е количество нулевых элементов 
в нечетных столбцах больше, чем количество отрицательных элементов в четных строках, то поменять местами С и В симметрично,иначе В и Е поменять 
местами несимметрично. При этом матрица А не меняется.
После чего если определитель матрицы А больше суммы днагональных элементов матрицы F, то вычисляется выражение: А^-1* AT - К * F, пначе вычисляется 
выражение (AT +G-F^-1)*К, где G-нижняя треугольная матрица, полученная из А. Выводятся по мере формпрования A. F и все матричные операции последовательно."""

from math import ceil
import random as r
import numpy as np
from matplotlib import pyplot as plt


def heatmap(data, row_labels, col_labels, ax, cbar_kw=None, **kwargs):  # аннотированная тепловая карта
    if cbar_kw is None:
        cbar_kw = {}
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    return im, cbar


def annotate_heatmap(im, data=None, textcolors=("black", "white"), threshold=0):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    kw = dict(horizontalalignment="center", verticalalignment="center")
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(data[i, j] > threshold)])
            text = im.axes.text(j, i, data[i, j], **kw)
            texts.append(text)
    return texts


try:
    n = int(input('Введите число N: '))
    k = int(input('Введите число K: '))
    while n < 5:
        n = int(input('Введите число N больше 4: '))


    middle_n = ceil(n / 2)  # Середина матрицы
    A = np.zeros((n, n))  # Задаём матрицу A
    for i in range(n):
        for j in range(n):
            A[i][j] = r.randint(-10, 10)
    AT = np.transpose(A)  # Транспонированная матрица А
    A_obr = np.linalg.inv(A)  # Обратная матрица А
    det_A = np.linalg.det(A)  # Определитель матрицы А
    F = A.copy()  # Задаём матрицу F
    G = np.zeros((n, n))  # Заготовка матрицы G

    print('\nМатрица А:')
    print(A)
    print('\nТранспонированная А:')
    print(AT)
    # Выделяем матрицы E C B
    if n % 2 == 1:
       E = [A[i][middle_n - 1:n] for i in range(middle_n)]
       C = [A[i][0:middle_n] for i in range(middle_n - 1, n)]
       B = [A[i][middle_n - 1:n] for i in range(middle_n - 1, n)]
    else:
       E = [A[i][middle_n:n] for i in range(0, middle_n)]
       C = [A[i][0:middle_n] for i in range(middle_n, n)]
       B = [A[i][middle_n:n] for i in range(middle_n, n)]

    zero_E = 0

    for i in range(middle_n):  # Считаем количество нулевых элементов в нечетных столбцах в матрице E
        for j in range(middle_n):
            if (j + 1) % 2 == 1:
                if E[i][j] == 0:
                    zero_E += 1

    ch = 0
    for i in range(middle_n):  # Считаем количество  отрицательных элементов в чётных строках в матрице E
        for j in range(middle_n):
            if (i + 1) % 2 == 0:
                if E[i][j] < 0:
                    ch = 1

    if zero_E > ch:
        print(f'\nВ матрице "E" количество нулевых элементов в нечетных столбцах ({zero_E})')
        print(f'больше чем количество отрицательных элементов в четных строках ({ch})')
        print('поэтому симметрично местами подматрицы C и B:')
        C, B = B, C
        for i in range(middle_n):
            C[i] = C[i][::-1]  # Симметрично меняем значения в C
            E[i] = E[i][::-1]  # Симметрично меняем значения в E
        if n % 2 == 1:
            for i in range(middle_n - 1, n):  # Перезаписываем С
                for j in range(middle_n):
                    F[i][j] = C[i - (middle_n - 1)][j]
            for i in range(middle, n):  # Перезаписываем B
                for j in range(middle - 1, n):
                    F[i][j] = B[i - (middle)][j - (middle - 1)]
        else:
            for i in range(middle_n, n):
                for j in range(middle_n):
                    F[i][j] = C[i - middle_n][j]
            for i in range(middle, n):
                for j in range(middle, n):
                    F[i][j] = B[i - middle][j - middle]
    else:
        print(f'\nВ матрице "E" количество нулевых элементов в нечетных столбцах ({zero_E})')
        print(f'меньше чем количество отрицательных элементов в четных строках ({ch}) или равно ей')
        print('поэтому несимметрично меняем местами подматрицы B и E:')
        B, E = E, B
        if n % 2 == 1:
            for i in range(middle_n, n):  # Перезаписываем B
                for j in range(middle_n - 1, n):
                    F[i][j] = B[i - (middle_n)][j - (middle_n - 1)]
            for i in range(middle_n):  # Перезаписываем Е
                for j in range(middle_n - 1, n):
                    F[i][j] = E[i][j - (middle_n - 1)]
        else:
            for i in range(middle_n, n):
                for j in range(middle_n, n):
                    F[i][j] = B[i - middle_n][j - middle_n]
            for i in range(0, middle_n):
                for j in range(middle_n, n):
                    F[i][j] = E[i][j - middle_n]
    print('\nМатрица F:')
    print(F)
    # Сумма диагональных элементов матрицы F
    sum_d_F = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                sum_d_F += F[i][j]
            if (i + j + 1) == n and ((i == j) != ((i + j + 1) == n)):
                sum_d_F += F[i][j]

    if det_A > sum_d_F:
        print(f'\nОпределитель матрицы А({int(det_A)})')
        print(f'больше суммы диагональных элементов матрицы F({int(sum_d_F)})')
        print('поэтому вычисляем выражение: A^-1 * AT – K * F:')
        try:
          KF = F * k  # K * F
          A_obrAT = np.matmul(A_obr, AT)  # A^-1 * AT
          result = A_obrAT - KF  # A^-1 * AT – K * F

          print('\nРезультат K * F:')
          print(KF)
          print("\nРезультат A_obr * AT:")
          print(A_obrAT)
          print('\nРезультат A^-1 * AT – K * F:')
          print(result)
        except np.linalg.LinAlgError:
          print("Одна из матриц является вырожденной (определитель равен 0),"
      " поэтому обратную матрицу найти невозможно.")
    else:
      print(f'\nОпределитель матрицы А({int(det_A)})')
      print(f'меньше суммы диагональных элементов матрицы F({int(sum_d_F)}) или равен ей')
      print('поэтому вычисляем выражение ((AТ + G - F^-1) * K :')
      for i in range(n):
            for j in range(n):
                if i >= j and (i + j + 1) >= n:
                    G[i][j] = A[i][j]
      ATG = AT + G  # AТ + G
      F_obr = np.linalg.inv(F)  # Обратная матрица F
      ATGF = ATG - F_obr  # AT + G - F^-1
      result = ATGF * k  # (AТ + G - F^-1) * K
      print('\n Обратная  матрица F:')
      print(F_obr)
      print('\nМатрица G:')
      print(G)
      print('\nРезультат AТ + G:')
      print(ATG)
      print('\nРезультат AТ + G - F:-1:')
      print(ATGF)
      print('\nРезультат (AТ + G - F^-1) * K:')
      print(result)
    av = [np.mean(abs(F[i::])) for i in range(n)]
    av = int(sum(av))  # сумма средних значений строк (используется при создании третьего графика)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    x = list(range(1, n + 1))
    for j in range(n):
        y = list(F[j, ::])  # обычный график
        axs[0, 0].plot(x, y, ',-', label=f"{j + 1} строка.")
        axs[0, 0].set(title="График с использованием функции plot:", xlabel='Номер элемента в строке',
                      ylabel='Значение элемента')
        axs[0, 0].grid()
        axs[0, 1].bar(x, y, 0.4, label=f"{j + 1} строка.")  # гистограмма
        axs[0, 1].set(title="График с использованием функции bar:", xlabel='Номер элемента в строке',
                      ylabel='Значение элемента')
        if n <= 10:
            axs[0, 1].legend(loc='lower right')
            axs[0, 1].legend(loc='lower right')
    explode = [0] * (n - 1)  # отношение средних значений от каждой строки
    explode.append(0.1)
    sizes = [round(np.mean(abs(F[i, ::])) * 100 / av, 1) for i in range(n)]
    axs[1, 0].set_title("График с использованием функции pie:")
    axs[1, 0].pie(sizes, labels=list(range(1, n + 1)), explode=explode, autopct='%1.1f%%', shadow=True)

    im, cbar = heatmap(F, list(range(n)), list(range(n)), ax=axs[1, 1], cmap="magma_r")
    texts = annotate_heatmap(im)
    axs[1, 1].set(title="Создание аннотированных тепловых карт:", xlabel="Номер столбца", ylabel="Номер строки")
    plt.suptitle("Использование библиотеки matplotlib")
    plt.tight_layout()
    plt.show()

    print('\nРабота программы завершена.')
except ValueError:  # ошибка на случай введения не числа в качестве порядка или коэффициента
    print('\nВведенный символ не является числом. Перезапустите программу и введите число.')
