from itertools import combinations
from operator import itemgetter

import numpy as np

def I_matr(n): #единичная матрицы
    M = np.zeros((n, n), int)
    for i in range(n):
        M[i, i] = 1
    return M

def Cnk(n, k):
    return (fact(n) / (fact(k) * fact(n-k))).__int__()

def fact(n):
    res = 1
    if n > 0:
        for i in range (1, n+1):
            res *= i
        return res
    else:
        return res

def all_comb(n): #все возможные комбинации 0 и 1 в векторах размерности n
    res = []
    for i in range(2 ** n):
        s = []
        for j in range(n):
            s.append(i % 2)
            i = i // 2
        res.append(s)
    res = np.asarray(res)   # в данной работе будем называть стандартным порядком
    return res

def all_I(r, m):           # Найдем все возможные I длины до r и сортируем их в нужном порядке
    res = []
    res.append([])      # Первым элементом всегда будет пустое множество
    I = []
    for i in range(m):  # Заполним I числами, которые собираемся сочетать (от 0 до n)
        I.append(i)
    curr = []
    for k in range(1, r+1):     # Для k от 0 (т.к. нулевой элемент - это пустое множество) до r-1 включительно
        curr.append([*combinations(I, k)])   # Сохраняем в res все возможные комбинации из I по k элементов
        curr[0].reverse()        # Каждый "блок" (по 1, 2, 3 и т.д. элементов) берем в обратном порядке
        curr[0].sort(key=itemgetter(k - 1), reverse=True)    # В случае, если будет такое, что сумма элементов в соседних I равна,
                                                            # но порядок не соответствует действительности
                                                            # т.е. старшие разряды стоят позже, чем младшие
                                                            # отсортируем по старшим разрядам
        for j in range(Cnk(len(I), k)):
            res.append(curr[0][j])
        curr = []
    return res

def k_rm(r, m):     # размерность кода Рида-Миллера
    sum = 0
    for i in range(r+1):
        sum += Cnk(m, i)
    return sum

def rm_G(r, m):     # Порождающая матрица кода Рида-Маллера в канонич. виде
    res = np.zeros((k_rm(r, m), 2 ** m), int)
    t = 0
    for i in all_I(r, m):
        res[t] = find_v(i, m)
        t += 1
    return res

def find_v(I, m):   # найдем вектор v_I
    if len(I) == 0:     # Если I - пустое множество, то
        return np.ones(2 ** m, int)     # Возвращаем вектор единиц длинной 2^m, согласно формуле
                                        # f_I (x0, x1, … , x𝑚−1) = 1, если I = ∅.
    j = 0
    res = np.zeros(2 ** m, int)                        # Если I ≠ ∅
    for x in all_comb(m):            # Пройдемся по всем комбинациям стандартного порядка
        f = 1
        for i in I:                  # По множеству I выбранных индексов, которое приняли на вход
            f *= (x[i] + 1) % 2      # f = П (𝑥𝑖 + 1), где + это сложение по модулю 2,  i принадлежит I
        res[j] = f               # Получившуюся на каждой итерации f записываем в вектор результатов
        j += 1
    return res              # получаем вектор v_I

def find_vt(I, m, t):   # найдем вектор v_It
    if len(I) == 0:     # Если I - пустое множество, то
        return np.ones(2 ** m, int)     # Возвращаем вектор единиц длинной 2^m, согласно формуле
                                        # f_I (x0, x1, … , x𝑚−1) = 1, если I = ∅.

    res = []                         # Если I ≠ ∅
    for x in all_comb(m):            # Пройдемся по всем комбинациям стандартного порядка
        f = 1
        for i in I:                  # По множеству I выбранных индексов, которое приняли на вход
            f *= (x[i] + t[i] + 1) % 2      # f = П (𝑥𝑖 + ti + 1), где + это сложение по модулю 2,  i принадлежит I
        res.append(f)                # Получившуюся на каждой итерации f записываем в вектор результатов
    # print("v {", I, "} {",  t,  "}", res)
    return res              # получаем вектор v_It

def _c(I, m):   # Нахождение комплементарным множеством к множеству I на Zm
    res = []
    for i in range(m):
        if i not in I:
            res.append(i)
    return res

def find_H_I(I, m):
    res = []
    for x in all_comb(m):  # Пройдемся по всем комбинациям стандартного порядка
        f = 1
        for i in I:  # По множеству I выбранных индексов, которое приняли на вход
            f *= (x[i] + 1) % 2  # f = П (𝑥𝑖 + 1), где + это сложение по модулю 2,  i принадлежит I
        if f == 1:
            res.append(x)  # Получившуюся на каждой итерации f записываем в вектор результатов
    return res

def major(w, r, m):
    i = r
    curr_w = w
    dead_edge = 2 ** (m - r - 1) - 1
    mi = []
    check = True

    while check:
        for J in lgth(m, i):
            edge = 2 ** (m - i - 1)
            zero = 0
            one = 0
            for t in find_H_I(J, m):
                c = np.dot(curr_w, find_vt(_c(J, m), m, t)) % 2
                if c == 0:
                    zero += 1
                if c == 1:
                    one += 1
            if zero > dead_edge and one > dead_edge:
                print("Необходима повторная отправка сообщения")
                return
            if zero > edge:
                mi.append(0)
            if one > edge:
                mi.append(1)
                curr_w = (curr_w + find_v(J, m)) % 2
        if i > 0:
            if len(curr_w) < dead_edge:
                for J in lgth(m, r+1):
                    mi.append(0)
                    check = False
            i -= 1
        else:
            check = False
    mi.reverse()
    return mi

def lgth(m, l):
    I = []
    for i in range(m):
        I.append(i)
    cur = []
    cur.append([*combinations(I, l)])
    if len(cur[0][0]) != 0:
        cur[0].sort(key=itemgetter(len(cur[0][0])-1))
    res = []
    for i in range(len(cur[0])):
        res.append(cur[0][i])
    return res

def rand_word(n): #рандомное слово длины n
    return np.random.randint(0, 2, n)

def createError(n, t):  #создание вектора ошибок
    error = np.zeros(n, int)
    errorNum = np.zeros(t, int)
    for i in range (0, t):
        errorNum[i] = np.random.randint(0, n)
    for i in range(t):
        error[errorNum[i]] = 1
    return error


if __name__ == '__main__':
    m = 4
    r = 2
    G = rm_G(r, m)
    print("Порождающая матрица : \n", G)

    U = rand_word(k_rm(r, m))
    print("Слово длины k: \n", U)
    V = np.dot(U, G) % 2
    print("Kодовое слово длины n: \n", V)

    for i in range(1, 3):
        Err = createError(2**m, i)
        print(i, "-кратная ошибка: \n", Err)
        W = (V + Err) % 2
        print("Слово с ошибкой: \n", W)

        Correct_W = major(W, r, m)
        if Correct_W:
            print("Исправленное слово: \n", Correct_W)
            V1 = np.dot(Correct_W, G) % 2
            print("Проверяем, умножив полученный вектор на порожлающую матрицу G(2,4): \n", V1)


    #print(all_comb(3))
    #I = {1, 2}
    #print(find_v(I, 3))
    #print(all_I(r))

    #w = [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    #print("w = \n", w)
    #J = {0, 1}
    #print(_c(J, m))
    #print(find_H_I(J, m))
    #Correct_W = major(w, r, m)
    #print("Исправленное слово: \n", Correct_W)
    #check = np.dot(Correct_W, G) % 2
    #print("Проверяем, умножив полученный вектор на порожлающую матрицу G(2,4): \n", check)



   #for l in range(Cnk(n, k) - 1):
         #   if sum(res[k][l]) == sum(res[k][l+1]):
          #      res[k].sort(key=itemgetter(k-1), reverse=True)
                #if res[k][l][k-1] < res[k][l + 1][k-1]:
                    #res.insert()
                    #print(res[k][l][k-1], res[k][l+1][k-1])