from sympy import symbols, Matrix, solve
from math import sin, cos, pi
import numpy as np

# модули сил
Q = 4
G = 1

# размеры
a = 25
b = 20
c = 8
R = 15
r = 10

P, R_Ax, R_Ay, R_Az, R_Bx, R_Bz = symbols('P,R_Ax,R_Ay,R_Az,R_Bx,R_Bz')

# радиус-векторы, проведённые из точки A
# в точку приложения соответствующей силы
r_Q = Matrix([r, a, 0])
r_G = Matrix([0, a - b, 0])
r_P = Matrix([-R, a + b + 3 * c, 0])
r_R_A = Matrix([0, 0, 0])
r_R_B = Matrix([0, a + b, 0])

# силы
Q_vec = Matrix([0, Q * cos(pi / 4), -Q * sin(pi / 4)])
G_vec = Matrix([0, 0, -G])
P_vec = Matrix([0, -P * cos(pi / 3), -P * sin(pi / 3)])
R_A_vec = Matrix([R_Ax, R_Ay, R_Az])
R_B_vec = Matrix([R_Bx, 0, R_Bz])

# равнодействующая сила
F_vec = Q_vec + G_vec + P_vec + R_A_vec + R_B_vec

# суммарный момент
M_vec = (r_Q.cross(Q_vec) + r_G.cross(G_vec) + r_P.cross(P_vec)
         + r_R_A.cross(R_A_vec) + r_R_B.cross(R_B_vec))

# словарь, где каждой искомой величине соответствует вычисленное значение
res = solve([F_vec, M_vec], [P, R_Ax, R_Ay, R_Az, R_Bx, R_Bz])

# словарь res с округлёнными значениями
fmt_res = dict(zip(
    res.keys(),
    np.array(tuple(res.values()), dtype=np.double).round(1)
))


with open('С.7_res.txt', 'w') as out_file:
    print(
        f'Модуль силы P:\t{fmt_res[P]} кН',
        f'Сила R_A:\t({fmt_res[R_Ax]},\t{fmt_res[R_Ay]},\t{fmt_res[R_Az]}) кН',
        f'Сила R_B:\t({fmt_res[R_Bx]},\t{0:.2f},\t{fmt_res[R_Bz]}) кН',
        sep='\n',
        file=out_file
    )
