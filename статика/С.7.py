from sympy import symbols, Matrix, solve
from math import sin, cos, pi
import numpy as np
from numpy.linalg import norm
from tabulate import tabulate

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
r_G = Matrix([0, b, 0])
r_P = Matrix([-R, a + b + 3 * c, 0])
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
M_vec = (r_Q.cross(Q_vec) + r_G.cross(G_vec)
         + r_P.cross(P_vec) + r_R_B.cross(R_B_vec))

# словарь, где каждой искомой величине соответствует вычисленное значение
res = solve([F_vec, M_vec], [P, R_Ax, R_Ay, R_Az, R_Bx, R_Bz])

P, R_Ax, R_Ay, R_Az, R_Bx, R_Bz = (
    res[P], res[R_Ax], res[R_Ay], res[R_Az], res[R_Bx], res[R_Bz]
)
R_A_vec = np.array([R_Ax, R_Ay, R_Az], dtype='float')
R_B_vec = np.array([R_Bx, 0, R_Bz], dtype='float')
R_A, R_B = norm(R_A_vec), norm(R_B_vec)

with open('С.7_res.txt', 'w') as out_file:
    R_A_vec = tuple(R_A_vec.round(1))
    R_B_vec = tuple(R_B_vec.round(1))
    table = [['', 'Вектор, кН', ' Норма, кН'],
             ['P', '', P],
             ['R_A', R_A_vec, R_A],
             ['R_B', R_B_vec, R_B]]
    print(tabulate(
        table, headers='firstrow', floatfmt='.2f',
        colalign=['center', 'right', 'right']
    ), file=out_file)
