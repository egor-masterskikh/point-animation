from sympy import symbols, Matrix, solve
import numpy as np
from numpy.linalg import norm
from tabulate import tabulate

P1 = 12
P2 = 14

M1 = 36
M2 = 28

R_Ay, R_Bx, R_By, R_Cx, R_Dx, R_Dy, R_Ex = symbols(
    'R_Ay,R_Bx,R_By,R_Cx,R_Dx,R_Dy,R_Ex'
)

P1_vec = Matrix([0, -P1, 0])
P2_vec = Matrix([0, -P2, 0])

R_A_vec = Matrix([0, R_Ay, 0])
R_B_vec = Matrix([R_Bx, R_By, 0])
R_C_vec = Matrix([R_Cx, 0, 0])
R_D_vec = Matrix([R_Dx, R_Dy, 0])
R_E_vec = Matrix([R_Ex, 0, 0])

M1_vec = Matrix([0, 0, M1])
M2_vec = Matrix([0, 0, -M2])

# радиус-векторы отсчитываются от точки D

r_P1 = Matrix([1, -1, 0])

r_R_A = Matrix([-4, 0, 0])
r_R_B = Matrix([0, -1, 0])
r_R_C = Matrix([2, -2, 0])
r_R_E = Matrix([2, 2, 0])

M_P1_vec = r_P1.cross(P1_vec)

M_A_vec = r_R_A.cross(R_A_vec)
M_B_vec = r_R_B.cross(R_B_vec)
M_C_vec = r_R_C.cross(R_C_vec)
M_E_vec = r_R_E.cross(R_E_vec)

# для балки AD
M_res_AD = M_A_vec + M1_vec

# для балки DE
M_res_DE = M_E_vec + M2_vec

# для конструкции ADE
F_res_ADE = R_A_vec + R_D_vec + R_E_vec + P2_vec

# для конструкции DBC
M_res_DBC = M_B_vec + M_C_vec + M_P1_vec
F_res_DBC = R_B_vec + R_C_vec - R_D_vec + P1_vec + P2_vec

res = solve(
    [M_res_AD, M_res_DE, F_res_ADE, M_res_DBC, F_res_DBC],
    [R_Ay, R_Bx, R_By, R_Cx, R_Dx, R_Dy, R_Ex]
)

R_Ay, R_Bx, R_By, R_Cx, R_Dx, R_Dy, R_Ex = (
    res[R_Ay], res[R_Bx], res[R_By], res[R_Cx],
    res[R_Dx], res[R_Dy], res[R_Ex]
)

# нормы реакций
R_A = abs(R_Ay)
R_B = norm(np.array([R_Bx, R_By], dtype='float'))
R_C = abs(R_Cx)
R_D = norm(np.array([R_Dx, R_Dy], dtype='float'))
R_E = abs(R_Ex)

with open('С.4_res.txt', 'w') as out_file:
    table = [['', 'Вектор, кН', 'Норма, кН'],
             ['R_A', (0, R_Ay), R_A],
             ['R_B', (R_Bx, R_By), R_B],
             ['R_C', (R_Cx, 0), R_C],
             ['R_D', (R_Dx, R_Dy), R_D],
             ['R_E', (R_Ex, 0), R_E]]
    print(tabulate(
        table, headers='firstrow', floatfmt='.2f', colalign=['right'] * 3
    ), file=out_file)
