import argparse
from pathlib import Path
import os
import sympy as sp
import numpy as np
from math import sin, cos, tan, radians
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# используются для указания типа значений,
# возвращаемых методами matplotlib'а
import matplotlib.figure as figure
import matplotlib.axes as axes
import matplotlib.lines as lines

t_sp = sp.Symbol('t')

r_sp = 2 + sp.sin(6 * t_sp)
phi_sp = 6.5 * t_sp + 1.2 * sp.cos(6 * t_sp)

x_sp = r_sp * sp.cos(phi_sp)
y_sp = r_sp * sp.sin(phi_sp)

v_x_sp = sp.diff(x_sp, t_sp)
v_y_sp = sp.diff(y_sp, t_sp)
v_sp = sp.sqrt(v_x_sp ** 2 + v_y_sp ** 2)

a_x_sp = sp.diff(v_x_sp, t_sp)
a_y_sp = sp.diff(v_y_sp, t_sp)
a_n_sp = sp.det(sp.Matrix(
    [[v_x_sp, v_y_sp],
     [a_x_sp, a_y_sp]]
)) / v_sp

# радиус кривизны траектории
R_sp = v_sp ** 2 / a_n_sp

x_f = sp.lambdify(
    args=t_sp,  # аргументы, которые будет принимать функция
    expr=x_sp,  # собственно логическая часть функции
    modules='numpy'
)
y_f = sp.lambdify(t_sp, y_sp, 'numpy')

v_x_f = sp.lambdify(t_sp, v_x_sp, 'numpy')
v_y_f = sp.lambdify(t_sp, v_y_sp, 'numpy')
v_f = sp.lambdify(t_sp, v_sp, 'numpy')

a_x_f = sp.lambdify(t_sp, a_x_sp, 'numpy')
a_y_f = sp.lambdify(t_sp, a_y_sp, 'numpy')
a_n_f = sp.lambdify(t_sp, a_n_sp, 'numpy')

R_f = sp.lambdify(t_sp, R_sp, 'numpy')

T_END = 20

t = np.linspace(0, T_END, 1000)  # массив моментов времени


def rot_matrix(phi):
    """
    :param phi: угол поворота в радианах
    :return: матрица поворота
    """
    return [[cos(phi), -sin(phi)],
            [sin(phi), cos(phi)]]


def rot2D(x: np.ndarray, y: np.ndarray, phi):
    """
    Поворачивает объект (массив точек) на заданный угол
    """
    return np.matmul(rot_matrix(phi), [x, y])


x = x_f(t)
y = y_f(t)
# TODO: коэффициент масштабирования должен подбираться автоматически
v_x = v_x_f(t) / 3
v_y = v_y_f(t) / 3
v = v_f(t) / 3
v_phi = np.arctan2(v_y, v_x)

a_x = a_x_f(t) / 20
a_y = a_y_f(t) / 20
a_n = a_n_f(t) / 20
a_phi = np.arctan2(a_y, a_x)

R = R_f(t)


fig: figure.Figure = plt.figure(figsize=(10, 10))

ax: axes._axes.Axes = fig.add_subplot()

# сохранение пропорций графика вне зависимости от конфигурации окна
ax.axis('equal')

# определение области графика, в которой будет производиться отрисовка
# TODO: границы должны определяться из
#  максимального и минимального значений координат и скоростей
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)

ax.plot(x, y, color='tab:blue')

arrow_len = 0.3  # длина стрелки
arrow_angle = 30  # угол раствора стрелки, в градусах
arrow_width = 2 * arrow_len * tan(radians(arrow_angle / 2))
arrow_x = np.array((-arrow_len, 0, -arrow_len))
arrow_y = np.array((-arrow_width / 2, 0, arrow_width / 2))

point: lines.Line2D = ax.plot(x[0], y[0], marker='o', color='tab:orange')[0]

v_line = ax.plot(
    (x[0], x[0] + v_x[0]),
    (y[0], y[0] + v_y[0]),
    color='tab:green'
)[0]

v_arrow_x, v_arrow_y = rot2D(arrow_x, arrow_y, v_phi[0])
v_arrow = ax.plot(
    x[0] + v_x[0] + v_arrow_x,
    y[0] + v_y[0] + v_arrow_y,
    color='tab:green'
)[0]

a_line = ax.plot(
    (x[0], x[0] + a_x[0]),
    (y[0], y[0] + a_y[0]),
    color='tab:red'
)[0]

a_arrow_x, a_arrow_y = rot2D(arrow_x, arrow_y, a_phi[0])
a_arrow = ax.plot(
    x[0] + a_x[0] + a_arrow_x,
    y[0] + a_y[0] + a_arrow_y,
    color='tab:red'
)[0]


R0_x, R0_y = rot2D(v_x / v, v_y / v, radians(90))
R_x, R_y = R0_x * R, R0_y * R
curvature_center = ax.plot(
    x[0] + R_x[0],
    y[0] + R_y[0],
    marker='o',
    color='tab:olive'
)[0]


def update(frame):
    point.set_data((x[frame],), (y[frame],))

    v_line.set_data(
        (x[frame], x[frame] + v_x[frame]),
        (y[frame], y[frame] + v_y[frame])
    )
    v_arrow_x, v_arrow_y = rot2D(arrow_x, arrow_y, v_phi[frame])
    v_arrow.set_data(
        (x[frame] + v_x[frame] + v_arrow_x,),
        (y[frame] + v_y[frame] + v_arrow_y,)
    )

    a_line.set_data(
        (x[frame], x[frame] + a_x[frame]),
        (y[frame], y[frame] + a_y[frame])
    )
    a_arrow_x, a_arrow_y = rot2D(arrow_x, arrow_y, a_phi[frame])
    a_arrow.set_data(
        (x[frame] + a_x[frame] + a_arrow_x,),
        (y[frame] + a_y[frame] + a_arrow_y,)
    )

    curvature_center.set_data(
        (x[frame] + R_x[frame],),
        (y[frame] + R_y[frame],)
    )


anim = animation.FuncAnimation(
    fig=fig,
    func=update,    # функция, запускаемая для каждого кадра
    frames=len(t),  # количество кадров
    interval=50,    # задержка в миллисекундах между кадрами
    # задержка в миллисекундах между последовательными запусками анимации
    repeat_delay=3000
)


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action='store_true')
args = parser.parse_args()

# в зависимости от аргумента командной строки анимация
# либо сохраняется,
# либо отображается непосредственно в окне

if args.save:
    anim_filepath = Path('report/animation.gif')
    anim.save(filename=str(anim_filepath), writer='pillow', fps=30)
    os.system(
        f'img2pdf {anim_filepath} -o '
        f'{anim_filepath.parent / anim_filepath.stem}.pdf'
    )

else:
    plt.show()
