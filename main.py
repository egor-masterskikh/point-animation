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

# +++++ СИМВОЛЬНЫЕ ВЫЧИСЛЕНИЯ +++++
t = sp.Symbol('t')

r = 2 + sp.sin(6 * t)
phi = 6.5 * t + 1.2 * sp.cos(6 * t)

x = r * sp.cos(phi)
y = r * sp.sin(phi)

v_x = sp.diff(x, t)
v_y = sp.diff(y, t)
v_sp = sp.sqrt(v_x ** 2 + v_y ** 2)

a_x = sp.diff(v_x, t)
a_y = sp.diff(v_y, t)
a_n = sp.det(sp.Matrix(
    [[v_x, v_y],
     [a_x, a_y]]
)) / v_sp

R = v_sp ** 2 / a_n  # радиус кривизны траектории
# ----- СИМВОЛЬНЫЕ ВЫЧИСЛЕНИЯ -----

# +++++ ПЕРЕВОД В ФУНКЦИИ +++++
x_f = sp.lambdify(
    args=t,  # аргументы, которые будет принимать функция
    expr=x,  # собственно логическая часть функции
    modules='numpy'
)
y_f = sp.lambdify(t, y, 'numpy')

v_x_f = sp.lambdify(t, v_x, 'numpy')
v_y_f = sp.lambdify(t, v_y, 'numpy')
v_f = sp.lambdify(t, v_sp, 'numpy')

a_x_f = sp.lambdify(t, a_x, 'numpy')
a_y_f = sp.lambdify(t, a_y, 'numpy')
a_n_f = sp.lambdify(t, a_n, 'numpy')

R_f = sp.lambdify(t, R, 'numpy')
# ----- ПЕРЕВОД В ФУНКЦИИ -----

# +++++ МАССИВЫ ДАННЫХ +++++
T_END = 20

t = np.linspace(0, T_END, 1000)  # массив моментов времени


def rot_matrix(alpha):
    """
    :param alpha: угол поворота в радианах
    :return: матрица поворота
    """
    return [[cos(alpha), -sin(alpha)],
            [sin(alpha), cos(alpha)]]


def rot2D(x: np.ndarray, y: np.ndarray, alpha):
    """
    Поворачивает объект (массив точек) на заданный угол
    """
    return np.matmul(rot_matrix(alpha), [x, y])


x = x_f(t)
y = y_f(t)

min_x, max_x = np.min(x), np.max(x)
min_y, max_y = np.min(y), np.max(y)
plot_left, plot_right = 1.75 * min_x, 1.75 * max_x
plot_bottom, plot_top = 1.75 * min_y, 1.75 * max_y
plot_width = plot_right - plot_left
plot_height = plot_top - plot_bottom

v_x = v_x_f(t)
v_y = v_y_f(t)
min_v_x, max_v_x = np.min(v_x), np.max(v_x)
min_v_y, max_v_y = np.min(v_y), np.max(v_y)
k = max(max_v_x / plot_right, max_v_y / plot_top)
v_x /= k
v_y /= k
v = v_f(t) / k
v_phi = np.arctan2(v_y, v_x)

a_x = a_x_f(t)
a_y = a_y_f(t)
min_a_x, max_a_x = np.min(a_x), np.max(a_x)
min_a_y, max_a_y = np.min(a_y), np.max(a_y)
k = max(max_a_x / plot_right, max_a_y / plot_top)
a_x /= k
a_y /= k
a_n = a_n_f(t) / k
a_phi = np.arctan2(a_y, a_x)

R = R_f(t)
# ----- МАССИВЫ ДАННЫХ -----

# +++++ НАСТРОЙКА ОТОБРАЖЕНИЯ +++++
fig: figure.Figure = plt.figure(figsize=(10, 10))

ax: axes._axes.Axes = fig.add_subplot()

# сохранение пропорций графика вне зависимости от конфигурации окна
ax.axis('equal')

# определение области графика, в которой будет производиться отрисовка
ax.set_xlim(1.75 * min_x, 1.75 * max_x)
ax.set_ylim(1.75 * min_y, 1.75 * max_y)

arrow_len = 0.3  # длина стрелки
arrow_angle = 30  # угол раствора стрелки, в градусах
arrow_width = 2 * arrow_len * tan(radians(arrow_angle / 2))
arrow_x = np.array((-arrow_len, 0, -arrow_len))
arrow_y = np.array((-arrow_width / 2, 0, arrow_width / 2))
# ----- НАСТРОЙКА ОТОБРАЖЕНИЯ -----

ax.plot(x, y, color='tab:blue')

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
