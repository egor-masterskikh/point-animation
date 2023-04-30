import argparse
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# используются для указания типа значений,
# возвращаемых методами matplotlib'а
import matplotlib.figure as figure
import matplotlib.axes as axes
import matplotlib.lines as lines

T_END = 20

t = np.linspace(0, T_END, 1000)  # массив моментов времени

# массив значений модуля радиус-вектора для каждого момента времени
r = 2 + np.sin(6 * t)

# массив значений угла поворота радиус-вектора для каждого момента времени
phi = 6.5 * t + 1.2 * np.cos(6 * t)

x = r * np.cos(phi)
y = r * np.sin(phi)

fig: figure.Figure = plt.figure(figsize=(10, 10))

ax: axes._axes.Axes = fig.add_subplot()

ax.plot(x, y)

# сохранение пропорций графика вне зависимости от конфигурации окна
ax.axis('equal')

P: lines.Line2D = ax.plot(x[0], y[0], marker='o')[0]


def update(frame):
    P.set_data((x[frame],), (y[frame],))


anim = animation.FuncAnimation(
    fig=fig,
    func=update,       # функция, запускаемая для каждого кадра
    frames=len(t),     # количество кадров
    interval=1,        # задержка между кадрами, в миллисекундах
    repeat_delay=3000  # задержка в миллисекундах между последовательными
                       # запусками анимации
)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action='store_true')
args = parser.parse_args()

# в зависимости от аргумента командной строки анимация
# либо сохраняется,
# либо отображается непосредственно в окне

if args.save:
    anim_filepath = Path('report/animation.gif')
    anim.save(
        filename=str(anim_filepath),
        writer='pillow',
        fps=30
    )
    os.system(
        f'img2pdf {anim_filepath} -o '
        f'{anim_filepath.parent / anim_filepath.stem}.pdf'
    )

else:
    plt.show()
