import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

class FieldDisplay:
    def __init__(self, maxSize_m, dx, y_min, y_max, probePos, sourcePos):
        plt.ion()
        self.probePos = probePos
        self.sourcePos = sourcePos
        self.fig, self.ax = plt.subplots()
        self.line = self.ax.plot(np.arange(0, maxSize_m, dx), [0]*int(maxSize_m/dx))[0]
        self.ax.plot(probePos*dx, 0, 'xr')
        self.ax.plot(sourcePos*dx, 0, 'ok')
        self.ax.set_xlim(0, maxSize_m)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel('x, м')
        self.ax.set_ylabel('Ez, В/м')
        self.ax.grid()
        
    def updateData(self, data):
        self.line.set_ydata(data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def stop(self):
        plt.ioff()

class Probe:
    def __init__(self, probePos, Nt, dt):
        self.Nt = Nt
        self.dt = dt
        self.probePos = probePos
        self.t = 0
        self.E = np.zeros(self.Nt)
        
    def addData(self, data):
        self.E[self.t] = data[self.probePos]
        self.t += 1

def showProbeSignal(probe):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    t = np.arange(0, probe.Nt * probe.dt, probe.dt)
    ax[0].plot(t, probe.E)
    ax[0].set_xlabel('t, с')
    ax[0].set_ylabel('Ez, В/м')
    ax[0].set_xlim(0, probe.Nt * probe.dt)
    ax[0].grid()

    sp = np.abs(fft(probe.E))
    sp = fftshift(sp)
    df = 1 / (probe.Nt * probe.dt)
    freq = np.arange(-probe.Nt * df / 2, probe.Nt * df / 2, df)
    ax[1].plot(freq, sp / max(sp))
    ax[1].set_xlabel('f, Гц')
    ax[1].set_ylabel('|S|/|Smax|')
    ax[1].set_xlim(0, 5e9)
    ax[1].grid()

    plt.subplots_adjust(wspace=0.4)
    plt.show()

# Параметры моделирования
eps = 2.5  # Диэлектрическая проницаемость
W0 = 120 * np.pi
Nt = 2000  # Число временных шагов
Nx = 501  # Число узлов пространственной сетки
maxSize_m = 5.0  # Размер области моделирования вдоль оси X, м
dx = maxSize_m / Nx  # Пространственный шаг
probePos = int(maxSize_m * 7 / 8 / dx)  # Позиция датчика
sourcePos = int(maxSize_m / 2 / dx)  # Позиция источника
Sc = 1.0  # Число Куранта
dt = dx * np.sqrt(eps) * Sc / 3e8  # Временной шаг
probe = Probe(probePos, Nt, dt)
display = FieldDisplay(maxSize_m, dx, -1.5, 1.5, probePos, sourcePos)
Ez = np.zeros(Nx)
Hy = np.zeros(Nx)

# Параметры гармонического сигнала
Fmin = 0.5e9  # Минимальная частота сигнала, Гц
source_signal = np.sin(2 * np.pi * Fmin * np.arange(Nt) * dt)

# Граничные условия для ABC второй степени
Ez_left1 = Ez[1]
Ez_right1 = Ez[-2]
Ez_left2 = Ez[2]
Ez_right2 = Ez[-3]

# Основной цикл FDTD
for q in range(1, Nt):
    # Обновление поля Hy
    Hy[:-1] += (Ez[1:] - Ez[:-1]) * Sc / W0

    # Обновление поля Ez
    Ez[1:] += (Hy[1:] - Hy[:-1]) * Sc * W0 / eps

    # Возбуждение источника
    Ez[sourcePos] += source_signal[q]

    # Граничные условия для ABC второй степени
    Ez[0] = 2 * (1 - Sc) / (1 + Sc) * Ez[1] + (1 - Sc) / (1 + Sc) * Ez_left2
    Ez_left2 = Ez_left1
    Ez_left1 = Ez[1]

    Ez[-1] = 2 * (1 - Sc) / (1 + Sc) * Ez[-2] + (1 - Sc) / (1 + Sc) * Ez_right2
    Ez_right2 = Ez_right1
    Ez_right1 = Ez[-2]

    probe.addData(Ez)
    if q % 10 == 0:
        display.updateData(Ez)

display.stop()
showProbeSignal(probe)
