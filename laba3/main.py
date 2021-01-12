import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def lolipop(values: np.ndarray, xs: np.ndarray, title):
    if xs is None:
        xs = np.arange(values.shape[0])

    plt.stem(xs, values)
    plt.title(title)
    plt.grid(True)
    plt.show()
    plt.close()


# Filter order
filter_order = 1

# Вычисление производной сигнала, КИХ-фильтр первого порядка
diff_filter = np.array([1, -1])
lolipop(diff_filter, None, title="Difference Filter (FIR)")

# Построение импульсной характеристики
N = diff_filter.shape[0] * 2 + 1
imp = np.zeros([N])
imp[N // 2] = 1
ir = signal.convolve(imp, diff_filter)
lolipop(ir, None, title="Impulse response (FIR)")

# Построение переходной характеристики
imp = np.concatenate([np.zeros([N // 2]), np.ones([N // 2 + 1])])
ir = signal.convolve(imp, diff_filter)
lolipop(ir, None, title="Step response (FIR)")

# Получение АЧХ и ФЧХ (для КИХ-фильтров достаточно передать сам фильтр в функцию ниже)
w, h = signal.freqz(diff_filter)
amp_response = np.abs(h)
phase_response = np.angle(h)

plt.plot(w, amp_response, label='amplitude response')
plt.plot(w, phase_response, label='phase response')
plt.title("Amplitude and Phase response (FIR, difference filter, 1-order)")
plt.grid(True)
plt.legend()
plt.show()
plt.close()

# Генерируем фильтр Батеруорда первого порядка (фильтр Батеруорда - БИХ-фильтр)
b, a = signal.butter(filter_order, Wn=0.7, btype='lowpass', output='ba')
# b, a = signal.butter(filter_order, Wn=0.01, btype='lowpass', output='ba')

# Получение импульсной характеристики
ir = signal.impulse([b, a])
lolipop(ir[1], ir[0], title="Impulse Response (IIR)")

# Получение переходной характерисики
sr = signal.step([b, a])
lolipop(sr[1], sr[0], title="Step response (IIR)")

# Получение АЧХ и ФЧХ для БИХ-фильтра
w, h = signal.freqz(b, a)
amp_response = np.abs(h)
phase_response = np.angle(h)

plt.plot(w, amp_response, label='amplitude response')
plt.plot(w, phase_response, label='phase response')
plt.title("Amplitude and Phase response (IIR, Butterworth, 1-order)")
plt.grid(True)
plt.legend()
plt.show()
plt.close()

# Генерация синусоиды
xs = np.arange(300)
sine_wave = np.sin(np.pi / 32 * xs)
plt.plot(sine_wave)
plt.title("Sine wave")
plt.show()
plt.close()

# Наложение фильтров на синусоиду
filtered_diff = signal.lfilter(b=diff_filter, a=[1.0], x=sine_wave)
filtered_butter = signal.lfilter(b, a, x=sine_wave)

# Изначально у фильтра Баттеруорда параметр критической частоты Wn == 0.7,
# что не давало видимого эффекта. Теперь там стоит 0.01, и эффект очевиден.
plt.plot(sine_wave, label='sine wave')
plt.plot(filtered_diff, label='difference filter (FIR, 1-order)')
plt.plot(filtered_butter, label='Butterworth filter (IIR, 1-order)')
plt.legend()
plt.title("Filtered Sinewave")
plt.show()
plt.close()
