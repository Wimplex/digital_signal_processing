import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft


def dft(signal, memory_ineffective=False):
    N = signal.shape[0]
    n = np.arange(N).reshape(-1, 1)

    if memory_ineffective:
        idx = np.dot(n, n.T)
        arg = 2 * np.pi * idx / N
        s = np.exp(-1j * arg)
        fourier_image = np.dot(signal, s)
    else:
        arg = 2 * np.pi * n / N
        fourier_image = np.fromiter(
            [np.dot(signal, np.exp(-1j * i * arg)) for i in range(N)],
            np.complex,
            count=N
        )

    return fourier_image

def task_simple_signals(signal_name):
    # Unit impulse
    unit_impulse = np.zeros((99, ))
    unit_impulse[50] = 1

    # Unit step
    a = np.zeros((49,)).squeeze()
    b = np.ones((50,)).squeeze()
    unit_step = np.concatenate((a, b))

    # Sine wave
    sine_wave = np.sin(0.1 * np.arange(100))

    signals = {
        'unit_impulse': unit_impulse,
        'unit_step': unit_step,
        'sine_wave': sine_wave
    }

    signal = signals[signal_name]
    plt.plot(signal, linewidth=1)
    plt.title(signal_name)
    plt.savefig('%s.png' % signal_name, dpi=250)

    fourier_image = dft(signal)
    magnitude, phase = np.abs(fourier_image), np.angle(fourier_image)

    fourier_image = fft(signal)
    magnitude1, phase1 = np.abs(fourier_image), np.angle(fourier_image)

    plt.figure(figsize=(16, 10))
    plt.title(signal_name)
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].plot(magnitude)
    axs[0, 0].set_title('Magnitude (my)')
    axs[1, 0].plot(phase)
    axs[1, 0].set_title('Phase (my)')
    axs[0, 1].plot(magnitude1)
    axs[0, 1].set_title('Magnitude (scipy)')
    axs[1, 1].plot(phase1)
    axs[1, 1].set_title('Phase (scipy)')
    plt.savefig('%s_dft.png' % signal_name, dpi=250)


def task_train_wistle():
    signal, sr = librosa.load('train_whistle.wav')
    plt.plot(signal)
    plt.savefig('train_wistle.png', dpi=250)

    fi = dft(signal)
    magnitude, phase = np.abs(fi), np.angle(fi)

    _, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(magnitude)
    axs[0].set_title('Magnitude')
    axs[1].plot(phase)
    axs[1].set_title('Phase')
    plt.savefig('train_wistle_dft.png', dpi=250)


# task_simple_signals('unit_impulse')
task_train_wistle()
