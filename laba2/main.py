import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from skimage.util import view_as_windows


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


def stft(signal, win_len, win_step, window_function='hanning'):
    # Get frames
    frames = view_as_windows(signal, window_shape=(win_len,), step=win_step)

    # Apply windowing function
    w_funcs = {'hanning': np.hanning, 'hamming': np.hamming, 'bartlett': np.bartlett, 'kaiser': np.kaiser}
    win = w_funcs[window_function](win_len + 1)[:-1]
    frames = frames * win

    # Apply fft per frame
    frames = frames.T
    fourier_image = fft(frames, axis=0)[:win_len // 2 + 1: -1]
    fourier_image = np.abs(fourier_image)
    return fourier_image


def plot_spectrogram(spectrum, signal_len, sample_rate, save_name=None):
    spectrum = 20 * np.log10(spectrum / np.max(spectrum))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(spectrum, origin='lower', cmap='viridis', extent=(0, signal_len / sample_rate, 0, sample_rate / 2 / 1000))
    ax.axis('tight')
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Frequency [kHz]')

    if save_name is not None:
        plt.savefig('%s.png' % save_name, dpi=250)
    else:
        plt.show()



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
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.savefig('%s_dft.png' % signal_name, dpi=250)


def task_train_wistle():
    signal, sr = librosa.load('train_whistle.wav')
    plt.plot(signal)
    plt.savefig('train_wistle.png', dpi=250)

    # fi = dft(signal) # Слишком медленно и неэффективно
    fi = fft(signal)
    magnitude, phase = np.abs(fi), np.angle(fi)

    sorted_magnitude = np.sort(magnitude, axis=0)
    print("Main harmonics (hz):", sorted_magnitude[-3:] / signal.shape[0] * sr)
    # Main harmonics (hz): [624.95316897 712.7378867  712.7378867 ]


    _, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(magnitude[:fi.shape[0] // 2])
    axs[0].set_title('Magnitude')
    axs[1].plot(phase[:fi.shape[0] // 2])
    axs[1].set_title('Phase')
    plt.savefig('train_wistle_dft.png', dpi=250)


def task_sputnik():
    signal, sr = librosa.load('sputnik_1.wav')
    N = signal.shape[0]

    # FFT
    # fourier_image = dft(signal) # Слишком медленно на длинных сигналах
    fourier_image = fft(signal)

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    axs[0].plot(np.arange(N) / sr, signal, linewidth=0.5, c='g')
    axs[0].set_xlabel('Time [sec]')
    axs[0].set_ylabel('Amplitude')

    axs[1].plot(np.abs(fourier_image)[:N // 2] / N * sr, linewidth=0.5)
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_ylabel('Magnitude value')

    axs[2].plot(np.angle(fourier_image)[:N // 2] / N * sr, linewidth=0.5)
    axs[2].set_xlabel('Frequency [Hz]')
    axs[2].set_ylabel('Phase value')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig('sputnik_fft.png', dpi=250)

    # Get spectrum and convert it to decibels

    # Чем больше win_len - тем сильнее размазывается спектрограмма вдоль оси частот
    spectrum = stft(signal, win_len=512, win_step=64, window_function='hanning')
    plot_spectrogram(spectrum, N, sr, 'sputnik_spec')


# task_simple_signals('unit_impulse')
# task_train_wistle()
task_sputnik()
