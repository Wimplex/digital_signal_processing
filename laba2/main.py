from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import librosa
import soundfile

from scipy.fft import fft, ifft, dct
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
    w_funcs = {'hanning': np.hanning, 'hamming': np.hamming, 'bartlett': np.bartlett,
               'kaiser': partial(np.kaiser, beta=3), 'blackman': np.blackman}
    win = w_funcs[window_function](win_len + 1)[:-1]
    frames = frames * win

    # Apply fft per frame
    frames = frames.T
    fourier_image = fft(frames, axis=0, workers=8)[:win_len // 2 + 1:-1]
    fourier_image = np.abs(fourier_image)
    return fourier_image


def _compute_filterbank(n_mfcc, win_len, sample_rate):

    # Declare convertation functions
    freq_to_mel = lambda x: 2595.0 * np.log10(1.0 + x / 700.0)
    mel_to_freq = lambda x: 700 * (10 ** (x / 2595.0) - 1.0)

    # Compute filterbank markup
    mel_min = freq_to_mel(0)
    mel_max = freq_to_mel(sample_rate)
    mels = np.linspace(mel_min, mel_max, n_mfcc)
    freqs = mel_to_freq(mels)
    filter_points = np.floor(freqs * (win_len // 2 + 1) / sample_rate).astype(np.int)

    # Compute filters
    filters = np.zeros([filter_points.shape[0], int(win_len / 2 + 1)])
    for i in range(1, filter_points.shape[0] - 1):
        filters[i, filter_points[i - 1]: filter_points[i]] = np.linspace(0, 1, filter_points[i] - filter_points[i - 1])
        filters[i, filter_points[i]: filter_points[i + 1]] = np.linspace(1, 0, filter_points[i + 1] - filter_points[i])

    filters = filters[:, :-1]
    return filters


def mfcc(n_mfcc, signal, sample_rate, win_len, win_step, window_function, use_dct=True):
    # Split into frames and apply window function
    frames = view_as_windows(signal, window_shape=(win_len,), step=win_step)
    w_funcs = {'hanning': np.hanning, 'hamming': np.hamming, 'bartlett': np.bartlett,
               'kaiser': partial(np.kaiser, beta=3), 'blackman': np.blackman}
    frames = frames * w_funcs[window_function](win_len + 1)[:-1]

    # Compute power spectrum (peridogram)
    frames = frames.T
    spectrum = fft(frames, axis=0, workers=8)[:int(win_len / 2)]
    spectrum = np.flip(spectrum, axis=0)
    power_spectrum = np.abs(spectrum) ** 2

    # Compute mel-filterbank
    filterbank = _compute_filterbank(n_mfcc, win_len, sample_rate)
    print(filterbank.shape)

    # Apply filterbank
    filtered_spectrum = np.dot(filterbank, power_spectrum).T
    log_spectrum = 10.0 * np.log10(filtered_spectrum)
    log_spectrum = log_spectrum[:, 1:-1]

    # Extract mfcc using dct-II
    mfcc = dct(log_spectrum, type=2, n=n_mfcc, workers=-1)

    return mfcc


def plot_spectrogram(spectrum, signal_len, sample_rate, normalize=False, save_name=None):
    if normalize:
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
    print("Main harmonics (hz):", sorted_magnitude[-10:] / signal.shape[0] * sr)
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
    fourier_image = fft(signal, workers=8)

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    axs[0].plot(np.arange(N) / sr, signal, linewidth=0.5, c='g')
    axs[0].set_xlabel('Time [sec]')
    axs[0].set_ylabel('Amplitude')

    axs[1].plot(np.abs(fourier_image)[:N // 2], linewidth=0.5)
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_ylabel('Magnitude value')

    axs[2].plot(np.angle(fourier_image)[:N // 2], linewidth=0.5)
    axs[2].set_xlabel('Frequency [Hz]')
    axs[2].set_ylabel('Phase value')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig('sputnik_fft.png', dpi=250)

    # Get spectrum and convert it to decibels

    # Чем больше win_len - тем сильнее размазывается спектрограмма вдоль оси частот
    spectrum = stft(signal, win_len=512, win_step=64, window_function='hanning')
    plot_spectrogram(spectrum, N, sr, 'sputnik_spec')


def task_telephone_number():
    signal, sr = librosa.load('dtmf.wav')
    spectrum = stft(signal, win_len=1024, win_step=32, window_function='blackman')
    plot_spectrogram(spectrum, signal_len=signal.shape[0], sample_rate=sr, save_name='telephone_number_spectrum')

    seconds_segments = [(0, 2), (4, 6), (8, 10)]
    for i, (min_s, max_s) in enumerate(seconds_segments):
        sig = signal[min_s * sr:max_s * sr]
        spectrum = fft(sig)[:sig.shape[0] // 2]

        magnitude_spec = np.abs(spectrum)
        uniques, idx = np.unique(magnitude_spec, return_index=True)
        biggest_harmonics = idx[np.argsort(uniques)[-2:]]
        print("#%s num harmonics: %s" % (i + 1, biggest_harmonics / sig.shape[0] * sr))
        # #1 num harmonics: [1209.  697.]
        # #2 num harmonics: [ 770. 1336.]
        # #3 num harmonics: [1477.  852.]
        # Что соответствует последовательности: 1-5-9


def task_informative_spectrum():
    signal, sr = librosa.load('bach_orig.wav')

    spectrum = fft(signal)
    magnitude, phase = np.abs(spectrum), np.angle(spectrum)

    only_phase_spectrum = np.ones_like(magnitude) * np.exp(1j * phase)
    phase_signal = np.real(ifft(only_phase_spectrum))
    soundfile.write('phase_bach.wav', phase_signal, samplerate=sr)

    only_magnitude_spectrum = magnitude * np.exp(1j * np.zeros_like(phase))
    magnitude_signal = np.real(ifft(only_magnitude_spectrum))
    soundfile.write('magnitude_bach.wav', magnitude_signal, samplerate=sr)

    # Ответ: на слух, больше полезной информации содержит в себе именно фазовый спектр


def task_mfcc():
    signal, sr = librosa.load('voice.wav')

    # Extract mfcc using my algorithm
    coefs = mfcc(
        n_mfcc=23,
        signal=signal,
        sample_rate=sr,
        win_len=2048,
        win_step=512,
        window_function='hamming',
        use_dct=False
    ).T
    plt.figure(figsize=(10, 5))
    plt.title("23 MFCC (My)")
    plt.imshow(coefs, aspect='auto', origin='lower', cmap='inferno')
    plt.show()

    # Extract mfcc using librosa api
    coefs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=23)
    plt.figure(figsize=(10, 5))
    plt.title("23 MFCC (librosa)")
    plt.imshow(coefs, aspect='auto', origin='lower', cmap='inferno')
    plt.show()



# task_simple_signals('unit_impulse')
# task_train_wistle()
# task_sputnik()
# task_telephone_number()
# task_informative_spectrum()
task_mfcc()
