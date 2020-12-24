from tqdm.auto import tqdm
import soundfile
import librosa
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import ifft


# def convolve(signal1, signal2):
#
#     # Prepare
#     if len(signal2) > len(signal1):
#         signal1, signal2 = signal2, signal1
#     signal2 = signal2[::-1]
#
#     # Pad signal1
#     kernel_size = signal2.shape[0]
#     signal1 = np.hstack((kernel_size - 1, signal1, kernel_size))
#
#     # Convolve
#     res = []
#     for i in range(0, signal1.shape[0] - kernel_size):
#         res.append(np.dot(signal2, signal1[i: i + kernel_size]))
#
#     return np.array(res)


def convolve(s, f, step):
    return np.fromiter(
        (
            s[max(0, i - f.size):i] @  f[::-1][-i:]
            for i in tqdm(range(1, s.size + 1, step))
        ),
        np.float,
        count=len(list(range(1, s.size + 1, step)))
    )


def karplus_strong(wavetable, output_sample_len):
    curr_idx = 0
    last_value = 0
    output_signal = []
    while len(output_signal) < output_sample_len:
        last_value = 0.5 * (wavetable[curr_idx] + last_value)
        wavetable[curr_idx] = last_value
        output_signal.append(last_value)
        curr_idx = (curr_idx + 1) % wavetable.shape[0]
    return np.array(output_signal)


def task_save_with_different_discretization_rates(signal):
    # Saving in different sample rates
    sampling_rates = [44100, 30000, 16000, 8000]
    for _sr in sampling_rates:
        soundfile.write('audio_%s.wav' % _sr, signal, samplerate=_sr)


def task_low_freqs_filtration(signal):
    global variant_freq

    # Создание фильтра низких частот
    low_frequencies_filter = (np.arange(512) * sr / 1024 < variant_freq).astype(float)
    low_frequencies_filter = np.hstack((low_frequencies_filter, low_frequencies_filter[::-1]))

    # Извлечение импульсного отклика
    fir = ifft(low_frequencies_filter)

    # Свертка
    conv = convolve(signal, fir.real[:512], step=1)
    soundfile.write('lowpassfilter_audio.wav', conv, samplerate=sr)
    # conv = scipy.signal.convolve(signal, fir, method='direct')


    plt.figure(figsize=(10, 16))
    fig, axs = plt.subplots(nrows=1, ncols=1)
    # axs.plot(signal)
    # axs.set_title('Recorded audio')
    axs.plot(conv)
    axs.set_title('Filtered signal')
    # axs.plot(low_frequencies_filter)
    # axs.set_title('Low-freq Filter')
    # axs.plot(fir.real)
    # axs.set_title('FIR (real)')
    plt.show()


def task_apply_reverb(signal):
    raw_ir, _ = librosa.load('toybox_ir.wav')
    plt.plot(raw_ir)
    plt.title('toybox_ir')
    plt.show()

    rev_signal = convolve(signal, raw_ir, step=1)
    plt.plot(rev_signal)
    plt.title('Signal with reverb')
    plt.show()
    soundfile.write('rev_signal.wav', rev_signal, sr)


def task_karplus_strong():
    sample_rate = 16000
    wave_table_size = sample_rate // 80
    # wavetable = np.random.uniform(0, 2, wave_table_size)
    # wavetable = np.cos(2 * np.pi * 147 * np.arange(wave_table_size) - 0.8)
    wavetable = np.random.randint(0, 3, wave_table_size) - 1
    wavetable = wavetable.astype(np.float)
    plt.plot(wavetable)
    plt.show()

    generated_signal = karplus_strong(wavetable, sample_rate * 2)
    plt.plot(generated_signal)
    plt.show()
    soundfile.write('generated_ks_alg.wav', generated_signal, samplerate=sample_rate)


# Загрузка данных
signal, sr = librosa.load('audio.wav')
print("Input sampling rate:", sr)

# Частота для фильтра низких частот
variant_num = 8
variant_freq = variant_num * 100 + 500


# task_save_with_different_discretization_rates(signal)
# task_low_freqs_filtration(signal)
# task_apply_reverb(signal)
task_karplus_strong()