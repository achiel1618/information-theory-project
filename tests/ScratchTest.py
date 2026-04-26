import numpy as np
import math
import wave
from AudioCD import AudioCD

def load_wav(filename):
    wave_object = wave.open(filename, 'rb')
    Fs = wave_object.getframerate()
    nch = wave_object.getnchannels()

    data = np.frombuffer(wave_object.readframes(wave_object.getnframes()), dtype=np.int16)
    data = data / 2**15

    ch_1 = data[0::nch]
    ch_2 = data[1::nch]
    audiofile = np.transpose(np.vstack((ch_1, ch_2)))

    return Fs, audiofile


def apply_periodic_scratches(cd, scratch_length, period=600000, start=30000):
    for loc in range(start, cd.cd_bits.size, period):
        cd.scratchCd(scratch_length, loc)


Fs, audiofile = load_wav("../Hallelujah.wav")

scratch_lengths = [100, 3000, 10000]
configs = [0, 1, 2, 3]

for scratch_length in scratch_lengths:
    print(f"\nScratch length: {scratch_length} bits")

    for config in configs:
        cd = AudioCD(Fs, config, max_interpolation=8)
        cd.writeCd(audiofile)

        apply_periodic_scratches(cd, scratch_length)

        out, interpolation_flags = cd.readCd()

        n_total = interpolation_flags.size
        n_erased = np.sum(interpolation_flags != 0)
        n_failed = np.sum(interpolation_flags == -1)

        p_erased = int(n_erased) / n_total
        p_failed = int(n_failed) / n_total

        print(
            f"Config {config}: "
            f"P(erasure flagged) = {p_erased:.6e}, "
            f"P(interpolation failed) = {p_failed:.6e}"
        )