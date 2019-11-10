import scipy.io.wavfile
import numpy as np

def main():
    fname = "PatternRecAs02_1.wav"
    rate, stereo_data = scipy.io.wavfile.read(fname)
    data = stereo_data[:,0]
    print(rate)
    print(data.shape)
    data_min = np.min(data)
    data_max = np.max(data)
    data_normalized = list(map(lambda x:  -x/data_min if x < 0 else x/data_max, data))

    window_size = 0.01
    window_samples = int(round(window_size * rate))

    print(window_samples)
    data_normalized_squared = np.square(data_normalized)

    energy = np.zeros(len(data_normalized))

    for t in range(len(stereo_data)):
        for i in range(t, min(t+window_samples, len(stereo_data))):
            energy[t] += data_normalized_squared[i]

    np.save(fname+"_energy", energy)



if __name__ == '__main__':
    main()
