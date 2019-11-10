import scipy.io.wavfile
import numpy as np
import scipy.stats
import numpy.ma as ma

def main():
    fname_train = "PatternRecAs02.wav"
    energy = np.load(fname_train+"_energy.npy")

    is_silence = np.zeros(len(energy), bool)
    is_silence[energy < 10] = True # Good Threshold
    print(is_silence)

    p_silence = np.sum(is_silence) / len(is_silence)
    p_voice = 1 - p_silence

    silence_samples = energy[is_silence == True]
    voice_samples = energy[is_silence == False]

    silence_mean = np.mean(silence_samples)
    silence_dev = np.std(silence_samples)
    voice_mean = np.mean(voice_samples)
    voice_dev = np.std(voice_samples)

    silence_pdf = scipy.stats.norm(loc=silence_mean, scale=silence_dev)
    voice_pdf = scipy.stats.norm(loc=voice_mean, scale=voice_dev)

    np.save("distributions", [silence_mean, silence_dev, p_silence, voice_mean, voice_dev, p_voice])

    silence = np.empty(len(energy))
    silence_likelihood = np.empty(len(energy))
    voice_likelihood = np.empty(len(energy))

    for c in range(len(energy)):
        silence_likelihood[c] = silence_pdf.pdf(energy[c]) * p_silence
        voice_likelihood[c] = voice_pdf.pdf(energy[c]) * p_voice
        silence[c] = silence_likelihood[c] > voice_likelihood[c]

    np.save(fname_train+"_silence", silence)
    np.save(fname_train+"_silence_likelihood", silence_likelihood)
    np.save(fname_train+"_voice_likelihood", voice_likelihood)

    fname_eval = "PatternRecAs02_1.wav"
    energy = np.load(fname_eval+"_energy.npy")

    silence = np.empty(len(energy))
    silence_likelihood = np.empty(len(energy))
    voice_likelihood = np.empty(len(energy))

    for c in range(len(energy)):
        silence_likelihood[c] = silence_pdf.pdf(energy[c]) * p_silence
        voice_likelihood[c] = voice_pdf.pdf(energy[c]) * p_voice
        silence[c] = silence_likelihood[c] > voice_likelihood[c]

    np.save(fname_eval+"_silence", silence)
    np.save(fname_eval+"_silence_likelihood", silence_likelihood)
    np.save(fname_eval+"_voice_likelihood", voice_likelihood)


if __name__ == '__main__':
    main()
