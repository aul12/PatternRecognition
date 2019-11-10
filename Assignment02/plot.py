import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import numpy.ma as ma

def classification_over_time(base_name):
    energy = np.load(base_name+"_energy.npy")
    silence = np.load(base_name+"_silence.npy")

    silence_samples = ma.masked_where(silence != True, energy)
    voice_samples = ma.masked_where(silence != False, energy)

    plt.figure()
    plt.plot(np.arange(0,len(energy)), silence_samples, 'g')
    plt.plot(np.arange(0,len(energy)), voice_samples, 'r')
    plt.legend(["Silence", "Voice"])
    plt.title("Classification over time based on energy feature (" + base_name + ")")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.show()

def likelihood_over_time(base_name):
    silence_likelihood = np.load(base_name+"_silence_likelihood.npy")
    voice_likelihood = np.load(base_name+"_voice_likelihood.npy")
    plt.figure()
    plt.plot(silence_likelihood)
    plt.plot(voice_likelihood)
    plt.legend(["Silence", "Voice"])
    plt.title("Scaled Likelihoods over time (" +base_name+")")
    plt.xlabel("Time")
    plt.ylabel("Scaled Likelihood")
    plt.show()


def main():
    fname_train = "PatternRecAs02.wav"
    fname_eval = "PatternRecAs02_1.wav"

    classification_over_time(fname_train)
    classification_over_time(fname_eval)

    silence_mean, silence_dev, p_silence, voice_mean, voice_dev, p_voice = np.load("distributions.npy")
    silence_pdf = scipy.stats.norm(loc=silence_mean, scale=silence_dev)
    voice_pdf = scipy.stats.norm(loc=voice_mean, scale=voice_dev)

    x = np.arange(-10,55,0.1)
    plt.figure()
    plt.plot(x, silence_pdf.pdf(x) * p_silence)
    plt.plot(x, voice_pdf.pdf(x) * p_voice)
    plt.legend(["Silence", "Voice"])
    plt.title("Scaled likelihood for classes")
    plt.xlabel("Energy")
    plt.ylabel("Scaled likelihood")
    plt.show()

    likelihood_over_time(fname_train)
    likelihood_over_time(fname_eval)


if __name__ == '__main__':
    main()
