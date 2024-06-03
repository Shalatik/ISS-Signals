# Simona Ceskova xcesko00
# 28.12.2021

import math
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
from scipy.io import wavfile

# 1 .ukol

FS, DATA = wavfile.read('../audio/xcesko00.wav')
DATA = DATA/FS  # normovani
TIME = np.arange(DATA.size) / FS
# print(np.arange(DATA.size))
# DATA.min(), DATA.max()

# print(f"vzorkovaci fekvence: {fs}")
print(f"pocet vzorku {DATA.size}")
print(f"pocet sekund {DATA.size/FS}")
print(f"max {DATA.max()}")
print(f"min {DATA.min()}")

plt.figure(figsize=(10, 5))
plt.plot(TIME, DATA)

plt.gca().set_xlabel('$t [s]$')
plt.gca().set_title('Zvukovy signal')
plt.tight_layout()

# 2 .ukol

# odecteni stedni hodnoty: DATA - np.mean(DATA)
# normalizovani delenim maximem absolutni hodnoty
DATA = (DATA - np.mean(DATA)) / abs(np.max(DATA - np.mean(DATA)))
print(f"Interval: {DATA.min(), DATA.max()}")

plt.figure(figsize=(10, 5))
plt.plot(TIME, DATA)
plt.gca().set_xlabel('$t [s]$')
# plt.gca().set_title('zvukovy signal')
plt.tight_layout()

MATRIX = []  # matice do ktere se ukladaji ramce jako sloupce
frame = 1024
overlap = 512
for i in range(int(math.ceil(DATA.size/frame))):  # pro velikost matice
    MATRIX.append(DATA[overlap*i: overlap*i + frame])  # prekryti 512 vzorku
    # print(MATRIX[i])

plt.figure(figsize=(10, 5))
plt.plot(np.arange(MATRIX[31].size) / FS, MATRIX[31])
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Znely ramec signalu')
plt.tight_layout()

# 3 .ukol


def DFT_function(sample):
    n = np.arange(frame)  # zpracovani jednotlivych vzorku do matice
    k = n.reshape(frame, 1)  # aktualni frekvence matice
    e = np.exp(-2j*np.pi*k*n/frame)  # vypocitani mocniny e
    sample = np.dot(e, sample)
    return sample


DFT_MATRIX = DFT_function(MATRIX[0])  # pouziti funkce na vytvoreni matice
F = (np.arange(frame/2))/(frame/FS)

print(np.allclose(DFT_MATRIX.real, (np.fft.fft(MATRIX[0])).real))

plt.figure(figsize=(10, 5))
plt.plot(F[:overlap], abs(DFT_MATRIX)[:overlap])
plt.gca().set_xlabel('$Frekvence [Hz]$')
plt.gca().set_title('DFT')
plt.tight_layout()

# 4 .ukol

f, TIME, sgr = spectrogram(DATA, FS, nperseg=1024, noverlap=512)
# +1e-20 kvuli nulam v logaritmu
sgr_log = 10 * np.log10((sgr+1e-20)**2)  # 10*log10[X[k]]^2

plt.figure(figsize=(10, 5))
plt.pcolormesh(TIME, f, sgr_log)
plt.gca().set_xlabel('t [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralni hustota vykonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()

# 5 .ukol


def cos_collect():
    cos_freqs = []
    for i in range(len(DFT_MATRIX[:512])):
        if abs(DFT_MATRIX[i]) > 12:
            cos_freqs.append(F[i])
    return cos_freqs


cos_matrix = cos_collect()
for i in cos_matrix:
    print(i)

# 6 .ukol

TIME = np.arange(DATA.size) / FS
cos_omegas = []
for i in cos_matrix:
    cos_omegas.append(np.cos((2*np.pi*i) * TIME))

cos_data = 0
for i in cos_omegas:
    cos_data = cos_data + i

# plt.plot(TIME,cos_data)
# plt.gca().set_xlabel('t [s]')
# plt.show()

wavfile.write("../audio/4cos.wav", FS, cos_data)

f, t, sgr = spectrogram(cos_data, FS)
sgr_log = 10 * np.log10(sgr+1e-20)

plt.figure(figsize=(10, 5))
plt.pcolormesh(t, f, sgr_log)
plt.gca().set_xlabel('t [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralni hustota vykonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()

# 7 .ukol

# 1
omega_filter = []
n_matrix = []
n_friend = []

for i in range(4):
    omega_filter = 2*np.pi*cos_matrix[i]/FS
    n_matrix.append(np.e**(omega_filter*1j))
    n_friend.append(np.conj(n_matrix[i]))

# koeficienty filteru
K_filter = np.poly((n_matrix[0], n_friend[0], n_matrix[1], n_friend[1],
                   n_matrix[2], n_friend[2], n_matrix[3], n_friend[3]))
print(K_filter)

# na vykresleni jsou potreba
b = K_filter
a = [1, 0, 0, 0, 0, 0, 0, 0, 0]

N_imp = 32
imp = [1, *np.zeros(N_imp-1)]
h = lfilter(b, a, imp)

plt.figure(figsize=(10, 5))
plt.stem(np.arange(N_imp), h, basefmt=' ')
plt.gca().set_xlabel('$n$')
plt.gca().set_title('Impulsní odezva $h[n]$')
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()

# 8.ukol

b = K_filter
a = [1, 0, 0, 0, 0, 0, 0, 0, 0]

# impulsni odezva
N_imp = 32
imp = [1, *np.zeros(N_imp-1)]
h = lfilter(b, a, imp)

# frekvencni charakteristika
w, H = freqz(b, a)

# nuly, poly
z, p, k = tf2zpk(b, a)

# stabilita
is_stable = (p.size == 0) or np.all(np.abs(p) < 1)

# filtrace
sf = lfilter(b, a, DATA)
f, t, sfgr = spectrogram(sf, FS)
sfgr_log = 10 * np.log10(sfgr+1e-20)


plt.figure(figsize=(4, 3.5))

# jednotkova kruznice
ang = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(ang), np.sin(ang))

# nuly, poly
plt.scatter(np.real(z), np.imag(z), marker='o',
            facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='poly')

plt.gca().set_xlabel('Realna slozka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginarni slozka $\mathbb{I}\{$z$\}$')

plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper left')

plt.tight_layout()

# 9 .ukol

_, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(w / 2 / np.pi * FS, np.abs(H))
ax[0].set_xlabel('Frekvence [Hz]')
ax[0].set_title('Modul frekvencni charakteristiky $|H(e^{j\omega})|$')

ax[1].plot(w / 2 / np.pi * FS, np.angle(H))
ax[1].set_xlabel('Frekvence [Hz]')
ax[1].set_title(
    'Argument frekvenkcni charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')

for ax1 in ax:
    ax1.grid(alpha=0.5, linestyle='--')

plt.tight_layout()

# 10 .ukol

TIME_new = np.arange(sf.size) / FS
new = sf / abs(sf.max())

plt.figure(figsize=(10, 5))
plt.plot(TIME_new, new)
plt.gca().set_xlabel('$t[data_norm]$')
plt.gca().set_title('Zvukovy signal')
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.pcolormesh(t, f, sfgr_log)
plt.gca().set_title('Spektrogram vyfiltrovaneho signalu')
plt.gca().set_xlabel('Cas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralni hustota vykonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()

wavfile.write("../audio/clean_z.wav", FS, new)
