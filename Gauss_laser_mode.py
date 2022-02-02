from LightPipes import *
import matplotlib.pyplot as plt
import os

wavelength = 500 * nm
size = 65 * mm
N = 200
w0 = 6 * mm
i = 0

# True - Laguerre-Gauss , False - Hermit-Gauss
LG = True

# m-порядок ; n - степень
n_max = 12
m_max = 64
if LG:
    s = r'Laguerre-Gauss laser modes'
else:
    s = r'Hermite-Gauss laser modes'


def save_to_file(F, I, Phi, s):
    plt.imshow(I, cmap='jet')
    plt.imsave(os.path.abspath('C:/Users/bekht/Desktop/diplom/HG2') + '/' + str(s) + '.png', I)
    # plt.show() #по очереди


F = Begin(size, wavelength, N)
for m in range(int(m_max / 2)):
    for n in range(n_max):
        F = GaussBeam(F, w0, LG=LG, n=n, m=m)
        I = Intensity(0, F)
        Phi = Phase(F)
        if LG:
            s = f'LG_{n}' + f'{m}' + " w=6 size=50"
        else:
            s = f'HG_{n}' + f'{m}' + " w=6 size=70"
        save_to_file(F, I, Phi, s)
    i += 1
