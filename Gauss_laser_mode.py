from LightPipes import *
import matplotlib.pyplot as plt
import os

wavelength = 500 * nm
size = 15 * mm
N = 200
w0 = 3 * mm
i = 0

# True - Laguerre-Gauss , False - Hermit-Gauss
LG = True

# m-порядок ; n - степень
n_max = 3
m_max = 16
if LG:
    s = r'Laguerre-Gauss laser modes'
else:
    s = r'Hermite-Gauss laser modes'


def save_to_file(F, I, Phi, s):
    plt.imshow(I, cmap='jet')
    plt.title(s)
    plt.axis('off')
    plt.savefig(os.path.abspath('C:/Users/bekht/PycharmProjects/vkr/dataset') + '/' + str(s) + '.jpg',
                bbox_inches='tight')
    plt.show() #по очереди


F = Begin(size, wavelength, N)
for m in range(int(m_max / 2)):
    for n in range(n_max):
        F = GaussBeam(F, w0, LG=LG, n=n, m=m)
        I = Intensity(0, F)
        Phi = Phase(F)
        if LG:
            s = f'LG_{n}' + f'{m}'
        else:
            s = f'HG_{n}' + f'{m}'
        save_to_file(F, I, Phi, s)
    i += 1
