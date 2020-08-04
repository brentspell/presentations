import time

import streamlit as st
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

#mpl.style.use('ggplot')
#mpl.rcParams['figure.figsize'] = (4, 3)
plt.set_cmap('hot')

π = np.pi
fs = 48000

"""
## What is the Fourier Transform?
"""

f = 440
l = 2
t = np.arange(l*fs)/fs
y = np.cos(2*π*f*t)

fig, ax = plt.subplots()
ax.plot(y[:1000])
st.pyplot(fig)

f"""
energy: {(y**2).mean():.2f}
"""


fig, ax = plt.subplots()
ax.set_xlim(400, 500)
ax.set_ylim(0, 1)
plot, = ax.plot([0])
stplot = st.pyplot(fig)

i = 1
d = 1
while True:
    R = fs // f
    C = len(y) // R // 20
    N = min(i * R * 20, len(y))
    Y = np.fft.rfft(y[:N])/N
    print(i)
    plot.set_data(np.linspace(0, fs/2, N//2+1), np.abs(Y))
    stplot.pyplot(fig)
    time.sleep(0.1)
    i += d
    if i == C:
        d = -1
    if i == 1:
        d = 1

    time.sleep(0.01)