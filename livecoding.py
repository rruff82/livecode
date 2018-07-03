# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:38:42 2018

@author: Ryan Ruff

This is my first live streaming session!

I'm going to attempt to create music using Python!

Let's start with some imports
"""

import numpy as np
import sounddevice as sd
import matplotlib
import matplotlib.pyplot as plt

tau = np.pi*2
sr = 44100

dur = 24
t = np.arange(0,dur,1/sr)



def plotfunc(fn,a=0,b=dur):
    x = np.arange(a,b,(b-a)/500)
    fig, graph = plt.subplots()

    graph.plot(x,fn(x))
    graph.grid()
    plt.show()
    return;

def playfunc(fn):

    
    out = fn(t)
    sd.play(out)
    return;    


def plotnplay(fn):
    plotfunc(fn,0,.01)
    playfunc(fn)
    return;
    
    
    
def A440(t):
    return np.sin(tau*440*t);

plotnplay(A440)

def envelope(x):
    return np.power((1-(x%.25)),15)


plot(envelope,0,1)

def mySound(t):
    return A440(t/2)*envelope(t);


def createSnare():
    whitenoise = np.random.uniform(-1,1,sr*dur) 
    snare_env = envelope(t)
    snare_pitch = np.sin(220*tau*t)
    return snare_env*snare_pitch*whitenoise;

snare = createSnare()
sd.play(snare)

def bassPitchShift(t):
    return (1-(t%.5));
plot(bassPitchShift,0,dur)

def bassEnv(t):
    return np.power((1-2*(t%.5)),.5);
plotfunc(bassEnv)

def bassSound(t):
    return np.sin(55*tau*(t%.5)*(1-.5*(t%.5)));

plotnplay(bassSound)

def createBass():
    return bassSound(t)*bassEnv(t);
bass = createBass()

sd.play(bass)

sd.play( (bass+snare)/2 )


def thick_sound(t):
    return (A440(t)+A440(2*t)/2+A440(3*t)/3+A440(4*t)/4+A440(5*t)/5+A440(6*t)/6+A440(8*t)/8)/2;

plotnplay(lambda x: thick_sound(x/2))

def majorChord(t):
    return (thick_sound(t/2)+thick_sound(t*5/4)+thick_sound(t*3/2)+thick_sound(t))/2

plotnplay(majorChord)

def pianoEnv(t):
    return 1-(t%1);


def pianoPitchShift(x):
    if (x < 4):
        return 1;
    elif (x < 8):
        return 1.5;
    elif (x < 12):
        return 1;
    elif (x < 16):
        return 4/3;
    elif (x < 20):
        return 3/2;
    else:
        return 1;

def createPiano():
    return pianoEnv(t)*[majorChord(a*pianoPitchShift(a)) for a in t];


piano = createPiano()

sd.play(piano)

sd.play( (piano+bass+snare)/3 )

# Thanks for watching!!!!!
# to be continued?


