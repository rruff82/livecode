# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:38:42 2018
Live Coding a Music Synthesizer!

@author: Ryan Ruff

MIT License

Copyright (c) 2018 Ryan Ruff (rruff82@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.    
    
"""

import numpy as np
import sounddevice as sd
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal

tau = np.pi*2
sr = 44100

blocksize = 1024


def plotfunc(fn,a=0,b=1):
    x = np.linspace(a,b,500)
    fig, graph = plt.subplots()

    graph.plot(x,fn(x))
    graph.grid()
    plt.show()
    return;

def playfunc(fn,dur=1):    
    t = np.arange(0,dur,1/sr)
    out = fn(t)
    sd.play(out)
    return;    


def plotnplay(fn,dur=1):
    plotfunc(fn,0,.01)
    playfunc(fn,dur)
    return;
    
    
    
def A440(t):
    return np.sin(tau*440*t);



def envelope(x):
    return np.power((1-(x%.25)),15)




def mySound(t):
    return A440(t/2)*envelope(t);


def createSnare(t):
    whitenoise = np.random.uniform(-1,1,t.size) 
    snare_env = envelope(t)
    snare_pitch = np.sin(220*tau*t)
    return snare_env*snare_pitch*whitenoise;


def bassPitchShift(t):
    return (1-(t%.5));


def bassEnv(t):
    return np.power((1-2*(t%.5)),.5);


def bassSound(t):
    return np.sin(55*tau*(t%.5)*(1-.5*(t%.5)));



def createBass(t):
    return bassSound(t)*bassEnv(t);



def createDnB(t):
    return (createBass(t)+createSnare(t));

def thick_sound(t):
    return (A440(t)+A440(2*t)/2+A440(3*t)/3+A440(4*t)/4+A440(5*t)/5+A440(6*t)/6+A440(8*t)/8)/2;



def majorChord(t):
    return (thick_sound(t/2)+thick_sound(t*5/4)+thick_sound(t*3/2)+thick_sound(t))/2



def pianoEnv(t):
    return 1-(t%1);


def step(t):
    return np.heaviside(t,1);


def pianoPitchShift(x):
    return 1+(1/3)*step(x-8)-(1/3)*step(x-12)+(1/2)*step(x-16)-(1/6)*step(x-20);



def createPiano(t):
    return pianoEnv(t)*majorChord(t*pianoPitchShift(t%24));


def createBasicLoop(t):
    return (.5*createPiano(t)+.2*createSnare(t)+3*createBass(t))/4;



def createGuitarSound(t):
    return .20*signal.square(220*tau*t);



def createADSR(A,D,S,R):
    decay_rate = (1-S)/D;
    release_rate = S/R;
    return (lambda t,L: step(t)*t/A*(1-step(t-A))
            +step(t-A)*(2*A-t)*decay_rate*(1-step(t-A-D))
            +step(t-A-D)*S*(1-step(t-(L-R)))
            +step(t-(L-R))*(((L-R)-t)*release_rate+S)*(1-step(t-L)));
            


plotfunc(createADSR(.05,.025,.5,.2,1),0,1)


def createGuitar(t):
    guitar_env = createADSR(.05,.025,.5,.2)
    return createGuitarSound(t)*guitar_env(t,1);

plotnplay(createGuitar,2)


equal_temperament = {'A1': 1, 'A#1': 2**(1/12), 'B1': 2**(2/12),
                     'C1': 2**(3/12), 'C#1': 2**(4/12), 'D1':2**(5/12),
                     'D#1': 2**(6/12), 'E1':2**(7/12), 'F1': 2**(8/12),
                     'F#1': 2**(9/12), 'G1':2**(10/12), 'G#1':2**(11/12),
                     'A2': 2, 'A#2': 2**(13/12), 'B2': 2**(14/12),
                     'C2': 2**(15/12), 'C#2': 2**(16/12), 'D2':2**(17/12),
                     'D#2': 2**(18/12), 'E2':2**(19/12), 'F2': 2**(20/12),
                     'F#2': 2**(21/12), 'G2':2**(22/12), 'G#2':2**(23/12)}

def create_note(timbre,envolope,start,length,pitch):
    return (lambda t: timbre((t-start)*pitch)*envolope(t-start,length))

melody_time = 0

def zero_function(t):
    return 0;

def new_note(note,length):
    global melody_time
    note_start = melody_time
    melody_time = note_start+length
    return create_note(createGuitar,createADSR(.05,.025,.5,.2),note_start,length,equal_temperament[note])
    
def add_funcs(f1,f2):
    return (lambda t: (f1(t)+f2(t)))

note_sequence = zero_function

def append_note(note,length):
    global note_sequence;
    note_sequence = add_funcs(note_sequence,new_note(note,length))


def star_spangled_banner():
    global note_sequence;
    global melody_time;
    melody_time = 0;
    note_sequence = zero_function;
    melody = [('G1',.5),('E1',.5),('C1',1.5),('E1',.5),('G1',1),('C2',2),
              ('E2',.5),('D2',.5),('C2',1),('E1',1),('F#1',1),('G1',2),
              ('G1',.5),('G1',.5),('E2',1.5),('D2',.5),('C2',1),('B2',1.5),('A2',.5),
              ('B2',1),('C2',1),('C2',1),('G1',1),('E1',.5),('C1',1.5),
              ('C1',.5),('C1',.5),('C1',1.5),('E1',.5),('G1',1),('C2',2),
              ('E2',.5),('D2',.5),('C2',1),('E1',1),('F#1',1),('G1',2),
              ('G1',.5),('G1',.5),('E2',1),('D2',1),('C2',1),('B2',2),
              ('A2',.5),('B2',.5),('C2',.5),('C2',1.5),('G1',1),('E1',1),('C1',1),
              ('E2',.5),('E2',.5),('E2',1),('F2',1),('G2',1),('G2',2),
              ('F2',.5),('E2',.5),('D2',1),('E2',.5),('F2',.5),('F2',2),
              ('F2',1),('E2',1.5),('D2',.5),('C2',1),('B2',1.5),('A2',.5),('B2',1),
              ('C2',1),('E1',1),('F#1',1),('G1',2),
              ('G1',1),('C2',1),('C2',1),('C2',1),('A2',1),('A2',1),('A2',1),
              ('D2',1),('F2',.5),('E2',.5),('D2',.5),('C2',.5),('C2',2),
              ('G1',.5),('G1',.5),('C2',1.5),('D2',.5),('E2',.5),('F2',.5),
              ('G2',2),('E2',.5),('C2',.5),('E2',1.5),('F2',.5),('D2',1),('C2',2)]
    for n in melody:
        append_note(n[0],n[1])
    plotnplay(note_sequence,melody_time)
    




def callback(indata, outdata, frames, time, status):
    if status:
        print(status) 
    timedata = np.arange(time.outputBufferDacTime,time.outputBufferDacTime+frames/sr,1/sr)
    outdata[:,0] = note_sequence(timedata)
    return;
    
def finished_callback():
    print("we're done, now what?")

def playback_loop():
    with sd.Stream(channels=1, callback=callback, samplerate=sr, blocksize=blocksize, finished_callback=finished_callback):
        sd.sleep(int(dur * 1000))

playback_loop()

strm = sd.Stream(channels=1, callback=callback, samplerate=sr, blocksize=blocksize, finished_callback=finished_callback)
strm.start()

strm.stop()



