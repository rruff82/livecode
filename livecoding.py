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
    

TODO:
    Eliminate "click" when starting new measure

"""

import numpy as np
import sounddevice as sd
import matplotlib
import matplotlib.pyplot as plt
import queue
from scipy import signal

tau = np.pi*2
sr = 44100

blocksize = 1024

equal_temperament = {'A1': 1, 'A#1': 2**(1/12), 'B1': 2**(2/12),
                     'C1': 2**(3/12), 'C#1': 2**(4/12), 'D1':2**(5/12),
                     'D#1': 2**(6/12), 'E1':2**(7/12), 'F1': 2**(8/12),
                     'F#1': 2**(9/12), 'G1':2**(10/12), 'G#1':2**(11/12),
                     'A2': 2, 'A#2': 2**(13/12), 'B2': 2**(14/12),
                     'C2': 2**(15/12), 'C#2': 2**(16/12), 'D2':2**(17/12),
                     'D#2': 2**(18/12), 'E2':2**(19/12), 'F2': 2**(20/12),
                     'F#2': 2**(21/12), 'G2':2**(22/12), 'G#2':2**(23/12)}
                     
current_key = equal_temperament['C1']


def zero_function(t):
    return 0;

def add_funcs(f1,f2):
    return (lambda t: (f1(t)+f2(t)))

def step(t):
    return np.heaviside(t,1);

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

def major_chord(t):
    return (A440(t/2)+A440(t*5/4)+A440(t*3/2)+A440(t))/2

def minor_chord(t):
    return (A440(t/2)+A440(t*6/5)+A440(t*3/2)+A440(t))/2

def dim7_chord(t):
    return (A440(t/2)+A440(t*6/5)+A440(t*7/5)+A440(5/3)+A440(t))/2
    

standard_chords = [major_chord,(lambda x: minor_chord(x*2**(2/12))),
                   (lambda x: major_chord(x*2**(4/12))),(lambda x: major_chord(x*2**(5/12))),
                   (lambda x: major_chord(x*2**(7/12))),(lambda x: minor_chord(x*2**(9/12))),
                   (lambda x: dim7_chord(x*2**(11/12)))]



class Phrase: 
    beats_per_measure = 4
    measures_per_phrase = 12
    beats_per_minute = 120
    start_time = 0
    prerender = None

    def seconds_per_beat(self):
        return 60/self.beats_per_minute;

    def seconds_per_measure(self):
        return self.seconds_per_beat()*self.beats_per_measure;

    def seconds_per_phrase(self):
        return self.seconds_per_measure()*self.measures_per_phrase;
    
    def get_func_by_measure(self,i):
        return np.full(int(self.seconds_per_measure()*sr),0);
    
    def get_measure_by_time(self,t):
        return np.floor_divide(t-self.start_time,self.seconds_per_measure()) % self.measures_per_phrase;
    
    def get_measure_start_time(self,i):
        return self.start_time+i*self.seconds_per_measure();  
    
    def render_phrase(self):
        for i in range(self.measures_per_phrase):
           t = np.arange(self.get_measure_start_time(i),self.get_measure_start_time(i)+self.seconds_per_measure(),1/sr)
           fn = self.get_func_by_measure(i)
           if (i == 0):
               out = fn(t)
           else:
               out = np.append(out,fn(t))
        return out;
    
    
    def prerender(self):
        for i in range(self.measures_per_phrase):
            t = np.arange(self.get_measure_start_time(i),self.get_measure_start_time(i)+self.seconds_per_measure(),1/sr)
            fn = self.get_func_by_measure(i)
            if (i == 0):
                self.prerender = fn(t)
            else:
                self.prerender = np.append(self.prerender,fn(t))
            
    def get_prerender_by_time(self,t):
        if (self.prerender is None):
            self.prerender()
        start_frame = int((t[0]-self.start_time%self.seconds_per_phrase())/sr)
        end_frame = int((t[-1]-self.start_time%self.seconds_per_phrase())/sr)
        
        if (end_frame > start_frame):
            self.prerender[start_frame:end_frame]
        else:
            np.append(self.prerender[start_frame:],self.prerender[:end_frame])
            
    
            
    


null_phrase = Phrase();


class DrumAndBass(Phrase):
    def snare_envelope(self,x):
        return np.power((1-(x%1)),15)
    
    def mySound(self,t):
        return A440(t/2)*self.snare_envelope(t);
    
    
    def createSnare(self,t):
        whitenoise = np.random.uniform(-1,1,t.size) 
        snare_env = self.snare_envelope(t/self.seconds_per_beat())
        snare_pitch = np.sin(220*tau*t*current_key)
        return snare_env*snare_pitch*whitenoise;


    def bassEnv(self,t):
        return np.power((1-(t%(2*self.seconds_per_beat()))/(2*self.seconds_per_beat())),2);
    
    
    def bassFreq(self,t):
        return 1-(t%(2*self.seconds_per_beat()))/(4*self.seconds_per_beat());
    
    
    def bassSound(self,t):
        return np.sin(55*tau*(t%(2*self.seconds_per_beat()))*current_key*self.bassFreq(t));
    
    
    def createBass(self,t):
        return self.bassSound(t)*self.bassEnv(t);
    
    def get_func_by_measure(self,i):
        return (lambda t: self.createBass(t-self.start_time)+self.createSnare(t-self.start_time));
    





dnb_phrase = DrumAndBass();


dnb_phrase.prerender()

sd.play(dnb_phrase.get_prerender_by_time(np.arange(0,1,1/sr)))

#phrase_test = dnb_phrase.render_phrase()
#sd.play(phrase_test)


class SimpleBlues(DrumAndBass):
    def thick_sound(self,t):
        return (A440(t)+A440(2*t)/2+A440(3*t)/3+A440(4*t)/4+A440(5*t)/5+A440(6*t)/6+A440(8*t)/8)/2;

#    def majorChord(self,t):
#        return (self.thick_sound(t/2)+self.thick_sound(t*5/4)+self.thick_sound(t*3/2)+self.thick_sound(t))/2

    def majorChord(self,t):
        return (A440(t/2)+A440(t*5/4)+A440(t*3/2)+A440(t))/2



    def pianoEnv(self,t):
        return 1-((t/self.seconds_per_beat())%1);


    def pitch_shift_by_measure(self,x):
        return 1+(1/3)*step(x-4)-(1/3)*step(x-6)+(1/2)*step(x-8)-(1/6)*step(x-10);

    def createPiano(self,t,i):
        return self.pianoEnv(t-self.get_measure_start_time(i))*self.majorChord(current_key*(t-self.get_measure_start_time(i))*self.pitch_shift_by_measure(i));

    def get_func_by_measure(self,i):
        dnb = super().get_func_by_measure(i)
        return (lambda t: (.5*self.createPiano(t,i)+2*dnb(t))/3);

piano_phrase = SimpleBlues()



#phrase_test = piano_phrase.render_phrase()
#sd.play(phrase_test)

"""
Standard Blues Progression
        I    __
        vi  /\
     /        \
    /         _\/
  \/_
V     ----->    IV    
vii^0           ii

"""


blues_matrix= np.matrix([[2,1,0,1,1,1,1],
                         [1,1,0,1,0,1,0],
                         [0,0,0,0,0,0,0],
                         [1,1,0,1,0,1,0],
                         [0,1,0,1,1,0,1],
                         [1,1,0,1,1,1,1],
                         [0,1,0,1,1,0,1]])
    
blues_matrix_major = np.matrix([[2,0,0,1,1,0,0],
                                [1,0,0,1,0,0,0],
                                [0,0,0,0,0,0,0],
                                [1,0,0,1,0,0,0],
                                [0,0,0,1,1,0,0],
                                [1,0,0,1,1,0,0],
                                [0,0,0,1,1,0,0]])
    
blues_matrix_minor = np.matrix([[0,1,0,0,0,1,1],
                                [0,1,0,0,0,1,0],
                                [0,0,0,0,0,0,0],
                                [0,1,0,0,0,1,0],
                                [0,1,0,0,0,0,1],
                                [0,1,0,0,0,2,1],
                                [0,1,0,0,0,0,1]])
    
major_weight = 1
minor_weight = .5

def get_weighted_blues_matrix():
    return blues_matrix_major*major_weight+blues_matrix_minor*minor_weight;

    
    
index_to_number = np.matrix([[1,0,0,0,0,0,0],
                             [0,2,0,0,0,0,0],
                             [0,0,3,0,0,0,0],
                             [0,0,0,4,0,0,0],
                             [0,0,0,0,5,0,0],
                             [0,0,0,0,0,6,0],
                             [0,0,0,0,0,0,7]])
    
start_vec = np.matrix([[1,0,0,0,0,0,0]])

edge_choices = np.matmul(start_vec,blues_matrix)
np.matmul(edge_choices,index_to_number)


def random_walk(pos,adj_mat):
    start_vec = np.full(adj_mat.shape[0],0)
    start_vec[pos-1] = 1
    edge_choices = np.matmul(np.matrix([start_vec]),adj_mat)
    weights = np.array(edge_choices)[0]
    weights = weights/ np.sum(weights)
    return np.random.choice([1,2,3,4,5,6,7],p=weights)
    

def create_walk(start_pos,adj_mat,num_steps):
    steps = np.arange(0,num_steps,1)
    for i in range(num_steps):
        if (i == 0):
            steps[i] = start_pos
        else:
            steps[i] = random_walk(steps[i-1],adj_mat)
    return steps;

test_walk = create_walk(1,blues_matrix,12)


def random_rhythm(num_frames,density=.5):
    my_rands = np.random.uniform(0,1,num_frames)
    odds = 1-(np.arange(num_frames)%2)
    every_other_odd = (np.arange(2,num_frames+2,1)%4)*odds/2
    every_fourth_odd = step(np.arange(7,num_frames+7,1)%8-7)
    my_mask = np.full(num_frames,1)+odds/8 + every_other_odd/4 + every_fourth_odd/2
    return step(my_rands-density)

def createADSR(A,D,S,R):
    decay_rate = (1-S)/D;
    release_rate = S/R;
    return (lambda t,L: step(t)*t/A*(1-step(t-A))  
            +step(t-A)*(-decay_rate*t+1+A*decay_rate)*(1-step(t-A-D))
            +step(t-A-D)*S*(1-step(t-(L-R)))
            +step(t-(L-R))*(((L-R)-t)*release_rate+S)*(1-step(t-L)));


def create_note_env(start,length,ADSR=createADSR(1/64,1/32,.75,1/32)):
    return (lambda t: ADSR(t-start,length));
                        
test_env = create_note_env(0,1)
plotfunc(test_env)

rhythm_pattern = random_rhythm(16)

def generate_envolope_from_rhythm(pattern):
    env = zero_function
    for i in range(len(pattern)):
        if (pattern[i] == 1):
            env = add_funcs(env,create_note_env(i,1))
    return env

test_env = generate_envolope_from_rhythm(rhythm_pattern)

plotfunc(test_env,0,16)
    
class RandomBlues(DrumAndBass):
    chord_progression = np.array(0)
    rhythm_pattern = np.array(0)
    rhythm_env = zero_function
    def __init__(self,adjacency_matrix):
        weights = np.array([major_weight,minor_weight])
        weights = weights / weights.sum()
        self.chord_progression = create_walk(np.random.choice([1,6],p=weights),blues_matrix,6)
        self.rhythm_pattern = random_rhythm(16)
        self.rhythm_env = generate_envolope_from_rhythm(self.rhythm_pattern)
    def chord_by_measure(self,i):
        return standard_chords[self.chord_progression[int(np.floor(i/2))]-1] 
            
    def pianoEnv(self,t):
       adj_t = (t-self.start_time)%(self.seconds_per_measure()*2)
       return self.rhythm_env(adj_t*8*self.seconds_per_beat());

    def createPiano(self,t,i):
        cur_chord = self.chord_by_measure(i)
        return self.pianoEnv(t)*cur_chord(current_key*(t-self.get_measure_start_time(i)));

    def get_func_by_measure(self,i):
        dnb = super().get_func_by_measure(i)
        return (lambda t: (.5*self.createPiano(t,i)+2*dnb(t))/3);

       
        
random_phrase = RandomBlues(get_weighted_blues_matrix())
print(random_phrase.chord_progression)
print(random_phrase.rhythm_pattern)


test_sample = random_phrase.render_phrase()
sd.play(test_sample)
sd.stop()



random_phrase1 = RandomBlues(get_weighted_blues_matrix())
random_phrase1.render_phrase()
random_phrase2 = RandomBlues(get_weighted_blues_matrix())
random_phrase2.render_phrase()


def createBasicLoop(t):
    return (.5*createPiano(t)+.2*createSnare(t)+3*createBass(t))/4;



def createGuitarSound(t):
    return .20*signal.square(220*tau*t);






def createGuitar(t):
    guitar_env = createADSR(.05,.025,.5,.2)
    return createGuitarSound(t)*guitar_env(t,1);




def create_note(timbre,envolope,start,length,pitch):
    return (lambda t: timbre((t-start)*pitch)*envolope(t-start,length))

def append_note(note,length,start_time,note_sequence=zero_function):    
    return add_funcs(note_sequence,new_note(note,length))



melody_time = 0



def new_note(note,length):
    global melody_time
    note_start = melody_time
    melody_time = note_start+length
    return create_note(createGuitar,createADSR(.05,.025,.5,.2),note_start,length,equal_temperament[note])
    


note_sequence = zero_function




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
    

def blues_scale():
    global note_sequence;
    global melody_time;
    melody_time = 0;
    note_sequence = zero_function;
    melody = [('C1',1), ('D#1',1), ('F1',1), ('F#1',1), ('G1',1),
               ('A#2',.5), ('C2',1.5)]
    for n in melody:
        append_note(n[0],n[1])
    plotnplay(note_sequence,melody_time)  

blues_scale()



phrase_queue = queue.Queue()
phrase_queue.put(random_phrase1)
phrase_queue.put(random_phrase1)
phrase_queue.put(random_phrase2)
phrase_queue.put(random_phrase1)



def fill_buffer(frames, time):
    if (fill_buffer.current_phrase is None):
        if (phrase_queue.empty()):
            return np.full(frames,0);
        else:
            fill_buffer.current_phrase = phrase_queue.get();
            fill_buffer.current_phrase.start_time = time;
            print("Starting new phrase")
    timedata = np.arange(time,time+frames/sr,1/sr)
    current_measure = fill_buffer.current_phrase.get_measure_by_time(timedata[0])
    end_measure = fill_buffer.current_phrase.get_measure_by_time(timedata[-1])
    if(end_measure < current_measure and not phrase_queue.empty()):
        print("Starting new measure and new phrase")
        transition_time = fill_buffer.current_phrase.get_measure_start_time(end_measure)
        left_half = fill_buffer.current_phrase.get_prerender_by_time(timedata < transition_time);
        fill_buffer.current_phrase = phrase_queue.get();
        fill_buffer.current_phrase.start_time = transition_time;
        right_half = fill_buffer.current_phrase.get_prerender_by_time(timedata >= transition_time);
        return np.append(left_half,right_half);
    else:
        return get_prerender_by_time(timedata);
    
    

fill_buffer.current_phrase = None




def callback(indata, outdata, frames, time, status):
    if status:
        print(status) 
    outdata[:,0] = fill_buffer(frames, time.outputBufferDacTime)
    return;
    
def finished_callback():
    print("we're done, now what?")


strm = sd.Stream(channels=1, callback=callback, samplerate=sr, blocksize=blocksize, finished_callback=finished_callback)
strm.start()




phrase_queue.put(piano_phrase)

strm.stop()



