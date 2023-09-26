# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:15:01 2023

@author: lORENZO TAUSANI
"""
#librerie utilizzate
from psychopy import visual, core, event, monitors
import random
import numpy as np
import pandas as pd
import time, os
import datetime

# Triggering = 1 -> il programma attende TTL dal microscopio
triggering = int(input('Triggering? (1=on, 0=off)'))

SBJ_ID = input('subjectID?')
nr_exp = input('experiment number of today?')   
data_di_oggi = datetime.date.today()
data_di_oggi = data_di_oggi.strftime("%Y_%m_%d")
nr_trials_or = int(input('how many drifting gratings trials?'))
nr_trials_flash = int(input('how many flash trials?'))

or_list=[] #stores the stimuli used in the experiment
timestamps_list = [] #stores the timings of the stimuli
frame_rate = 60
IFI = 1/frame_rate # time between each frame flip
trial_time = 5 #time (in secs) of a single trial. STANDARD = 5
Spontaneous_time = 120# time (in secs) of the spontaneous activity recording (gray screen). STANDARD = 120
intertrial_time_or = 10 # time (in secs) between two consecutive orientation trials. STANDARD = 5. In this time gray screen is presented
time_flash = 0.25
time_black = 2
acquisition_speed_Hz = 30
frame_to_acquire = ((Spontaneous_time*3)+(time_black*(nr_trials_flash+1))+(time_flash*nr_trials_flash)+(nr_trials_or*trial_time)+(intertrial_time_or*(nr_trials_or-1)))*acquisition_speed_Hz
print('acquisisci almeno ' + str(frame_to_acquire)+' frames')


my_monitor = monitors.Monitor(name='TOPO_TAUSANI') #name of the monitor used
my_monitor.setDistance(10) #distance of the observer from the screen in cm. It is used to compute the degree amplitude 'deg'


win0 = visual.Window([1280,800], monitor=my_monitor,screen=1,units='deg',fullscr=True) 

#gray screen parameters
gray_screen = visual.GratingStim(win=win0, pos=(0,0), color=(0,0,0), size=1000, contrast=0, units='deg')
black_screen = visual.GratingStim(win=win0, mask='None', size=1000, pos=(0, 0), sf=0, color=[-1, -1, -1])


print('waiting for trigger')
if triggering ==1:
    import u3
    # Create a connection to the LabJack device
    labjack = u3.U3()
    labjack.configU3()
    labjack.configIO(EnableCounter0=1,TimerCounterPinOffset = 6)
    start = labjack.getFeedback(u3.Counter0())[0]
    nFrames_list=[]
    
    # Wait for a trigger signal from the LabJack
    while labjack.getFeedback(u3.Counter0())[0]==start:
        time.sleep(0.001) #aspetta tra un check e l'altro di labjack
else:
    start=0

# Presentazione dello stimolo di contrasto medio per 5 secondi
or_list.append('initial gray')
gray_screen.draw()
win0.flip()
start_time = time.time()
def save_timestamps(start, start_time, triggering=triggering):
    timestamps_list.append(time.time()-start_time)
    if triggering ==1:
        nFrames_list.append(labjack.getFeedback(u3.Counter0())[0]-start)
# timestamps_list.append(start_time)
# if triggering ==1:
#     nFrames_list.append(labjack.getFeedback(u3.Counter0())[0]-start)
save_timestamps(start, start_time)
print('gray')
core.wait(Spontaneous_time)

if nr_trials_flash>0:
    #presentazione primo stimolo nero
    or_list.append('initial black')
    black_screen.draw()
    win0.flip()
    save_timestamps(start, start_time)
    core.wait(time_black)
    
    #flash di luce
    for i_flash in range(nr_trials_flash):
        or_list.append('gray flash')
        gray_screen.draw()
        win0.flip()
        save_timestamps(start, start_time)
        print('flash '+str(i_flash))
        core.wait(time_flash)
        
        
        or_list.append('black')
        black_screen.draw()
        win0.flip()
        save_timestamps(start, start_time)
        core.wait(time_black)
        
        
     
    or_list.append('after flash gray')
    gray_screen.draw()
    win0.flip()
    print('gray')
    save_timestamps(start, start_time)
    core.wait(Spontaneous_time)


grating_ex = visual.GratingStim(win=win0, pos=(0,0), size=2000, sf=0.04, ori=90, phase=(0,0), tex='sqr', units='deg')
'''
PARAMETRI DEL GRATING:
    TEX: 'sqr' = onda quadra -> bordi netti, 'sin' = onda seno, più smussata
    SF: frequenza spaziale -> alta=barre più strette
    ORI: angolatura delle barre
'''

#set the number of equally spaced orientations (STANDARD = 8)
orientations = np.linspace(0, 315, 8)

#Create a pseudorandom sequence of stimuli (next 20 lines)
stim_len = len(orientations)
arr = np.arange(stim_len)
v_stims = np.zeros(nr_trials_or)

for i in range(nr_trials_or//stim_len):
    if i==0:
        np.random.shuffle(arr)
        v_stims[i*stim_len:i*stim_len+stim_len]=arr
        #print(arr)
        #print(v_stims)
    else:
        np.random.shuffle(arr)
        #print(arr)
        v_stims[i*stim_len:i*stim_len+stim_len]=arr
        #print(v_stims)
if nr_trials_or//stim_len>0:
    trials_remaining = nr_trials_or-stim_len*(i+1)
    last_els = np.random.choice(arr, size=trials_remaining, replace=False)
    print(last_els)
    v_stims[i*stim_len+stim_len:i*stim_len+stim_len+trials_remaining]=last_els

#direction of movement will be randomized
movment_dir = ['+','-']

for i_or in range(nr_trials_or):
    if i_or>0:    
        gray_screen.draw()
        or_list.append('gray')
        win0.flip()

        save_timestamps(start, start_time)
        print('gray intertrial '+str(i_or))
        core.wait(intertrial_time_or)

    selected_or = orientations[int(v_stims[i_or])]
    selected_movDir = movment_dir[random.randint(0, 1)]
    or_list.append(str(selected_or)+selected_movDir)
    print('orientation selected: '+str(selected_or)+selected_movDir)
    save_timestamps(start, start_time)
    for j in range(round(trial_time*frame_rate)):
        grating_ex.ori=selected_or
        #velocità di movimento (minore = + veloce) e direzione (+/-). 2*IFI -> temporal frequency of 2Hz (visto con marco 18 04 23)
        grating_ex.setPhase(2*IFI,selected_movDir) 
        grating_ex.contrast=-1 #float between -1 and 1
        grating_ex.draw()
        win0.flip()
        
or_list.append('final gray')
gray_screen.draw()
win0.flip()
save_timestamps(start, start_time)
print('gray')
core.wait(Spontaneous_time)

win0.close()
save_timestamps(start, start_time)
or_list.append('END')


# Creazione di un DataFrame
if triggering ==1:
    df = pd.DataFrame({'Orientamenti': or_list,'Computer_time': timestamps_list,'N_frames': nFrames_list})
else:
    df = pd.DataFrame({'Orientamenti': or_list, 'Computer_time': timestamps_list})
# Esportazione del DataFrame in un file Excel

filename = 'Sbj_'+ SBJ_ID+'_'+ data_di_oggi +'_NExp_'+nr_exp+'_NTr_'+str(nr_trials_or)+'.xlsx'
df.to_excel(os.path.join(r'C:\Users\Matteo\Documents\Dati\Tausani', filename), index=False)
    
    