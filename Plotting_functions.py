import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
import re
import numpy as np
import copy
from matplotlib.gridspec import GridSpec
import Utils
from Utils import *


def Plot_AvgOrientations(Mean_SEM_dict, Fluorescence_type = 'F',ax=[], ori = '', only_or = False):
  
  keys_of_interest = []
  if only_or == False:
    for key in Mean_SEM_dict.keys():
        if key.isnumeric():
            keys_of_interest.append(int(key))

    keys_of_interest = sorted(keys_of_interest)
    keys_of_interest = [str(num) for num in keys_of_interest]
    if not(ori==''):
      keys_of_interest = [key + ori for key in keys_of_interest]
      print(keys_of_interest)
  else:
    keys_of_interest = ['+','-']

  if ax ==[]:
    # Crea un nuovo plot
    fig, ax = plt.subplots()

  # Traccia le linee e le bande di errore per ogni chiave del dizionario
  colors = cm.jet(np.linspace(0, 1, len(keys_of_interest)))
  labels = []
  patches = []

  for i, key in enumerate(keys_of_interest):
        mean_stim = Mean_SEM_dict[key][:,0]
        sem_stim = Mean_SEM_dict[key][:,1]

        intT_key = 'gray '+key

        mean_intT = Mean_SEM_dict[intT_key][:,0]
        sem_intT = Mean_SEM_dict[intT_key][:,1]

        mean = np.concatenate((mean_stim, mean_intT))
        sem = np.concatenate((sem_stim, sem_intT))

        # Traccia la linea
        line, = ax.plot(mean, color=colors[i])

        # Aggiungi la banda di errore shaded
        upper_bound = mean + sem
        lower_bound = mean - sem
        ax.fill_between(range(len(mean)), lower_bound, upper_bound, color=colors[i], alpha=0.2)
        
        patch = Patch(facecolor=colors[i])
        # Aggiungi l'etichetta alla legenda
        labels.append(key)
        patches.append(patch)

  # Imposta il colore di sfondo per x > len(mean_stim)
  ax.axvspan(len(mean_stim), len(mean), facecolor='gray', alpha=0.2)

  # Aggiungi la legenda fuori dal plot
  ax.legend(patches, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

  # Aggiungi le etichette degli assi e un titolo
  ax.set_xlabel('Frames')
  ax.set_ylabel(Fluorescence_type)

  #plt.savefig(session_name+Fluorescence_type+'_avgOrientations.png')
  # Mostra il grafico
  #plt.show()

def Plot_AvgFlash(Mean_SEM_dict, Fluorescence_type = 'F',ax=[]):
  mean_gf = Mean_SEM_dict['gray flash'][:,0]
  sem_gf = Mean_SEM_dict['gray flash'][:,1]
  mean_b = Mean_SEM_dict['black'][:,0]
  sem_b = Mean_SEM_dict['black'][:,1]
  mean_flash = np.concatenate((mean_gf, mean_b))
  sem_flash = np.concatenate((sem_gf, sem_b))
  if ax==[]:
    fig, ax = plt.subplots()  # Calcola la banda di errore

  # Calcola la banda di errore
  upper_bound = mean_flash + sem_flash
  lower_bound = mean_flash - sem_flash
  ochre_yellow = (204/255, 119/255, 34/255)

  # Disegna il grafico della media con SEM shaded
  ax.plot(mean_flash, color=ochre_yellow)
  ax.fill_between(range(len(mean_flash)), lower_bound, upper_bound, color=ochre_yellow , alpha=0.2)
  # Imposta il colore di sfondo per x > len(mean_stim)
  ax.axvspan(len(mean_gf), len(mean_flash), facecolor='black', alpha=0.2)

  # Aggiungi le etichette degli assi
  ax.set_xlabel('Frames')
  ax.set_ylabel(Fluorescence_type)
  #plt.savefig(session_name+Fluorescence_type+'_avgFlash.png')

  # Mostra il grafico
  #plt.show()

def Plot_AvgSBA(Mean_SEM_dict,Fluorescence_type = 'F',ax=[]):
  SBAs = ['initial gray', 'after flash gray', 'final gray']
  if ax==[]:
    fig, ax = plt.subplots()

  # Traccia le linee e le bande di errore per ogni chiave del dizionario
  colors = cm.jet(np.linspace(0, 1, len(SBAs)))
  labels = []
  patches = []

  for i, key in enumerate(SBAs):
          if key in Mean_SEM_dict:
            mean= Mean_SEM_dict[key][:,0]
            sem = Mean_SEM_dict[key][:,1]

            # Traccia la linea
            line, = ax.plot(mean, color=colors[i])

            # Aggiungi la banda di errore shaded
            upper_bound = mean + sem
            lower_bound = mean - sem
            ax.fill_between(range(len(mean)), lower_bound, upper_bound, color=colors[i], alpha=0.2)
            
            patch = Patch(facecolor=colors[i])
            # Aggiungi l'etichetta alla legenda
            labels.append(key)
            patches.append(patch)


  # Aggiungi la legenda fuori dal plot
  ax.legend(patches, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

  # Aggiungi le etichette degli assi e un titolo
  ax.set_xlabel('Frames')
  ax.set_ylabel(Fluorescence_type)

  #plt.savefig(session_name+Fluorescence_type+'_avgSBAs.png')
  # Mostra il grafico
  #plt.show()


def plot_cell_tuning(cell_OSI_dict, cell_id, Cell_Max_dict, y_range=[], ax=[]):
  
  Tuning_curve_avgSem = cell_OSI_dict['cell_'+str(cell_id)]
  x = np.arange(Tuning_curve_avgSem.shape[1])
  if ax ==[]:
    fig, ax = plt.subplots()
  # Plot the mean values as a line
  ax.plot(x, Tuning_curve_avgSem[0], color='purple', label='mean')

  # Plot the standard error as a shaded region
  ax.fill_between(x, Tuning_curve_avgSem[0] - Tuning_curve_avgSem[1], Tuning_curve_avgSem[0] + Tuning_curve_avgSem[1], color='purple', alpha=0.2, label='standard error')

  # Add a legend and axis labels
  ax.set_xlabel('Orientation')
  ax.set_ylabel('(Fstim-Fpre)/Fpre')
  if len(cell_OSI_dict['OSI'].shape)>1:
    OSI_value = cell_OSI_dict['OSI'][cell_id,0]
  else:
    OSI_value = cell_OSI_dict['OSI'][cell_id]
  ax.set_title('cell_'+str(cell_id)+', OSI: '+str(np.round(OSI_value,decimals=2)))
  if y_range!=[]:
    ax.set_ylim(y_range)
  numeric_keys, numeric_keys_int = get_orientation_keys(Cell_Max_dict)
  xticks = list(numeric_keys)
  ax.set_xticks(range(len(xticks)), xticks)

def cumulativePlot_OSI(OSI_v, ax=[]):
  sorted_OSI_v = np.sort(OSI_v)

  # calculate the cumulative distribution function (CDF)
  cumulative_prob = np.cumsum(np.ones_like(sorted_OSI_v)) / len(sorted_OSI_v)

  if ax == []:
    fig, ax = plt.subplots()

  ax.plot(sorted_OSI_v, cumulative_prob)
  ax.set_xlabel('OSI')
  ax.set_ylabel('Cumulative probability')
  ax.set_xlim([0,1])
  perc_above05 = (np.sum(sorted_OSI_v>=0.4)/len(sorted_OSI_v))*100
  ax.set_title('OSI>0.4: '+str(np.sum(sorted_OSI_v>=0.4))+'/'+str(len(sorted_OSI_v))+' cells ('+str(np.round(perc_above05,decimals=2))+' %)')


  # Draw a vertical line at x=2
  ax.axvline(x=0.4, color='red')

def Orientation_freq_plot(OSI_v, cell_OSI_dict, ax=[]):
  OSI_idx05 = OSI_v>0.4
  PrefOr = cell_OSI_dict['PrefOr']
  if len(cell_OSI_dict['PrefOr'].shape)>1:
    PrefOr05 = PrefOr[OSI_idx05,0]
  else:
    PrefOr05 = PrefOr[OSI_idx05]
  # Get the counts of unique lists
  Considers_list = False
  if len(PrefOr05.shape)>1: #per i primi exps dove c era anche 360 gradi
    for idx,p_or in enumerate(PrefOr05):
      if idx ==0:
        l = len(p_or)
      elif not(l==len(p_or)):
        Considers_list = True
        break

  if Considers_list == False:
    if len(PrefOr05.shape)>1:
      unique_lists, counts = np.unique(PrefOr05, return_counts=True, axis=0)
    else:
     unique_lists, counts = np.unique(PrefOr05, return_counts=True)
    color_dict = {'0, 180': 'blue','45, 225': 'orange','90, 270': 'green', '135, 315': 'red'}
  else:
    unique_lists, counts = np.unique(PrefOr05, return_counts=True)
    color_dict = {'0, 180, 360': 'blue','45, 225': 'orange','90, 270': 'green', '135, 315': 'red'}

  if len(unique_lists)>1:
    unique_strings = [', '.join(map(str, lst)) for lst in unique_lists]
  else:
    unique_strings=[]

  if ax==[]:
    fig, ax = plt.subplots()
  colors = [color_dict.get(lst, 'gray') for lst in unique_strings]
  ax.pie(counts, labels=unique_strings, autopct='%1.1f%%', colors=colors)

  # Add a title
  ax.set_title('OSI>0.4 cells distribution')

def summaryPlot_OSI(cell_OSI_dict,Cell_Max_dict,session_name, stat =[], Fluorescence_type='F'):
  if len(cell_OSI_dict['OSI'].shape)>1:
    OSI_v = copy.deepcopy(cell_OSI_dict['OSI'])[:,0]
  else:
    OSI_v = copy.deepcopy(cell_OSI_dict['OSI'])

  OSI_v[OSI_v>1]=np.nan
  OSI_v[OSI_v<0]=np.nan
  sorted_idxs = np.argsort(OSI_v)
  Best = np.nanargmax(OSI_v)
  sorted_idxs = sorted_idxs[:np.argmax(sorted_idxs==Best)+1]

  fig = plt.figure(layout="constrained")
  fig = plt.figure(figsize=(50, 20))
  gs = GridSpec(2, 2, figure=fig)
  ax1 = fig.add_subplot(gs[0, :-1])
  cumulativePlot_OSI(OSI_v, ax=ax1)
  # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
  ax2 = fig.add_subplot(gs[0, -1])
  Orientation_freq_plot(OSI_v, cell_OSI_dict, ax=ax2)

  # Create a nested grid for the second subplot
  gs1 = gs[1, 0].subgridspec(2, 1, hspace=0.75)
  ax3 = fig.add_subplot(gs1[0])
  plot_cell_tuning(cell_OSI_dict, sorted_idxs[-1], Cell_Max_dict, y_range=[], ax=ax3)
    
  ax4 = fig.add_subplot(gs1[1])
  plot_cell_tuning(cell_OSI_dict, sorted_idxs[-2], Cell_Max_dict, y_range=[], ax=ax4)
  
  if not(stat==[]):
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_aspect('equal', adjustable='box')
    highOSI_cell_map(stat,OSI_v,cell_OSI_dict,ax=ax5)
  else:
    gs2 = gs[1, 1].subgridspec(2, 1, hspace=0.75)
    ax5 = fig.add_subplot(gs2[0])
    plot_cell_tuning(cell_OSI_dict, sorted_idxs[-3], Cell_Max_dict, y_range=[], ax=ax5)

    ax6 = fig.add_subplot(gs2[1])
    plot_cell_tuning(cell_OSI_dict, sorted_idxs[-4], Cell_Max_dict, y_range=[], ax=ax6)

  plt.subplots_adjust(hspace=0.3,wspace=0.2)
  plt.savefig(session_name+'_'+Fluorescence_type+'_OSI.png')
  plt.show()

def summaryPlot_AvgActivity(Mean_SEM_dict,session_name, Fluorescence_type = 'DF_F_zscored'):

  fig = plt.figure(figsize=(50, 20))
  gs = GridSpec(2, 3, figure=fig)
  ax1 = fig.add_subplot(gs[0, :-1])
  Plot_AvgOrientations(Mean_SEM_dict,Fluorescence_type = Fluorescence_type,ax=ax1)
  # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
  if 'gray flash' in Mean_SEM_dict:
    ax2 = fig.add_subplot(gs[0, -1])
    Plot_AvgFlash(Mean_SEM_dict,Fluorescence_type = Fluorescence_type,ax=ax2)
  ax3 = fig.add_subplot(gs[1, :])
  Plot_AvgSBA(Mean_SEM_dict,Fluorescence_type = Fluorescence_type,ax=ax3)  
  plt.subplots_adjust(hspace=0.3,wspace=0.75)
  plt.savefig(session_name+'_'+Fluorescence_type+'_avgActivity.png')
  plt.show()

def highOSI_cell_map(stat, OSI_v, cell_OSI_dict, ax=[]):
    OSI_idx05 = OSI_v > 0.4
    if len(cell_OSI_dict['PrefOr'].shape)>1:
      PrefOr = cell_OSI_dict['PrefOr'][:, 0]
    else:
      PrefOr = cell_OSI_dict['PrefOr']

    Considers_list = False
    if len(PrefOr.shape)>1: #per i primi exps dove c era anche 360 gradi
      for idx,p_or in enumerate(PrefOr):
        if idx ==0:
          l = len(p_or)
        elif not(l==len(p_or)):
          Considers_list = True
          break
    if Considers_list == False:
      color_dict = {'[0, 180]': 'blue','[45, 225]': 'orange','[90, 270]': 'green', '[135, 315]': 'red'}
    else:
      print('9 orientations present')
      color_dict = {'0, 180, 360': 'blue','45, 225': 'orange','90, 270': 'green', '135, 315': 'red'}
    
    # Create a black 512x512 background
    if ax == []:
        fig, ax = plt.subplots(figsize=[10, 10])
    ax.set_xlim([0, 512])
    ax.set_ylim([0, 512])
    ax.set_facecolor('black')

    # Create two lists to hold the non-grey and grey circles
    non_grey_circles = []
    grey_circles = []

    # Iterate over the circle centers and radii
    for idx, cell in enumerate(stat):
        c = copy.deepcopy(cell['med'])
        c.reverse()
        r = cell['radius']

        if OSI_idx05[idx] == True:
            # Determine the circle color based on the radius
            if Considers_list:
              color = color_dict[str(PrefOr[idx])]
            else:
              color = color_dict[str(list(PrefOr[idx]))]
            # Create the circle patch with the given center and radius
            circle = plt.Circle(c, r, color=color)
            non_grey_circles.append(circle)
        else:
            # Create the circle patch with the given center and radius
            circle = plt.Circle(c, r, color='grey', alpha=0.2)
            grey_circles.append(circle)


    # Add the grey circles to the plot first, and then the coloured ones
    for circle in grey_circles:
        ax.add_artist(circle)

    for circle in non_grey_circles:
        ax.add_artist(circle)

    ax.set_title('OSI>0.4 cells position')

#da sistemare
def plot_stim_cellwise(logical_dict, Fluorescence,Cell_stat_dict, cell_id):
  Fcell = Fluorescence[cell_id,:]
  Pref_ori_cell = Cell_stat_dict['PrefOr'][cell_id][0]

  ori = input('which orientation? (pref or is '+str(Pref_ori_cell)+', OSI is '+str(Cell_stat_dict['OSI'][cell_id,0])+')')
  timestamps_ori = logical_dict[ori]
  for ts in timestamps_ori:
    stim_dur = ts[1]-ts[0]
    plt.axvline(x=stim_dur, color='r', linestyle='--')
    plt.axvline(x=2*stim_dur, color='r', linestyle='--')
    ts[0] = ts[0] - stim_dur
    ts[1] = ts[1] + stim_dur
    plt.plot(Fcell[ts[0]:ts[1]])

