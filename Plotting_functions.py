import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
import re
import numpy as np


def Plot_AvgOrientations(Mean_SEM_dict,session_name):

  numeric_keys = []

  for key in Mean_SEM_dict.keys():
      if key.isnumeric():
          numeric_keys.append(int(key))

  numeric_keys = sorted(numeric_keys)
  numeric_keys = [str(num) for num in numeric_keys]


  # Crea un nuovo plot
  fig, ax = plt.subplots()

  # Traccia le linee e le bande di errore per ogni chiave del dizionario
  colors = cm.jet(np.linspace(0, 1, len(numeric_keys)))
  labels = []
  patches = []

  for i, key in enumerate(numeric_keys):
        mean_stim = Mean_SEM_dict[key][:,0]
        sem_stim = Mean_SEM_dict[key][:,1]

        intT_key = 'gray '+key

        mean_intT = Mean_SEM_dict[intT_key][:,0]
        sem_intT = Mean_SEM_dict[intT_key][:,1]

        mean = np.concatenate((mean_stim, mean_intT))
        sem = np.concatenate((sem_stim, sem_intT))

        # Traccia la linea
        line, = plt.plot(mean, color=colors[i])

        # Aggiungi la banda di errore shaded
        upper_bound = mean + sem
        lower_bound = mean - sem
        plt.fill_between(range(len(mean)), lower_bound, upper_bound, color=colors[i], alpha=0.2)
        
        patch = Patch(facecolor=colors[i])
        # Aggiungi l'etichetta alla legenda
        labels.append(key)
        patches.append(patch)

  # Imposta il colore di sfondo per x > len(mean_stim)
  plt.axvspan(len(mean_stim), len(mean), facecolor='gray', alpha=0.2)

  # Aggiungi la legenda fuori dal plot
  plt.legend(patches, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

  # Aggiungi le etichette degli assi e un titolo
  plt.xlabel('Frames')
  plt.ylabel('Z scored DF/F')

  plt.savefig(session_name+'_avgOrientations.png')
  # Mostra il grafico
  plt.show()

def Plot_AvgFlash(Mean_SEM_dict,session_name):
  mean_gf = Mean_SEM_dict['gray flash'][:,0]
  sem_gf = Mean_SEM_dict['gray flash'][:,1]
  mean_b = Mean_SEM_dict['black'][:,0]
  sem_b = Mean_SEM_dict['black'][:,1]
  mean_flash = np.concatenate((mean_gf, mean_b))
  sem_flash = np.concatenate((sem_gf, sem_b))


  plt.figure()
  # Calcola la banda di errore
  upper_bound = mean_flash + sem_flash
  lower_bound = mean_flash - sem_flash
  ochre_yellow = (204/255, 119/255, 34/255)

  # Disegna il grafico della media con SEM shaded
  plt.plot(mean_flash, color=ochre_yellow)
  plt.fill_between(range(len(mean_flash)), lower_bound, upper_bound, color=ochre_yellow , alpha=0.2)
  # Imposta il colore di sfondo per x > len(mean_stim)
  plt.axvspan(len(mean_gf), len(mean_flash), facecolor='black', alpha=0.2)

  # Aggiungi le etichette degli assi
  plt.xlabel('Frames')
  plt.ylabel('Z scored DF/F')
  plt.savefig(session_name+'_avgFlash.png')

  # Mostra il grafico
  plt.show()


def Plot_AvgSBA(Mean_SEM_dict,session_name):
  SBAs = ['initial gray', 'after flash gray', 'final gray']

  fig, ax = plt.subplots()

  # Traccia le linee e le bande di errore per ogni chiave del dizionario
  colors = cm.jet(np.linspace(0, 1, len(SBAs)))
  labels = []
  patches = []

  for i, key in enumerate(SBAs):
          mean= Mean_SEM_dict[key][:,0]
          sem = Mean_SEM_dict[key][:,1]

          # Traccia la linea
          line, = plt.plot(mean, color=colors[i])

          # Aggiungi la banda di errore shaded
          upper_bound = mean + sem
          lower_bound = mean - sem
          plt.fill_between(range(len(mean)), lower_bound, upper_bound, color=colors[i], alpha=0.2)
          
          patch = Patch(facecolor=colors[i])
          # Aggiungi l'etichetta alla legenda
          labels.append(key)
          patches.append(patch)


  # Aggiungi la legenda fuori dal plot
  plt.legend(patches, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

  # Aggiungi le etichette degli assi e un titolo
  plt.xlabel('Frames')
  plt.ylabel('Z scored DF/F')

  plt.savefig(session_name+'_avgSBAs.png')
  # Mostra il grafico
  plt.show()


def plot_cell_tuning(cell_OSI_dict, cell_id, Cell_Max_dict):
  
  Tuning_curve_avgSem = cell_OSI_dict['cell_'+str(cell_id)]
  x = np.arange(Tuning_curve_avgSem.shape[1])

  # Plot the mean values as a line
  plt.plot(x, Tuning_curve_avgSem[0], color='blue', label='mean')

  # Plot the standard error as a shaded region
  plt.fill_between(x, Tuning_curve_avgSem[0] - Tuning_curve_avgSem[1], Tuning_curve_avgSem[0] + Tuning_curve_avgSem[1], color='lightblue', alpha=0.5, label='standard error')

  # Add a legend and axis labels
  plt.legend()
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.ylim([-0.1,5])
  xticks = list(Cell_Max_dict.keys())
  plt.xticks(range(len(xticks)), xticks)


  # Show the plot
  plt.show()
  