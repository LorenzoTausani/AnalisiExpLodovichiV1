import numpy as np
import os
import re
from scipy.stats import mode

def get_orientation_keys(Mean_SEM_dict):
  numeric_keys_int = []
  for key in Mean_SEM_dict.keys():
      if key.isnumeric():
          numeric_keys_int.append(int(key))

  numeric_keys_int = sorted(numeric_keys_int)
  numeric_keys = [str(num) for num in numeric_keys_int]

  return numeric_keys, numeric_keys_int

def SEMf(Fluorescence_matrix):
   Std = np.std(Fluorescence_matrix, axis=0)
   nr_neurons = Fluorescence_matrix.shape[0]
   SEM = Std/np.sqrt(nr_neurons)
   return SEM

def Create_logical_dict(session_name,stimoli,df):
    SBAs = ['initial gray', 'initial black', 'after flash gray', 'final gray']
    logical_dict_filename = session_name+'_logical_dict.npz'
    if not(os.path.isfile(logical_dict_filename)):
        logical_dict ={} #contiene gli indici dei vari stimoli
        for stim in df['Orientamenti'].unique():
            if stim != 'END':
                if stim in SBAs:
                    logical_dict[stim]= stimoli==stim
                else: #stimoli ripetuti (orientamenti, flash)
                    vettore =stimoli==stim # Definisci il vettore booleano di True e False
                    # Converti il vettore booleano in una stringa di 0 e 1
                    stringa = ''.join('1' if x else '0' for x in vettore)
                    # Trova tutte le sequenze di 1 consecutive nella stringa e calcola la loro lunghezza
                    indici_inizio_gruppi = [match.start() for match in re.finditer('1+', stringa)]
                    indici_fine_gruppi = [match.end() for match in re.finditer('1+', stringa)]
                    indici_array = np.column_stack((indici_inizio_gruppi, indici_fine_gruppi))
                    logical_dict[str(stim)] = indici_array


        np.savez(logical_dict_filename, **logical_dict)
    else:
        logical_dict = np.load(logical_dict_filename)

    return logical_dict

def Create_Mean_SEM_dict(session_name,logical_dict, Fluorescence, Fluorescence_type = 'F'):
    #Fluorescence_type can be set to F, Fneu, F_neuSubtract, DF_F, DF_F_zscored
    SBAs = ['initial gray', 'initial black', 'after flash gray', 'final gray']
    Mean_SEM_dict_filename = session_name+Fluorescence_type+'_Mean_SEM_dict.npz'
    if not(os.path.isfile(Mean_SEM_dict_filename)):
        Mean_SEM_dict = {}
        for key in logical_dict.keys():
            if key in SBAs:
                Mean = np.mean(Fluorescence[:,logical_dict[key]], axis=0)
                SEM = SEMf(Fluorescence[:,logical_dict[key]])
                Mean_SEM = np.column_stack((Mean, SEM))
                Mean_SEM_dict[key] = Mean_SEM
            else:
                M_inizio_fine = logical_dict[key]
                stim_lens = M_inizio_fine[:, 1] - M_inizio_fine[:, 0]
                durata_corretta_stim = int(mode(stim_lens)[0])
                Betw_cells_mean = np.full((M_inizio_fine.shape[0],durata_corretta_stim), np.nan)
                for i, row in enumerate(M_inizio_fine):
                    if np.abs(stim_lens[i]-durata_corretta_stim)< durata_corretta_stim/10:
                        Betw_cells_mean[i,:] = np.mean(Fluorescence[:,row[0]:row[0]+durata_corretta_stim], axis=0)
                        #Nota: è impossibile che due sequenze dello stesso tipo siano adiacenti
                Mean = np.mean(Betw_cells_mean, axis=0)
                SEM = SEMf(Betw_cells_mean)
                Mean_SEM = np.column_stack((Mean, SEM))
                Mean_SEM_dict[key] = Mean_SEM

        np.savez(Mean_SEM_dict_filename, **Mean_SEM_dict)
    else:
        Mean_SEM_dict = np.load(Mean_SEM_dict_filename)
    return Mean_SEM_dict


def Cell_max(logical_dict, Fluorescence, session_name, durata_corretta_stim ='mode', Fluorescence_type='F'):
  #Fluorescence_type can be set to F, Fneu, F_neuSubtract, DF_F, DF_F_zscored
  Cell_max_dict_filename = session_name+Fluorescence_type+'_Cell_max_dict_'+durata_corretta_stim+'.npz'
  #durata_corretta_stim può anche essere settato come intero, che indichi il numero di frame da considerare
  if not(os.path.isfile(Cell_max_dict_filename)):
    if not(isinstance(durata_corretta_stim, str)):
        durata_corretta_stim = str(durata_corretta_stim)
    Cell_Max_dict = {}
    numeric_keys, numeric_keys_int = get_orientation_keys(logical_dict)

    for i, key in enumerate(numeric_keys): #per ogni orientamento...
        M_inizio_fine = logical_dict[key] #seleziona l'array con i tempi di inizio e quelli di fine
        stim_lens = M_inizio_fine[:, 1] - M_inizio_fine[:, 0] #calcola la durata di ciascun periodo di stimolazione con l'orientamento di interesse
        if durata_corretta_stim =='mode':
            durata_corretta_stim = int(mode(stim_lens)[0]) #la durata corretta dello stimolo è assunto essere la moda delle durate
        else:
            durata_corretta_stim = int(durata_corretta_stim)

        # nr. cellule x nr stimolazioni con un certo orientamento
        Cells_maxs = np.full((Fluorescence.shape[0],M_inizio_fine.shape[0]), np.nan)
        for cell in range(Fluorescence.shape[0]): #per ogni cellula...
            cell_trace = Fluorescence[cell,:] #estraggo l'intera traccia di fluorescenza di quella cellula
            for i, row in enumerate(M_inizio_fine): #per ogni stimolazione con un certo orientamento
                if np.abs(stim_lens[i]-durata_corretta_stim)< durata_corretta_stim/20:#se lo stimolo ha la giusta durata
                    Avg_PreStim = np.mean(cell_trace[row[0]-(durata_corretta_stim):row[0]]) #medio i valori di fluorescenza nei durata_corretta_stim frame prima dello stimolo (gray)
                    Avg_stim = np.mean(cell_trace[row[0]:row[0]+durata_corretta_stim]) #medio i valori di fluorescenza nei durata_corretta_stim frame dello stimolo (gray)
                    Cells_maxs[cell,i] = (Avg_stim-Avg_PreStim)/Avg_PreStim #i.e.  (F - F0) / F0
        Cell_Max_dict[key] = Cells_maxs
  else:
    Cell_Max_dict = np.load(Cell_max_dict_filename)
  return Cell_Max_dict