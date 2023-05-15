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
   Std = np.nanstd(Fluorescence_matrix, axis=0)
   nr_neurons = Fluorescence_matrix.shape[0] #andrebbe cambiato togliendo i nan
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


def Create_Cell_max_dict(logical_dict, Fluorescence, session_name, averaging_window ='mode', Fluorescence_type='F'):
  #Fluorescence_type can be set to F, Fneu, F_neuSubtract, DF_F, DF_F_zscored
  
  #averaging_window può anche essere settato come intero, che indichi il numero di frame da considerare
  
  if not(isinstance(averaging_window, str)):
      averaging_window = str(averaging_window)
  Cell_max_dict_filename = session_name+Fluorescence_type+'_Cell_max_dict_'+averaging_window+'.npz'
  if not(os.path.isfile(Cell_max_dict_filename)):
    Cell_Max_dict = {}
    numeric_keys, numeric_keys_int = get_orientation_keys(logical_dict)

    for i, key in enumerate(numeric_keys): #per ogni orientamento...
        M_inizio_fine = logical_dict[key] #seleziona l'array con i tempi di inizio e quelli di fine
        stim_lens = M_inizio_fine[:, 1] - M_inizio_fine[:, 0] #calcola la durata di ciascun periodo di stimolazione con l'orientamento di interesse
        durata_corretta_stim = int(mode(stim_lens)[0])
        if averaging_window =='mode':
            averaging_window = durata_corretta_stim #la durata corretta dello stimolo è assunto essere la moda delle durate
        else:
            averaging_window = int(averaging_window)

        # nr. cellule x nr stimolazioni con un certo orientamento
        Cells_maxs = np.full((Fluorescence.shape[0],M_inizio_fine.shape[0]), np.nan)
        for cell in range(Fluorescence.shape[0]): #per ogni cellula...
            cell_trace = Fluorescence[cell,:] #estraggo l'intera traccia di fluorescenza di quella cellula
            for i, row in enumerate(M_inizio_fine): #per ogni stimolazione con un certo orientamento
                if np.abs(stim_lens[i]-durata_corretta_stim)< durata_corretta_stim/20:#se lo stimolo ha la giusta durata
                    Avg_PreStim = np.mean(cell_trace[(row[0]-averaging_window):row[0]]) #medio i valori di fluorescenza nei averaging_window frame prima dello stimolo (gray)
                    Avg_stim = np.mean(cell_trace[row[0]:(row[0]+averaging_window)]) #medio i valori di fluorescenza nei averaging_window frame dello stimolo (gray)
                    Cells_maxs[cell,i] = (Avg_stim-Avg_PreStim)/Avg_PreStim #i.e.  (F - F0) / F0
        Cell_Max_dict[key] = Cells_maxs
    np.savez(Cell_max_dict_filename, **Cell_Max_dict)
  else:
    Cell_Max_dict = np.load(Cell_max_dict_filename)
  return Cell_Max_dict


def OSIf(Tuning_curve_avgSem, numeric_keys_int, idxs_4orth_ori = [0,1,2,3,4,5,6,7,8,1,2,3,4,5,6],plus180or = True):
  
  idx_max = np.nanargmax(Tuning_curve_avgSem[0,:]) #idx with maximum average activity
  preferred_or = numeric_keys_int[idx_max]
  #Next is if you want to consider R_pref and R_ortho also considering the +180 orientations
  if plus180or:
    R_pref  = np.mean([Tuning_curve_avgSem[0,idx_max],Tuning_curve_avgSem[0,idxs_4orth_ori[idx_max+4]]])
    R_ortho = np.mean([Tuning_curve_avgSem[0,idxs_4orth_ori[idx_max+2]],Tuning_curve_avgSem[0,idxs_4orth_ori[idx_max+6]]])
  else:
    R_pref  = Tuning_curve_avgSem[0,idx_max]
    R_ortho = Tuning_curve_avgSem[0,idxs_4orth_ori[idx_max+2]]

  OSI = (R_pref -R_ortho)/(R_pref + R_ortho)
  return OSI, preferred_or

def OSIf_alternative(Tuning_curve_avgSem, numeric_keys_int):  #preferisci questa a OSIf
  degrees_combinations=[[0,4,8],[1,5],[2,6],[3,7]] #i.e. [[0,180,360],[45,225],[90,270],[135,315]]
  orthogonal_combinations = [[2,6],[3,7],[0,4,8],[1,5]]
  if np.sum(Tuning_curve_avgSem[0,:]<0)>0: #se c'è almeno un valore sotto lo zero...
    Tuning_curve_avgSem[0,:] = Tuning_curve_avgSem[0,:] + np.abs(np.min(Tuning_curve_avgSem[0,:]))

  Or_plus_180avg = np.full((len(degrees_combinations)),np.nan)
  
  for idx,tupl in enumerate(degrees_combinations):
    el_tobe_avg = np.full((len(tupl)),np.nan)
    for i,el in enumerate(tupl):
      el_tobe_avg[i]=Tuning_curve_avgSem[0,el]
    Or_plus_180avg[idx]=np.nanmean(el_tobe_avg)

  pref_or_idx = np.nanargmax(Or_plus_180avg)
  preferred_or = [numeric_keys_int[i] for i in degrees_combinations[pref_or_idx]]
  ortho_or_idx = degrees_combinations.index(orthogonal_combinations[pref_or_idx])

  R_pref = Or_plus_180avg[pref_or_idx]
  R_ortho = Or_plus_180avg[ortho_or_idx]
  OSI = (R_pref -R_ortho)/(R_pref + R_ortho)

  Or2 = [sublst for sublst in degrees_combinations if sublst != degrees_combinations[idx_max] and sublst != orthogonal_combinations[idx_max]]
  if Or_plus_180avg[degrees_combinations.index(Or2[0])]>Or_plus_180avg[degrees_combinations.index(Or2[1])]:
    R_pref2 = Or_plus_180avg[degrees_combinations.index(Or2[0])]
    R_ortho2 = Or_plus_180avg[degrees_combinations.index(Or2[1])]
    preferred_or2 = [numeric_keys_int[i] for i in degrees_combinations[degrees_combinations.index(Or2[0])]]
  else:
    R_pref2 = Or_plus_180avg[degrees_combinations.index(Or2[1])]
    R_ortho2 = Or_plus_180avg[degrees_combinations.index(Or2[0])]
    preferred_or2 = [numeric_keys_int[i] for i in degrees_combinations[degrees_combinations.index(Or2[1])]]

  OSI2 = (R_pref2 -R_ortho2)/(R_pref2 + R_ortho2)

  Best_OSI = np.max([OSI,OSI2])
  OSI_arr=np.array((OSI,OSI2,Best_OSI))
  preferred_or_list = [preferred_or,preferred_or2]
  return OSI_arr,preferred_or_list


def Create_OSI_dict(Cell_Max_dict,F_neuSubtract,OSI_alternative=True):
  nr_cells = F_neuSubtract.shape[0]
  if OSI_alternative:
    OSI_v = np.full((nr_cells,3), np.nan)
    PrefOr_v = []   
  else:
    OSI_v = np.full((nr_cells), np.nan)
    PrefOr_v = np.full((nr_cells), np.nan)
  numeric_keys, numeric_keys_int = get_orientation_keys(Cell_Max_dict)
  idxs_4orth_ori = [0,1,2,3,4,5,6,7,8,1,2,3,4,5,6]

  cell_OSI_dict ={}
  for cell_id in range(nr_cells):
    Tuning_curve_avgSem = np.full((2,len(Cell_Max_dict.keys())), np.nan) #len(Cell_Max_dict.keys() = nr of orientations

    for i,key in enumerate(Cell_Max_dict.keys()):
      Tuning_curve_avgSem[0,i] = np.nanmean(Cell_Max_dict[key][cell_id])
      Tuning_curve_avgSem[1,i] = SEMf(Cell_Max_dict[key][cell_id])
    cell_OSI_dict['cell_'+str(cell_id)] = Tuning_curve_avgSem
    if OSI_alternative:
       OSI_arr,preferred_or_list = OSIf_alternative(Tuning_curve_avgSem, numeric_keys_int)
       OSI_v[cell_id,:] = OSI_arr
       PrefOr_v.append(preferred_or_list)
    else:
      OSI, preferred_or = OSIf(Tuning_curve_avgSem, numeric_keys_int, idxs_4orth_ori = idxs_4orth_ori,plus180or = True)
      OSI_v[cell_id] = OSI
      PrefOr_v[cell_id] = preferred_or
  cell_OSI_dict['OSI'] = OSI_v
  cell_OSI_dict['PrefOr'] = PrefOr_v
  return cell_OSI_dict

