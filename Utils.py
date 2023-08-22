import numpy as np
import os
import re
import pandas as pd
import glob
from scipy.stats import mode
from scipy.stats import zscore
import shutil
import matplotlib.pyplot as plt
import Plotting_functions
from Plotting_functions import *

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

def single_session_analysis(Session_folder='manual_selection', session_name='none',Force_reanalysis = False):
  getoutput=False
  if Session_folder=='manual_selection':
    getoutput=True
    from google.colab import drive
    drive.mount('/content/drive')
    #ricerda del folder della sessione di interesse
    Main_folder = '/content/drive/MyDrive/esperimenti2p_Tausani/'
    dir_list = os.listdir(Main_folder)
    sbj_list = '\n'.join([f'{i}: {sbj}' for i, sbj in enumerate(dir_list)])
    idx_sbj = int(input('Which subject?\n'+sbj_list))
    sbj_folder = os.path.join(Main_folder,dir_list[idx_sbj])
    dir_list = os.listdir(sbj_folder)
    sbj_list = '\n'.join([f'{i}: {sbj}' for i, sbj in enumerate(dir_list)])
    idx_session = int(input('Which session?\n'+sbj_list))
    session_name = dir_list[idx_session]
    Session_folder = os.path.join(sbj_folder,session_name) # Session folder è il path alla sessione di interesse
    os.chdir(Session_folder)

    #se vuoi rianalizzare cancella tutto
    if Force_reanalysis:
      if os.path.isdir(os.path.join(Session_folder, 'Analyzed_data')):
        shutil.rmtree(os.path.join(Session_folder,'Analyzed_data/'))
      if os.path.isdir(os.path.join(Session_folder, 'Plots')):
        shutil.rmtree(os.path.join(Session_folder,'Plots/'))

  df, StimVec = Df_loader_and_StimVec(Session_folder, not_consider_direction = False)
  F = np.load('F.npy')
  Fneu = np.load('Fneu.npy')
  iscell = np.load('iscell.npy') #iscell[:,0]==1 sono cellule
  if getoutput==True: #da rimuovere
    stat = np.load('stat.npy', allow_pickle=True)
    stat = stat[iscell[:,0]==1]


  cut = len(StimVec)
  if getoutput:
    plt.plot(np.mean(F,axis = 0))
    # Show the plot
    plt.show()
    plt.pause(0.1)
    cut = int(input('at which frame you want to cut the series (all = ' +str(len(StimVec))+ ')?'))
    StimVec = StimVec[:cut]
    df = df[df['N_frames']<cut]
  F = F[iscell[:,0]==1,:cut]
  Fneu = Fneu[iscell[:,0]==1,:cut]
  F_neuSubtract = F - 0.7*Fneu
  F_neuSubtract[F_neuSubtract<0]=0
  #normalizzare?

  os.makedirs(os.path.join(Session_folder,'Analyzed_data/'), exist_ok=True); os.chdir(os.path.join(Session_folder,'Analyzed_data/'))
  logical_dict = Create_logical_dict(session_name,StimVec,df)
  # F0 = np.mean(F_neuSubtract[:,logical_dict['final gray']], axis = 1)[:, np.newaxis]
  # DF_F = (F_neuSubtract - F0)/ F0
  # DF_F_zscored = zscore(DF_F, axis=1)  

  Mean_SEM_dict_F_neuSubtract = Create_Mean_SEM_dict(session_name,logical_dict, F_neuSubtract, Fluorescence_type = 'F_neuSubtract')
  Cell_Max_dict_F_neuSubtract_mode = Create_Cell_max_dict(logical_dict, F_neuSubtract, session_name, averaging_window ='mode', Fluorescence_type='F_neuSubtract')
  cell_OSI_dict = Create_OSI_dict(Cell_Max_dict_F_neuSubtract_mode,session_name)
  Cell_stat_dict = Create_Cell_stat_dict(logical_dict, F_neuSubtract, session_name, averaging_window ='mode', Fluorescence_type='F_neuSubtract', OSI_alternative=True)

  os.makedirs(os.path.join(Session_folder,'Plots/'), exist_ok=True); os.chdir(os.path.join(Session_folder,'Plots/'))
  Plotting_functions.summaryPlot_AvgActivity(Mean_SEM_dict_F_neuSubtract,session_name, Fluorescence_type = 'F_neuSubtract')
  if getoutput==True: #da rimouovere
    Plotting_functions.summaryPlot_OSI(cell_OSI_dict,Cell_Max_dict_F_neuSubtract_mode,session_name,stat=stat,Fluorescence_type='F_neuSubtract')
  else:
    Plotting_functions.summaryPlot_OSI(cell_OSI_dict,Cell_Max_dict_F_neuSubtract_mode,session_name,stat=[],Fluorescence_type='F_neuSubtract')
  if getoutput:
    return locals()

def Analyze_all(Force_reanalysis = True):
  from google.colab import drive
  drive.mount('/content/drive')

  Main_folder = '/content/drive/MyDrive/esperimenti2p_Tausani/'
  os.chdir(Main_folder)
  dir_list = os.listdir(Main_folder)

  for c_dir in dir_list:
    Subj_folder = Main_folder+c_dir+'/'
    os.chdir(Subj_folder)
    dir_list = os.listdir(Subj_folder)
    for session_name in dir_list:
      Session_folder = Subj_folder+session_name+'/'
      os.chdir(Session_folder)
      #la sessione non ha precedenti analisi, e contiene i file per l'analisi necessari
      Analyzed_files_notPresent = ('Analyzed_data' not in os.listdir())
      Necessary_files_present = any(file.endswith('.npy') for file in os.listdir()) and any(file.endswith('.xlsx') for file in os.listdir())
      if not(Necessary_files_present):
        print("\033[1mRequired files not found - session "+ session_name+"\033[0m")
      else:
        if (Analyzed_files_notPresent or Force_reanalysis):
          print("\033[1mAnalyzing session "+ session_name+"\033[0m")
          if Force_reanalysis and not(Analyzed_files_notPresent):
            shutil.rmtree(Session_folder+'Analyzed_data/')
            shutil.rmtree(Session_folder+'Plots/')

          single_session_analysis(Session_folder=Session_folder, session_name=session_name)

def old_version_df(df):
    def contains_numeric(string):
        pattern = r'\d+'
        if re.search(pattern, string):
            return True
        else:
            return False

    def exclude_chars(string):
        pattern = r'\.0[+-]'
        return re.sub(pattern, '', string)

    for idx,lbl in enumerate(df['Orientamenti']):
        if contains_numeric(lbl):
            df['Orientamenti'][idx]=exclude_chars(lbl)
    
    return df

def Df_loader_and_StimVec(Session_folder, not_consider_direction = True):
  # use the glob module to find the Excel file with the specified extension
  excel_files = glob.glob(os.path.join(Session_folder, "*.xlsx"))
  #print(excel_files[0])

  # Carica il file Excel in un DataFrame
  df = pd.read_excel(excel_files[0])
  #chiamo ogni gray in funzione dell'orientamento precedente
  def contains_numeric_characters(s):
    return any(char.isdigit() for char in s)

  for it, row in df.iterrows():
    if contains_numeric_characters(str(row['Orientamenti'])):
      if str(row['Orientamenti'])[-1]=='+' or str(row['Orientamenti'])[-1]=='-': 
        df['Orientamenti'][it] = str(int(float(row['Orientamenti'][:-1])))+row['Orientamenti'][-1]
      else:
        df['Orientamenti'][it] = str(row['Orientamenti'])
    elif row['Orientamenti']=='gray':
      orientamento = df['Orientamenti'][it-1]
      df['Orientamenti'][it] = 'gray '+str(orientamento)
  
  if not_consider_direction:
     for stim in df['Orientamenti']:
        if '+' in stim:
           df = old_version_df(df)
           print('direction is not considered in the analysis')
           break

  # Crea un array vuoto per il vettore di stimoli
  StimVec = np.empty(df['N_frames'].max(), dtype=object)
  # Itera sul DataFrame e assegna il tipo di stimolo a ogni unità di tempo
  top=0
  for it, row in df.iterrows():
      if it==0:
        prec_row = row
      else:
        StimVec[top:row['N_frames']] = prec_row['Orientamenti']
        top=row['N_frames']
        prec_row = row
        
  return df, StimVec


def Create_logical_dict(session_name,stimoli,df):
    def contains_plus_character(vector): #function to check if any element of df['Orientamenti'].unique() contains a '+' sign
      for string in vector:
          if '+' in string:
              return True
      return False
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

        if contains_plus_character(df['Orientamenti'].unique()):
          #ora creo le voci integrate
          ors  = df['Orientamenti'].unique()
          pattern = r'\d+\.\d+|\d+'  # espressione regolare per cercare tutti i numeri

          new_keys = []
          for elem in ors:
              matches = re.findall(pattern, elem)
              try:
                matches = matches[0]
              except:
                print('')
              if not(matches == []):
                new_keys.append(matches)

          new_keys = list(set(new_keys))+['+','-']
          new_keys =new_keys + ['gray '+ n for n in new_keys]

          for new_key in new_keys:
            if "+" not in new_key and "-" not in new_key: #i.e. if new_key è un numero
              key_plus = new_key+'+'
              key_minus = new_key+'-'

              alt_keys = [key_plus, key_minus]
            elif 'gray' in new_key:
              plus_minus = new_key[-1]
              alt_keys = [key for key in logical_dict.keys() if 'gray' in key and plus_minus in key]
            else:
              plus_minus = new_key[-1]
              alt_keys = [key for key in logical_dict.keys() if not('gray' in key) and plus_minus in key]
            
            # Concatenate the arrays vertically (axis=0) to form a single array
            arrays_list = []
            for k in alt_keys:
              try:
                arrays_list.append(logical_dict[k])
              except:
                print(k+' is missing')
            concatenated_array = np.concatenate(arrays_list, axis=0)

            # Sort the rows in ascending order based on the first column (index 0)
            sorted_array = concatenated_array[np.argsort(concatenated_array[:, 0])]
            logical_dict[new_key] = sorted_array
            
        np.savez(logical_dict_filename, **logical_dict)
    else:
        logical_dict = np.load(logical_dict_filename)

    return logical_dict

def Create_Mean_SEM_dict(session_name,logical_dict, Fluorescence,  Fluorescence_type = 'F'):
    #Fluorescence_type can be set to F, Fneu, F_neuSubtract, DF_F, DF_F_zscored
    SBAs = ['initial gray', 'initial black', 'after flash gray', 'final gray']
    Mean_SEM_dict_filename = session_name+Fluorescence_type+'_Mean_SEM_dict.npz'
    if not(os.path.isfile(Mean_SEM_dict_filename)):
        Mean_SEM_dict = {}
        for key in logical_dict.keys():
            if key in SBAs:
                if Fluorescence.shape[1]>= np.where(logical_dict[key])[0][-1]: #se la traccia è stata tutta registrata
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
                    giusta_durata = np.abs(stim_lens[i]-durata_corretta_stim)< durata_corretta_stim/10
                    fluo_registrata = Fluorescence.shape[1]>=M_inizio_fine[i, 1]
                    if giusta_durata and fluo_registrata:
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
  Cell_max_dict_filename = session_name+'_'+Fluorescence_type+'_Cell_max_dict_'+averaging_window+'.npz'
  if not(os.path.isfile(Cell_max_dict_filename)):
    Cell_Max_dict = {}
    Cell_Max_dict['Fluorescence_type']=Fluorescence_type
    Cell_Max_dict['averaging_window']=averaging_window
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
        Avg_stim_V = np.full((Fluorescence.shape[0],M_inizio_fine.shape[0]), np.nan)
        Avg_PreStim_V = np.full((Fluorescence.shape[0],M_inizio_fine.shape[0]), np.nan)
        for cell in range(Fluorescence.shape[0]): #per ogni cellula...
            cell_trace = Fluorescence[cell,:] #estraggo l'intera traccia di fluorescenza di quella cellula
            for i, row in enumerate(M_inizio_fine): #per ogni stimolazione con un certo orientamento
                giusta_durata = np.abs(stim_lens[i]-durata_corretta_stim)< durata_corretta_stim/10
                fluo_registrata = Fluorescence.shape[1]>=M_inizio_fine[i, 1]                
                if giusta_durata and fluo_registrata:#se lo stimolo ha la giusta durata
                    Avg_PreStim = np.mean(cell_trace[(row[0]-averaging_window):row[0]]) #medio i valori di fluorescenza nei averaging_window frame prima dello stimolo (gray)
                    Avg_stim = np.mean(cell_trace[row[0]:(row[0]+averaging_window)]) #medio i valori di fluorescenza nei averaging_window frame dello stimolo (gray)
                    #Cells_maxs[cell,i] = (Avg_PreStim-Avg_stim)/Avg_stim #i.e.  (F - F0) / F0
                    Avg_stim_V[cell,i] = Avg_stim 
                    Avg_PreStim_V[cell,i] = Avg_PreStim 
                    #Cells_maxs[cell,i] = Avg_stim
                    # Min = np.min(cell_trace[(row[0]-averaging_window):row[0]])
                    # Max = np.max(cell_trace[(row[0]-averaging_window):row[0]])
                    # Cells_maxs[cell,i] = (Max-Min)/(Min+Max)
                    Cells_maxs[cell,i] = trace_good(cell_trace[row[0]:(row[0]+averaging_window)])
        Cell_Max_dict[key] = Cells_maxs
        Cell_Max_dict[key+'_PreStim'] = Avg_PreStim_V
        Cell_Max_dict[key+'_Stim'] = Avg_stim_V

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
  if len(numeric_keys_int)>8:
    degrees_combinations=[[0,4,8],[1,5],[2,6],[3,7]] #i.e. [[0,180,360],[45,225],[90,270],[135,315]]
    orthogonal_combinations = [[2,6],[3,7],[0,4,8],[1,5]]
  else:
    degrees_combinations=[[0,4],[1,5],[2,6],[3,7]] #i.e. [[0,180],[45,225],[90,270],[135,315]]
    orthogonal_combinations = [[2,6],[3,7],[0,4],[1,5]]
     
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

  Or2 = [sublst for sublst in degrees_combinations if sublst != degrees_combinations[pref_or_idx] and sublst != orthogonal_combinations[pref_or_idx]]
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


def Create_OSI_dict(Cell_Max_dict,session_name, OSI_alternative=True):
  OSI_dict_filename = session_name+'_'+str(Cell_Max_dict['Fluorescence_type'])+'_OSI_dict_'+str(Cell_Max_dict['averaging_window'])+'.npz'
  if not(os.path.isfile(OSI_dict_filename)):
    nr_cells = Cell_Max_dict['0'].shape[0]
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
      Tuning_curve_avgSem = np.full((2,len(numeric_keys)), np.nan) #len(Cell_Max_dict.keys() = nr of orientations

      for i,key in enumerate(numeric_keys):
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
    cell_OSI_dict['PrefOr'] = np.array(PrefOr_v)
  else:
    Cell_Max_dict = np.load(OSI_dict_filename)
  return cell_OSI_dict


def trace_good(Fluorescence):
    if len(Fluorescence.shape)>1:
      quartile_25 = np.percentile(Fluorescence, 25,axis=1)
      quartile_99 = np.percentile(Fluorescence, 99,axis=1)
    else:
      quartile_25 = np.percentile(Fluorescence, 25)
      quartile_99 = np.percentile(Fluorescence, 99)
       
    STDs_Q1 = []
    for i,q25 in enumerate(quartile_25):
      traccia = Fluorescence[i,:]
      dati_primo_quartile =  traccia[(traccia <= q25)]
      STDs_Q1.append(np.std(dati_primo_quartile))
    STDs_Q1 = np.array(STDs_Q1)
    metrica = quartile_99/STDs_Q1
    metrica[metrica==np.Inf] =0
    return metrica

def Create_Cell_stat_dict(logical_dict, Fluorescence, session_name, averaging_window ='mode', Fluorescence_type='F', OSI_alternative=True):
  #Fluorescence_type can be set to F, Fneu, F_neuSubtract, DF_F, DF_F_zscored
  #averaging_window può anche essere settato come intero, che indichi il numero di frame da considerare

  if not(isinstance(averaging_window, str)):
      averaging_window = str(averaging_window)
  nr_cells = Fluorescence.shape[0]
  if OSI_alternative:
    OSI_writing = 'OSI_alt'
    OSI_v = np.full((nr_cells,3), np.nan)
    PrefOr_v = []
  else:
    Pl180yn = int(input('do you want to consider also the parallel orientation? (1=y,0=n)'))
    OSI_writing = 'OSI_classic' + ('+180' if Pl180yn == 1 else '')
    OSI_v = np.full((nr_cells), np.nan)
    PrefOr_v = np.full((nr_cells), np.nan)
  Cell_stat_dict_filename = session_name+'_'+Fluorescence_type+'_Cell_stat_dict_'+averaging_window+OSI_writing+'.npz'
  if not(os.path.isfile(Cell_stat_dict_filename)):
    Cell_stat_dict = {}
    Cell_stat_dict['Fluorescence_type']=Fluorescence_type
    Cell_stat_dict['averaging_window']=averaging_window
    Cell_stat_dict['Trace_goodness_metric'] = trace_good(Fluorescence)
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
        Cells_maxs = np.full((nr_cells,M_inizio_fine.shape[0]), np.nan)
        for cell in range(nr_cells): #per ogni cellula...
            cell_trace = Fluorescence[cell,:] #estraggo l'intera traccia di fluorescenza di quella cellula
            for i, row in enumerate(M_inizio_fine): #per ogni stimolazione con un certo orientamento
                giusta_durata = np.abs(stim_lens[i]-durata_corretta_stim)< durata_corretta_stim/10
                fluo_registrata = Fluorescence.shape[1]>=M_inizio_fine[i, 1]                
                if giusta_durata and fluo_registrata:#se lo stimolo ha la giusta durata
                    Avg_PreStim = np.mean(cell_trace[(row[0]-averaging_window):row[0]]) #medio i valori di fluorescenza nei averaging_window frame prima dello stimolo (gray)
                    Avg_stim = np.mean(cell_trace[row[0]:(row[0]+averaging_window)]) #medio i valori di fluorescenza nei averaging_window frame dello stimolo (gray)
                    Cells_maxs[cell,i] = (Avg_stim-Avg_PreStim)/Avg_PreStim #i.e.  (F - F0) / F0
        Cell_stat_dict[key] = Cells_maxs


    idxs_4orth_ori = [0,1,2,3,4,5,6,7,8,1,2,3,4,5,6] #si potrebbe forse calcolare in un modo più intelligente
    Cell_ori_tuning_curve_mean = np.full((nr_cells,len(numeric_keys)), np.nan) #len(Cell_stat_dict.keys() = nr of orientations
    Cell_ori_tuning_curve_sem = np.full((nr_cells,len(numeric_keys)), np.nan)

    for cell_id in range(nr_cells):
      Tuning_curve_avgSem = np.full((2,len(numeric_keys)), np.nan) #len(Cell_Max_dict.keys() = nr of orientations
      for i,key in enumerate(numeric_keys):
        Cell_ori_tuning_curve_mean[cell_id,i] = np.nanmean(Cell_stat_dict[key][cell_id])
        Cell_ori_tuning_curve_sem[cell_id,i] = SEMf(Cell_stat_dict[key][cell_id])
      Tuning_curve_avgSem[0,:]=Cell_ori_tuning_curve_mean[cell_id,:]
      Tuning_curve_avgSem[1,:]=Cell_ori_tuning_curve_sem[cell_id,:]
      if OSI_alternative:
        OSI_arr,preferred_or_list = OSIf_alternative(Tuning_curve_avgSem, numeric_keys_int)
        OSI_v[cell_id,:] = OSI_arr
        PrefOr_v.append(preferred_or_list)
      else:
        OSI, preferred_or = OSIf(Tuning_curve_avgSem, numeric_keys_int, idxs_4orth_ori = idxs_4orth_ori,plus180or = Pl180yn)
        OSI_v[cell_id] = OSI
        PrefOr_v[cell_id] = preferred_or
    Cell_stat_dict['Cell_ori_tuning_curve_mean'] = Cell_ori_tuning_curve_mean
    Cell_stat_dict['Cell_ori_tuning_curve_sem'] = Cell_ori_tuning_curve_sem
    Cell_stat_dict['OSI'] = OSI_v
    Cell_stat_dict['PrefOr'] = np.array(PrefOr_v)
    np.savez(Cell_stat_dict_filename, **Cell_stat_dict)
  else:
    Cell_stat_dict = np.load(Cell_stat_dict_filename)
  return Cell_stat_dict