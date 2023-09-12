import numpy as np
import os
import re
import pandas as pd
import glob
from scipy.stats import mode
from scipy.stats import zscore
from scipy import stats
import shutil
import matplotlib.pyplot as plt
import Plotting_functions
from Plotting_functions import *
import ast

def get_orientation_keys(Mean_SEM_dict):
  numeric_keys_int = []
  for key in Mean_SEM_dict.keys():
      if key.isnumeric():
          numeric_keys_int.append(int(key))

  numeric_keys_int = sorted(numeric_keys_int)
  numeric_keys = [str(num) for num in numeric_keys_int]

  return numeric_keys, numeric_keys_int

def dF_F_Yuste_method(Fluorescence,timepoint):
  Frames_10s = 10 * 30
  traccia = Fluorescence[:,timepoint-Frames_10s:timepoint]
  median_Fluorescence = np.percentile(traccia, 50,axis=1)
  Avg_first50 = []
  for i,q25 in enumerate(median_Fluorescence):
    traccia = Fluorescence[i,timepoint-Frames_10s:timepoint]
    dati_prima_meta =  traccia[(traccia <= q25)]
    Avg_first50.append(np.mean(dati_prima_meta))
  Avg_first50 = np.array(Avg_first50)
  dF_F_Yuste_timepoint = (Fluorescence[:,timepoint]-Avg_first50)/Avg_first50
  return dF_F_Yuste_timepoint


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
  F_to_use = F
  if getoutput:
    Yuste_yn = int(input('Do you want to compute Yuste \' smoothing and use it for the calculations? 1=yes/0=no'))
    if Yuste_yn == 1:
      dF_F_Yuste = np.zeros((F.shape[0],F.shape[1]-300))
      for i in range(F.shape[1]):
        if i>=300:
          c=i-300
          dF_F_Yuste[:,c] = dF_F_Yuste_method(F,i)
      dF_F_Yuste =np.concatenate((np.zeros((dF_F_Yuste.shape[0],300)), dF_F_Yuste), axis=1)
      F_to_use = dF_F_Yuste

  Mean_SEM_dict_F = Create_Mean_SEM_dict(session_name,logical_dict, F_to_use, Fluorescence_type = 'F')
  Cell_Max_dict_F = Create_Cell_max_dict(logical_dict, F_to_use, session_name, averaging_window ='mode', Fluorescence_type='F')
  cell_OSI_dict = Create_OSI_dict(Cell_Max_dict_F,session_name)
  Cell_stat_dict = Create_Cell_stat_dict(logical_dict, F_to_use, session_name, averaging_window ='mode', Fluorescence_type='F', OSI_alternative=False)
  
 
  os.makedirs(os.path.join(Session_folder,'Plots/'), exist_ok=True); os.chdir(os.path.join(Session_folder,'Plots/'))
  p_value,perc_diff_wGray2 = Comparison_gray_stim(F_to_use, logical_dict,session_name)
  Plotting_functions.summaryPlot_AvgActivity(Mean_SEM_dict_F,session_name, Fluorescence_type = 'F')
  # if getoutput==True: #da rimouovere
  #   Plotting_functions.summaryPlot_OSI(cell_OSI_dict,Cell_Max_dict_F,session_name,stat=stat,Fluorescence_type='F')
  # else:
  #   Plotting_functions.summaryPlot_OSI(cell_OSI_dict,Cell_Max_dict_F,session_name,stat=[],Fluorescence_type='F')
  #if getoutput:
  return locals()

def Analyze_all(Force_reanalysis = True, select_subjects = True):
  from google.colab import drive
  drive.mount('/content/drive')
  correlation_dict = {}
  correlation_stats_tensor = None
  Main_folder = '/content/drive/MyDrive/esperimenti2p_Tausani/'
  os.chdir(Main_folder)
  dir_list = os.listdir(Main_folder)
  if select_subjects:
    sbj_list = '\n'.join([f'{i}: {sbj}' for i, sbj in enumerate(dir_list)])
    idx_sbj = ast.literal_eval(input('Which subjects (write in [])?\n'+sbj_list))
  else:
     idx_sbj=range(len(dir_list))
  comp_list = []
  for nr,c_dir in enumerate(dir_list):
    if nr in idx_sbj:
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
              if os.path.isdir(os.path.join(Session_folder, 'Analyzed_data')):
                shutil.rmtree(os.path.join(Session_folder,'Analyzed_data/'))
              if os.path.isdir(os.path.join(Session_folder, 'Plots')):
                shutil.rmtree(os.path.join(Session_folder,'Plots/'))

            return_dict = single_session_analysis(Session_folder=Session_folder, session_name=session_name)
            comp_item = np.zeros(2)
            comp_item[0] = return_dict['p_value']
            comp_item[1] = return_dict['perc_diff_wGray2']
            comp_list.append([session_name,comp_item])

            #vado a raccogliere le statistiche di correlazione che mi interessano
            correlation_dict[session_name] =compute_correlation(return_dict['F_neuSubtract'], return_dict['logical_dict'])
            correlation_stats = np.zeros((correlation_dict[session_name].shape[0],2))
            for matrix_idx in range(correlation_dict[session_name].shape[0]):
               correlation_tensor = correlation_dict[session_name][matrix_idx,:,:] #prendo ciascuna delle matrici di correlazione
               correlation_vec_no_symmetry = correlation_tensor[np.triu_indices(6, k=1)] #numpy.triu_indices(n, k=0, m=None) Return the indices for the upper-triangle of an (n, m) array.
               correlation_stats[matrix_idx,0] = np.mean(correlation_vec_no_symmetry)
               correlation_stats[matrix_idx,1] = np.std(correlation_vec_no_symmetry)
            if correlation_stats_tensor is None:
                correlation_stats_tensor = np.expand_dims(correlation_stats, axis=0)
            else:
                correlation_stats = np.expand_dims(correlation_stats, axis=0)
                correlation_stats_tensor = np.concatenate((correlation_stats_tensor, correlation_stats), axis=0)
  sesson_names = [item[0] for item in comp_list]
  p_values = [item[1][0] for item in comp_list]
  Percent_increase = [item[1][1] for item in comp_list]
  df_stim_vs_gray = pd.DataFrame({'Session name': sesson_names, 'P_val': p_values, '% change wrt grey2': Percent_increase})        

  return df_stim_vs_gray, correlation_stats_tensor

   

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
                    # COMMENTATO CI SONO I METODI ALTERNATIVI PER CALCOLO DELL'OSI PROVATI
                    # traccia = cell_trace[(row[0]-averaging_window):row[0]]
                    # median_Fluorescence = np.percentile(traccia, 50)
                    # dati_prima_meta =  traccia[(traccia <= median_Fluorescence)]
                    # Avg_PreStim =(np.mean(dati_prima_meta))
                    # Min = np.min(cell_trace[(row[0]-averaging_window):row[0]])
                    # Max = np.max(cell_trace[(row[0]-averaging_window):row[0]])
                    # Cells_maxs[cell,i] = (Max-Min)/Min
                    #Cells_maxs[cell,i] = trace_good(cell_trace[row[0]:(row[0]+averaging_window)])
                    Avg_stim = np.mean(cell_trace[row[0]:(row[0]+averaging_window)]) #medio i valori di fluorescenza nei averaging_window frame dello stimolo (gray)
                    #Cells_maxs[cell,i] = (Avg_stim-Avg_PreStim)/Avg_PreStim #i.e.  (F - F0) / F0
                    Avg_stim_V[cell,i] = Avg_stim
                    Avg_PreStim_V[cell,i] = Avg_PreStim
                    Cells_maxs[cell,i] = Avg_stim

        Cell_Max_dict[key] = Cells_maxs
        Cell_Max_dict[key+'_PreStim'] = Avg_PreStim_V
        Cell_Max_dict[key+'_Stim'] = Avg_stim_V

  else:
    Cell_Max_dict = np.load(Cell_max_dict_filename)
  return Cell_Max_dict


def OSIf(Tuning_curve_avgSem, numeric_keys_int, idxs_4orth_ori = [0,1,2,3,4,5,6,7,1,2,3,4,5,6],plus180or = False):
  
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


def Create_OSI_dict(Cell_Max_dict,session_name, OSI_alternative=False):
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
    idxs_4orth_ori = [0,1,2,3,4,5,6,7,1,2,3,4,5,6] #8 solo per le prime sessioni. Da aggiustare

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
        OSI, preferred_or = OSIf(Tuning_curve_avgSem, numeric_keys_int, idxs_4orth_ori = idxs_4orth_ori,plus180or = False)
        OSI_v[cell_id] = OSI
        PrefOr_v[cell_id] = preferred_or
    cell_OSI_dict['OSI'] = OSI_v
    cell_OSI_dict['PrefOr'] = np.array(PrefOr_v)
  else:
    Cell_Max_dict = np.load(OSI_dict_filename)
  return cell_OSI_dict


def trace_good(Fluorescence): #rimetti a posto come una volta
    if len(Fluorescence.shape)>1:
      quartile_25 = np.percentile(Fluorescence, 25,axis=1)
      quartile_99 = np.percentile(Fluorescence, 99,axis=1)
      STDs_Q1 = []
      for i,q25 in enumerate(quartile_25):
        traccia = Fluorescence[i,:]
        dati_primo_quartile =  traccia[(traccia <= q25)]
        STDs_Q1.append(np.std(dati_primo_quartile))
      STDs_Q1 = np.array(STDs_Q1)
      metrica = quartile_99/STDs_Q1
    else:
      quartile_5 = np.percentile(Fluorescence, 5)
      quartile_95 = np.percentile(Fluorescence, 95)
      metrica = (quartile_95-quartile_5)/quartile_5

    if len(Fluorescence.shape)>1:
      metrica[metrica==np.Inf] =0
    else:
      if metrica==np.Inf:
         metrica=0
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
    #Pl180yn = int(input('do you want to consider also the parallel orientation? (1=y,0=n)'))
    Pl180yn = 0
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
                    #Avg_PreStim = np.mean(cell_trace[(row[0]-averaging_window):row[0]]) #medio i valori di fluorescenza nei averaging_window frame prima dello stimolo (gray)
                    #Avg_stim = np.mean(cell_trace[row[0]:(row[0]+averaging_window)]) #medio i valori di fluorescenza nei averaging_window frame dello stimolo (gray)
                    #Cells_maxs[cell,i] = (Avg_stim-Avg_PreStim)/Avg_PreStim #i.e.  (F - F0) / F0
                    Avg_stim = np.mean(cell_trace[row[0]:(row[0]+averaging_window)])
                    Cells_maxs[cell,i] = Avg_stim
        Cell_stat_dict[key] = Cells_maxs


    idxs_4orth_ori = [0,1,2,3,4,5,6,7,1,2,3,4,5,6] #si potrebbe forse calcolare in un modo più intelligente
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


def Comparison_gray_stim(Fluorescence, logical_dict,session_name):
  

  str_keys, list_keys = get_orientation_keys(logical_dict)
  Activity_arr = np.zeros((Fluorescence.shape[0],100,4))
  Activity_arr[:] = np.nan
  
  for c_ID,cell in enumerate(Fluorescence):
    pointer=0
    for i,k in enumerate(str_keys):
      stim_times = logical_dict[k]
      for stim in stim_times:
        Activity_arr[c_ID,pointer,0] = np.mean(cell[stim[0]:stim[1]])
        pointer+=1
      pointer = pointer - stim_times.shape[0]
      stim_times = logical_dict['gray '+k]
      for c,stim in enumerate(stim_times):
        Activity_arr[c_ID,pointer,1] = np.mean(cell[stim[0]:stim[1]])
        Activity_arr[c_ID,pointer,2] = np.mean(cell[stim[0]:stim[0]+150])
        Activity_arr[c_ID,pointer,3] = np.mean(cell[stim[0]+150:stim[1]])
        pointer+=1

  Activity_arr2 = np.zeros((Fluorescence.shape[0],10))
  Activity_arr2[:] = np.nan
  Activity_arr2[:,0] = np.mean(Fluorescence[:,logical_dict['initial gray']], axis=1)
  Activity_arr2[:,1] = np.mean(Fluorescence[:,logical_dict['final gray']], axis=1)
  Activity_arr2[:,2:6] = np.nanmean(Activity_arr, axis = 1)
  # Step 1: Randomly select 24 unique column indices (5sx24 = 120s)
  selected_columns = np.random.choice(Activity_arr.shape[1]-1, 24, replace=False)
  Activity_arr2[:,6:10] = np.nanmean(Activity_arr[:, selected_columns], axis=1)     
  conditions = ["Sp1", "Sp2", "Stim", "Gray", "Gray1", "Gray2"]
  plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
  plt.boxplot(Activity_arr2[:,:6], labels=conditions)
  # Add labels and title
  plt.xlabel("Conditions")
  plt.ylabel("Fluorescence")
  perc_diff_wGray = np.nanmean(((Activity_arr2[:,2] - Activity_arr2[:,3])/Activity_arr2[:,3])*100)
  perc_diff_wGray2 = np.nanmean(((Activity_arr2[:,2] - Activity_arr2[:,5])/Activity_arr2[:,5])*100)
  _, p_value = stats.wilcoxon(Activity_arr2[:,2] - Activity_arr2[:,3], alternative='greater')
  plt.title("P value "+str("{:.2e}".format(p_value))+', % diff '+str("{:.2}".format(perc_diff_wGray)))
  plt.savefig(session_name+'Fluorescence_periods_comparison.png')
  plt.show()

  #return Activity_arr,Activity_arr2
  return p_value,perc_diff_wGray2

def compute_correlation(Fluorescence, logical_dict):
  def costruisci_timeseries_spezzata(Fluorescence,intervalli_logical_dict):
    numero_colonne_ts_spezzata = intervalli_logical_dict[:, 1] - intervalli_logical_dict[:, 0] + 1

    # Inizializza la nuova matrice con zeri
    ts_spezzata = np.zeros((Fluorescence.shape[0], np.sum(numero_colonne_ts_spezzata)))

    # Copia le colonne appropriate dalla matrice originale nella nuova matrice
    indice_colonna_ts_spezzata = 0
    for row in range(intervalli_logical_dict.shape[0]):
        start, end = intervalli_logical_dict[row]
        ts_spezzata[:, indice_colonna_ts_spezzata:indice_colonna_ts_spezzata + (end - start + 1)] = Fluorescence[:, start:end + 1]
        indice_colonna_ts_spezzata += (end - start + 1)
    return ts_spezzata 


  keys_of_interest = ['initial gray', 'final gray','+','gray +']
  nr_cells = Fluorescence.shape[0]
  correlation_tensor = np.zeros((len(keys_of_interest)+1,nr_cells,nr_cells))
  correlation_tensor[0,:,:] = np.corrcoef(Fluorescence)

  for idx,key in enumerate(keys_of_interest):
    if key=='initial gray' or key=='final gray':
      timeseries_of_interest = Fluorescence[:,logical_dict[key]] 
    else:
      if key=='+':
        other_key = '-'
      else:
        other_key = 'gray -'
      intervalli1 = logical_dict[key]
      intervalli2 = logical_dict[other_key]
      intervalli = np.concatenate((intervalli1, intervalli2), axis=0)
      intervalli = intervalli[intervalli[:,0].argsort()]
      timeseries_of_interest = costruisci_timeseries_spezzata(Fluorescence,intervalli)
    correlation_tensor[idx+1,:,:] = np.corrcoef(timeseries_of_interest)
  return correlation_tensor


def comparison_between_sessions_plots(df):
  df = df.sort_values('Session name')
  nr_of_sessions = df.shape[0]
  tabella_comparazioni = np.empty((nr_of_sessions, 4))
  tabella_comparazioni[:] = np.nan
  #sveglio_yn = int(input('vuoi analizzare animali svegli o anestetizzati? (1=sveglio, 0=anestetizzato)'))
  #psilo_type = int(input('pre vs psilo alta o bassa? (1=alta, 0=bassa)'))
  stat_of_interest = df['% change wrt grey2'].to_numpy()

  row_idx = 0
  Exp_day_list=[]
  for i,session in enumerate(df['Session name']):
    session_info = session.split('_')
    if i == 0:
      day = session_info[0]
      Exp_day_name = session_info[1]+'_'+day
      Exp_day_list.append(Exp_day_name)
    elif not(session_info[0] == day) or not(session_info[1] == df['Session name'].tolist()[i-1].split('_')[1]):
      day = session_info[0]
      Exp_day_name = session_info[1]+'_'+day
      Exp_day_list.append(Exp_day_name)
      row_idx = row_idx+1
    session_info = session.split('_')
    sbj_name = session_info[1]
    session_type = session_info[2]
    if session_info[-1]=='sveglio':
      continue
    col_idx = int(session_type[-1])
    if col_idx>2:
      col_idx = 2
    col_idx =col_idx -1 #indici python
    if 'psilo' in session_type:
      col_idx = col_idx + 2
    tabella_comparazioni[row_idx, col_idx] = stat_of_interest[i]
  last_row_index = np.max(np.where(np.nansum(tabella_comparazioni,axis=1) > 0))

  # Remove rows following the last row containing numbers
  tabella_comparazioni = tabella_comparazioni[:last_row_index +1]
  return tabella_comparazioni