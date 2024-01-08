import numpy as np
import os
import re
import pandas as pd
import glob
from scipy.stats import mode
from scipy.stats import zscore
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
from typing import Union,  Dict, Any, List
import Plotting_functions
from Plotting_functions import *
from Generic_tools.Generic_list_operations import *
from Generic_tools.Generic_foldering_operations import *
from Generic_tools.Generic_numeric_operations import *
from Generic_tools.Generic_string_operations import *
from Generic_tools.Generic_stimulation_handler import *
import ast
import colorsys
from collections import Counter

class CL_stimulation_data(stimulation_data):
   
   def __init__(self, path: str, Stim_var: str = 'Orientamenti', Time_var: str = 'N_frames', not_consider_direction = False):
        super().__init__(path,Stim_var,Time_var)
        self.not_consider_direction = not_consider_direction
   
   def old_version_df(self,df):
        for idx,lbl in enumerate(df[self.Stim_var]):
            if contains_character(lbl,pattern = r'\d+'):
                df[self.Stim_var][idx]=exclude_chars(lbl, pattern=r'\.0[+-]')
        return df

   def Stim_var_rename(self, stimulation_df: pd.DataFrame) -> pd.DataFrame:
      #chiamo ogni gray in funzione dell'orientamento precedente    
      for it, row in stimulation_df.iterrows():
        if contains_character(str(row[self.Stim_var]), pattern = r'\d+'):
          if str(row[self.Stim_var])[-1]=='+' or str(row[self.Stim_var])[-1]=='-': 
            stimulation_df[self.Stim_var][it] = str(int(float(row[self.Stim_var][:-1])))+row[self.Stim_var][-1]
          else:
            stimulation_df[self.Stim_var][it] = str(row[self.Stim_var])
        elif row[self.Stim_var]=='gray':
          orientamento = stimulation_df[self.Stim_var][it-1]
          stimulation_df[self.Stim_var][it] = 'gray '+str(orientamento)
      
      if self.not_consider_direction:
        for stim in stimulation_df[self.Stim_var]:
            if '+' in stim:
              stimulation_df = self.old_version_df(stimulation_df)
              print('direction is not considered in the analysis')
              break
      return stimulation_df
   
   def get_len_phys_recording(self, stimulation_df: pd.DataFrame) -> Union[int, float]:
      out_list = []
      curr_folder_name = os.path.basename(self.path) #prima pre, poi psilo
      pre_psilo_names = curr_folder_name.split('-')
      base =  os.path.join('/',*self.path.split('/')[:-1])

      for p in pre_psilo_names:
        SF = os.path.join(base, p)
        os.chdir(SF)
        Fneu = np.load('Fneu.npy')
        out_list.append(Fneu.shape[1])
      
      os.chdir(self.path)
      return out_list
   
   def add_keys_logicalDict(self, logical_dict: Dict[str, Any]) -> Dict[str, Any]:
      
      """
      Add integrated keys to the logical dictionary (e.g. 180, +, gray -, ...).

      Parameters:
      - logical_dict (Dict[str, Any]): The input logical dictionary.

      Returns:
      Dict[str, Any]: The logical dictionary with added keys.
      """
      stim_names= logical_dict.keys()
      if any(contains_character(string, pattern=r'\+') for string in stim_names): #check if any element of stim_names contains a '+' sign
        #ora creo le voci integrate: orientamenti, direzioni drift [+,-] e relativi grays
        pattern = r'\d+\.\d+|\d+'  #pattern per trovare numeri decimali o interi in una stringa
        #ottieni il primo elemento trovato da re.findall(pattern, elem) per ciascun elemento in stim_names che restituisce almeno una corrispondenza
        new_keys = [re.findall(pattern, elem)[0] for elem in stim_names if re.findall(pattern, elem)]
        new_keys = list(set(new_keys))+['+','-']; new_keys =new_keys + ['gray '+ n for n in new_keys] #new_keys ora contiene le voci integrate
        
        for new_key in new_keys:
          plus_minus = new_key[-1] #serve solo per keys non numeriche
          if "+" not in new_key and "-" not in new_key: #i.e. if new_key è un numero
            key_plus = new_key+'+'; key_minus = new_key+'-'; alt_keys = [key_plus, key_minus]
          elif 'gray' in new_key: #dovrebbero entrare solo gray+ e gray-
            alt_keys = [key for key in logical_dict.keys() if 'gray' in key and plus_minus in key]
          else:
            alt_keys = [key for key in logical_dict.keys() if not('gray' in key) and plus_minus in key]
          # Concatenate the arrays vertically (axis=0) to form a single array
          arrays_list = []
          for k in alt_keys:
            try:
              arrays_list.append(logical_dict[k])
            except:
              print(k+' is missing')
          concatenated_array = np.concatenate(arrays_list, axis=0)
          logical_dict[new_key] = concatenated_array[np.argsort(concatenated_array[:, 0])] # Sort the rows in ascending order based on the first column (index 0)
      return logical_dict
    
def get_OSI(stimulation_data_obj, phys_recording: np.ndarray, n_it: int =0, change_existing_dict_files=True) -> Tuple[Dict, pd.DataFrame, Dict]:
  """
  Calculate Orientation Selectivity Index (OSI) based on stimulation data and physiological recordings.

  Parameters:
  - stimulation_data_obj: Stimulation data object.
  - phys_recording (np.ndarray): Physiological recording data.
  - n_it (int): Iteration index.
  - change_existing_dict_files (bool): Flag to indicate whether to change existing dictionary files.

  Returns:
  - Tuple[Dict, pd.DataFrame, Dict]: Tuple containing Increase_stim_vs_pre (DF/F stim vs pre), Tuning_curve_avg_DF (average tuning curve for each cell + preferred ori and OSI), and Cell_ori_tuning_curve_sem.
  """
  #phys_recording_type can be set to F, Fneu, F_neuSubtract, DF_F, DF_F_zscored
  #averaging_window può anche essere settato come intero, che indichi il numero di frame da considerare
  logical_dict = stimulation_data_obj.logical_dict[n_it]
  numeric_keys, _ = get_orientation_keys(logical_dict)
  Increase_stim_vs_pre = {}; Cell_ori_tuning_curve_mean = {}; Cell_ori_tuning_curve_sem ={}
  for i, key in enumerate(numeric_keys): #per ogni orientamento...
    grating_phys_recordings = stimulation_data_obj.get_stim_phys_recording(key, phys_recording, idx_logical_dict=n_it)
    gray_phys_recordings = stimulation_data_obj.get_stim_phys_recording(key, phys_recording, idx_logical_dict=n_it,get_pre_stim=True)
    Avg_PreStim = np.mean(gray_phys_recordings, axis = 2) #medio i valori di fluorescenza nei averaging_window frame prima dello stimolo (gray)
    Avg_stim = np.mean(grating_phys_recordings, axis = 2) #medio i valori di fluorescenza nei averaging_window frame dello stimolo
    Increase_stim_vs_pre[key] = (Avg_stim-Avg_PreStim)/Avg_PreStim #i.e.  (F - F0) / F0
    Cell_ori_tuning_curve_mean[key] = np.nanmean(Increase_stim_vs_pre[key],axis=0)
    Cell_ori_tuning_curve_sem[key] = SEMf(Increase_stim_vs_pre[key]) 
  Tuning_curve_avg_DF= compute_OSI(Cell_ori_tuning_curve_mean)
  Tuning_curve_avg_DF['Trace goodness'] = trace_goodness_metric(phys_recording)

  return Increase_stim_vs_pre, Tuning_curve_avg_DF, Cell_ori_tuning_curve_sem

def compute_OSI(Cell_ori_tuning_curve_mean: Dict)-> pd.DataFrame:
  """
  Compute Orientation Selectivity Index (OSI) from Cell_ori_tuning_curve_mean.

  Parameters:
  - Cell_ori_tuning_curve_mean (Dict): Dictionary containing orientation tuning curve means for each cell.

  Returns:
  - pd.DataFrame: DataFrame containing the OSI values.
  """
  Tuning_curve_avg_DF = pd.DataFrame(Cell_ori_tuning_curve_mean)
  or_most_active = Tuning_curve_avg_DF.idxmax(axis=1).to_numpy()
  OSI_v = np.full_like(or_most_active, np.nan)

  for r_idx, max_or in enumerate(or_most_active):
    p_ors = get_parallel_orientations(max_or)
    if '360' not in Cell_ori_tuning_curve_mean.keys() and 360 in p_ors:
      p_ors.remove(360)
    ortho_ors = get_orthogonal_orientations(max_or)
    R_pref = Tuning_curve_avg_DF.loc[r_idx,[max_or]].to_numpy()
    #R_pref = np.nanmean(Tuning_curve_avg_DF.loc[r_idx,[str(ori) for ori in p_ors]]) #qui max è considerato sommando l'orientamento parallelo
    R_ortho = np.nanmean(Tuning_curve_avg_DF.loc[r_idx,[str(ori) for ori in ortho_ors]])
    OSI_v[r_idx] = (R_pref -R_ortho)/(R_pref + R_ortho)

  Tuning_curve_avg_DF['Preferred or'] = or_most_active
  Tuning_curve_avg_DF['OSI'] = OSI_v

  return Tuning_curve_avg_DF

def get_parallel_orientations(orientation: Union[int, str]) -> List[int]:
  """
  Given an orientation, returns a list of parallel orientations (e.g. 0° -> 0° and 180°).

  Parameters:
  - orientation (Union[int, str]): The input orientation.

  Returns:
  - List[int]: A list containing the input orientation and its parallel orientations.
  """
  orientation = int(orientation); p_orientations = [orientation]; flat_ors = [0,180,360]
  if orientation not in flat_ors:
      if orientation<180:
        p_orientations.append(orientation+180)
      elif orientation > 180:
        p_orientations.append(orientation-180)
  else: 
    p_orientations = flat_ors #p_orientations = [ori for ori in flat_ors if ori != orientation]
  return p_orientations


def get_orthogonal_orientations(orientation: Union[int, str]) -> List[int]:
  """
  Given an orientation, returns a list of orthogonal orientations (e.g. 0° -> 90° and 270°).

  Parameters:
  - orientation (Union[int, str]): The input orientation.

  Returns:
  - List[int]: A list containing the orthogonal orientations corresponding to the parallel orientations.
  """
  orientation = int(orientation); orthogonals_ors = []
  p_orientations = get_parallel_orientations(orientation)
  for p_or in p_orientations:
    if p_or<270:
      orthogonals_ors.append(p_or+90)
    elif p_or<360:
      orthogonals_ors.append(p_or-360+90)
  return orthogonals_ors


def get_orientation_keys(Mean_SEM_dict):
  numeric_keys_int = [] #    # Creazione di una lista vuota chiamata 'numeric_keys_int' per memorizzare chiavi numeriche come interi.
  for key in Mean_SEM_dict.keys(): # Iterazione attraverso tutte le chiavi nel dizionario 'Mean_SEM_dict'.
      if key.isnumeric(): # Verifica se la chiave è una stringa numerica.
          numeric_keys_int.append(int(key)) # Se la chiave è numerica, la converte in un intero e la aggiunge a 'numeric_keys_int'.

  numeric_keys_int = sorted(numeric_keys_int) # Ordina la lista 'numeric_keys_int' in ordine crescente.
  numeric_keys = [str(num) for num in numeric_keys_int]  # Creazione di una nuova lista 'numeric_keys' che contiene le chiavi di numeric_keys_int convertite in stringhe.

  return numeric_keys, numeric_keys_int

#DA METTERE A POSTO
def trace_goodness_metric(phys_data: np.ndarray) -> np.ndarray:
    """
    Calculate a metric indicating the "goodness" of a of the physiological signal. Originally defined for 2p data.

    Parameters:
    - phys_data (np.ndarray): The physiological data (n cells x timebins).

    Returns:
    - np.ndarray: goodness metric for each cell
    """
    if len(phys_data.shape)>1:
      quartile_25 = np.percentile(phys_data, 25,axis=1)
      quartile_99 = np.percentile(phys_data, 99,axis=1)
      STDs_Q1 = []
      for i,q25 in enumerate(quartile_25):
        traccia = phys_data[i,:]
        dati_primo_quartile =  traccia[(traccia <= q25)]
        STDs_Q1.append(np.std(dati_primo_quartile))
      STDs_Q1 = np.array(STDs_Q1)
      metrica = quartile_99/STDs_Q1

    else:
      quartile_5 = np.percentile(phys_data, 5)
      quartile_95 = np.percentile(phys_data, 95)
      metrica = (quartile_95-quartile_5)/quartile_5

    # Handle cases where the metric is infinite
    if len(phys_data.shape)>1:
      metrica[metrica==np.Inf] =0
    else:
      if metrica==np.Inf:
         metrica=0
    return metrica

def dF_F_Yuste_method(Fluorescence,timepoint):
  '''
  Input:
  Fluorescence: matrice delle fluorescenze (nr cellule x timebins)
  timepoint: intero che rappresenta il timepoint desiderato
  '''
  # Definizione di una costante 'Frames_10s' con il valore 10 secondi convertito in frames (alla freq. di acquisizione (30 Hz))
  Frames_10s = 10 * 30
  traccia = Fluorescence[:,timepoint-Frames_10s:timepoint] #prendi la traccia di tutte le cellule da 10secondi prima del timepoint fino al timepoint
  median_Fluorescence = np.percentile(traccia, 50,axis=1) # Calcolo della mediana della 'traccia' lungo l'asse delle colonne (axis=1)
  Avg_first50 = []
  for i,q50 in enumerate(median_Fluorescence): #per ciascuna cellula...
    traccia = Fluorescence[i,timepoint-Frames_10s:timepoint] # prendi la fluorescenza tra 10secondi prima del timepoint fino al timepoint
    dati_prima_meta =  traccia[(traccia <= q50)] #seleziona i valori della fluorescenza della cellula al di sotto della mediana
    Avg_first50.append(np.mean(dati_prima_meta)) # appendi la media dei valori selezionati nella linea precedente alla lista delle medie della prima metà dei dati Avg_first50
  Avg_first50 = np.array(Avg_first50)  # Convertire 'Avg_first50' in un array numpy.
  dF_F_Yuste_timepoint = (Fluorescence[:,timepoint]-Avg_first50)/Avg_first50  # Calcolo di dF/F per il 'timepoint' corrente.
  return dF_F_Yuste_timepoint #fluorescenza normalizzata del timepoint

def single_session_analysis(Session_folder='manual_selection', session_name='none',Force_reanalysis = False, change_existing_dict_files=True, PCA_yn = 1):
  nr_PCA_components=10  # Impostazione del numero di componenti PCA desiderate (predefinito a 10).
  getoutput=False
  if Session_folder=='manual_selection':
    getoutput=True
    from google.colab import drive
    drive.mount('/content/drive')
    #ricerda del folder della sessione di interesse
    Main_folder = '/content/drive/MyDrive/esperimenti2p_Tausani/'
    sbj = multioption_prompt(os.listdir(Main_folder), in_prompt='Which subject?')
    sbj_folder = os.path.join(Main_folder,sbj)
    session_name = multioption_prompt(os.listdir(sbj_folder), in_prompt='Which session?')
    Session_folder = os.path.join(sbj_folder,session_name) 
    os.chdir(Session_folder)# cambia la directory corrente alla cartella della sessione selezionata

    #se vuoi rianalizzare cancella tutti i foldere delle analisi preesistenti
    if Force_reanalysis:
      remove_dirs(root = Session_folder, folders_to_remove =['Analyzed_data','Plots'])

  stim_data = CL_stimulation_data(Session_folder, Stim_var = 'Orientamenti', Time_var = 'N_frames',not_consider_direction = False)
  df_list, StimVec_list,len_Fneu_list = stim_data.get_stim_data()
  
  F_raw = np.load('F.npy')
  Fneu_raw = np.load('Fneu.npy')
  iscell = np.load('iscell.npy') #iscell[:,0]==1 sono cellule
  if getoutput==True: #da rimuovere
    stat = np.load('stat.npy', allow_pickle=True)
    stat = stat[iscell[:,0]==1]

  def single_session_processing(stim_data_obj,n_it, F,Fneu,iscell,getoutput,change_existing_dict_files):
    Session_folder = stim_data_obj.path; session_name = os.path.basename(Session_folder)
    df = stim_data_obj.Stim_dfs[n_it]; StimVec = stim_data_obj.StimVecs[n_it]
    StimVec, df, [F,Fneu] = cut_recording(StimVec,df, [F[iscell[:,0]==1,:], Fneu[iscell[:,0]==1,:]] , df_Time_var='N_frames', do_custom_cutting = getoutput)
    F_neuSubtract = F - 0.7*Fneu
    F_neuSubtract[F_neuSubtract<0]=0 #normalizzare?
    F_to_use = F_neuSubtract
    
    os.makedirs(os.path.join(Session_folder,'Analyzed_data/'), exist_ok=True); os.chdir(os.path.join(Session_folder,'Analyzed_data/'))
    logical_dict = stim_data_obj.create_logical_dict(n_it, change_existing_dict_files=change_existing_dict_files)
    # F0 = np.mean(F_neuSubtract[:,logical_dict['final gray']], axis = 1)[:, np.newaxis];DF_F = (F_neuSubtract - F0)/ F0; DF_F_zscored = zscore(DF_F, axis=1)
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

    get_stats_results = stim_data_obj.get_stats(phys_recording = F_to_use, functions_to_apply=[get_OSI])    

    
    os.makedirs(os.path.join(Session_folder,'Plots/'), exist_ok=True); os.chdir(os.path.join(Session_folder,'Plots/'))
    p_value,perc_diff_wGray2, perc_diff_wGray2_vector = Comparison_gray_stim(F_to_use, logical_dict,session_name)
    indices_tuned = np.where([cell_OSI_dict['OSI']>0.5])[1]
    indices_responding = np.where([perc_diff_wGray2_vector>6])[1]
    nr_segmented_cells = len(perc_diff_wGray2_vector)
    nr_responsive_cells = len(indices_responding)

    if nr_responsive_cells>0:
      fraction_responding = nr_responsive_cells/len(perc_diff_wGray2_vector)
      _,perc_diff_wGray2_responding_only,_ =Comparison_gray_stim(F_to_use[indices_tuned,:], logical_dict,session_name, omitplot = True)
      fraction_tuned =  len(indices_tuned)/len(perc_diff_wGray2_vector)
      indices_responding_and_tuned = np.intersect1d(indices_responding,indices_tuned)
      fraction_responding_tuned =  len(indices_responding_and_tuned)/nr_responsive_cells
      avg_tuning_all_responding= np.mean(cell_OSI_dict['OSI'][indices_responding])
      avg_tuning_all_tuned_responding = np.mean(cell_OSI_dict['OSI'][indices_responding_and_tuned])

      session_name_column = [session_name] * nr_responsive_cells
      session_name_column = [s_name+'_cell'+str(nr) for nr,s_name in enumerate(session_name_column)]
      perc_diff_wGray2_col = perc_diff_wGray2_vector[indices_responding]
      tuning_col = cell_OSI_dict['OSI'][indices_responding]
      responding_cells_df = pd.DataFrame({'responding cell name': session_name_column, '% change wrt grey2': perc_diff_wGray2_col, 'OSI': tuning_col})
      # if PCA_yn == 1 and nr_responsive_cells>nr_PCA_components:
      #   F_responding = F_to_use[indices_responding,:]
      #   mean = np.mean(F_responding, axis=0)
      #   std_dev = np.std(F_responding, axis=0)
      #   data_standardized = (F_responding - mean) / std_dev
      #   # Step 2: Perform PCA
      #   pca = PCA(n_components=nr_PCA_components) # You can change the number of components as needed
      #   pca.fit(data_standardized)
      #   eigenspectra = pca.explained_variance_ratio_

    else:
      perc_diff_wGray2_responding_only = np.nan
      fraction_responding =np.nan
      fraction_tuned =  np.nan
      fraction_responding_tuned = np.nan
      avg_tuning_all_responding = np.nan
      avg_tuning_all_tuned_responding = np.nan
      responding_cells_df = []


    # value_counts = Counter(cell_OSI_dict['PrefOr'][indices_tuned])
    # for value, count in value_counts.items():
    #   print(f"{value}: {count} times")

    Plotting_functions.summaryPlot_AvgActivity(Mean_SEM_dict_F,session_name, Fluorescence_type = 'F_neuSubtract')
    # if getoutput==True: #da rimouovere
    #   Plotting_functions.summaryPlot_OSI(cell_OSI_dict,Cell_Max_dict_F,session_name,stat=stat,Fluorescence_type='F')
    # else:
    #   Plotting_functions.summaryPlot_OSI(cell_OSI_dict,Cell_Max_dict_F,session_name,stat=[],Fluorescence_type='F')
    #if getoutput:
    return locals()
  
  results_list = []
  c=0
  n_it =0
  for df, StimVec, len_Fneu in zip(df_list, StimVec_list,len_Fneu_list):
    F = F_raw[:,c:c+len_Fneu]
    Fneu = Fneu_raw[:,c:c+len_Fneu]
    c = len_Fneu
    get_stats_results = single_session_processing(stim_data,n_it,F,Fneu,iscell,getoutput,change_existing_dict_files)
    return get_stats_results
    return_dict = single_session_processing(session_name,Session_folder,F,Fneu,iscell,df,StimVec,getoutput,change_existing_dict_files)
    results_list.append(return_dict)
  return results_list
    


def Analyze_all(Force_reanalysis = True, select_subjects = True, change_existing_dict_files=True, lower_bound_timebins_concat = 45000):
  from google.colab import drive
  drive.mount('/content/drive')
  type_corr = input('which fluorescence do you want to use for correlations? (options: F, Fneu, F_neuSubtract)')
  V_names_corrs = ['Corr all trace','Corr spontanea1','Corr spontanea2','Corr stim', 'Corr gray']
  correlation_dict = {}
  Concat_responsive_cells = None
  correlation_stats_tensor = None
  Main_folder = '/content/drive/MyDrive/esperimenti2p_Tausani/'
  columns = ['Session', 'PCA1', 'PCA2', 'PCA3','PCA4','PCA5','PCA6','PCA7','PCA8','PCA9','PCA10']  # Replace with your column names
  PCA_explVar_df = pd.DataFrame(columns=columns)
  nr_responsive_cells_dict = {}
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
              remove_dirs(root = Session_folder, folders_to_remove =['Analyzed_data','Plots'])
              
            return_dict = single_session_analysis(Session_folder=Session_folder, session_name=session_name, change_existing_dict_files=change_existing_dict_files)
            indices_responding = return_dict['indices_responding']
            if ('F_responding' in return_dict):
              if Concat_responsive_cells is None:
                Concat_responsive_cells = return_dict['F_responding']
                nr_responsive_cells_dict[session_name] = return_dict['nr_responsive_cells']
                column_len = Concat_responsive_cells.shape[1]
              else:
                cells_to_concat = return_dict['F_responding']
                nr_timebins = cells_to_concat.shape[1]
                if not(nr_timebins==column_len) and (nr_timebins >= lower_bound_timebins_concat):
                  column_len = min(nr_timebins, column_len)
                if (nr_timebins >= lower_bound_timebins_concat):
                  Concat_responsive_cells = np.concatenate((Concat_responsive_cells[:,:column_len], cells_to_concat[:,:column_len]), axis=0)
                  nr_responsive_cells_dict[session_name] = return_dict['nr_responsive_cells']
            else:
               print("\033[1mNOTE:Session "+ session_name+" DOES NOT have F_responding"+"\033[0m")

            responding_cells_df = return_dict['responding_cells_df']
            comp_item = np.zeros(2)
            comp_item[0] = return_dict['p_value']
            comp_item[1] = return_dict['perc_diff_wGray2']
            comp_list.append([session_name,comp_item,return_dict['fraction_responding'],return_dict['fraction_tuned'],return_dict['fraction_responding_tuned'],return_dict['avg_tuning_all_responding'],return_dict['avg_tuning_all_tuned_responding'],return_dict['nr_segmented_cells']])
            #vado a raccogliere le statistiche di correlazione che mi interessano
            correlation_dict[session_name] =compute_correlation(return_dict[type_corr], return_dict['logical_dict'])
            correlation_stats = np.zeros((correlation_dict[session_name].shape[0],2))
            for matrix_idx in range(correlation_dict[session_name].shape[0]):
               correlation_tensor = correlation_dict[session_name][matrix_idx,:,:] #prendo ciascuna delle matrici di correlazione
               if isinstance(responding_cells_df, pd.DataFrame):
                ct = correlation_tensor[indices_responding,:]
                ct = ct[:,indices_responding] 
                responding_cells_df[V_names_corrs[matrix_idx]] = np.mean(ct, axis=0) #columnwise average of correlation per each responding cell
               correlation_vec_no_symmetry = correlation_tensor[np.triu_indices(correlation_tensor.shape[0], k=1)] #numpy.triu_indices(n, k=0, m=None) Return the indices for the upper-triangle of an (n, m) array.
               correlation_stats[matrix_idx,0] = np.mean(correlation_vec_no_symmetry)
               correlation_stats[matrix_idx,1] = np.std(correlation_vec_no_symmetry)
            if correlation_stats_tensor is None:
                correlation_stats_tensor = np.expand_dims(correlation_stats, axis=0)
            else:
                correlation_stats = np.expand_dims(correlation_stats, axis=0)
                correlation_stats_tensor = np.concatenate((correlation_stats_tensor, correlation_stats), axis=0)
            
            if isinstance(responding_cells_df, pd.DataFrame):
               if 'responding_cells_df_ALL' in locals():
                responding_cells_df_ALL = pd.concat([responding_cells_df_ALL, responding_cells_df], axis=0)
               else:
                responding_cells_df_ALL = responding_cells_df

            if 'eigenspectra' in return_dict:
              eigenspectra = return_dict['eigenspectra']
              new_row = {
                  'Session': session_name,
                  'PCA1':eigenspectra[0],
                  'PCA2':eigenspectra[1], 
                  'PCA3':eigenspectra[2], 
                  'PCA4':eigenspectra[3], 
                  'PCA5':eigenspectra[4], 
                  'PCA6':eigenspectra[5], 
                  'PCA7':eigenspectra[6], 
                  'PCA8':eigenspectra[7], 
                  'PCA9':eigenspectra[8], 
                  'PCA10':eigenspectra[9], 
              }
              PCA_explVar_df = PCA_explVar_df.append(new_row, ignore_index=True)

               
                
  sesson_names = [item[0] for item in comp_list]
  p_values = [item[1][0] for item in comp_list]
  Percent_increase = [item[1][1] for item in comp_list]
  perc_responding_V = [item[2]*100 for item in comp_list]
  perc_tuned_V = [item[3]*100 for item in comp_list]
  perc_responding_tuned_V = [item[4]*100 for item in comp_list]
  avg_tuning_all_responding_V = [item[5] for item in comp_list]
  avg_tuning_all_tuned_responding_V = [item[6] for item in comp_list]
  nr_segmented_cells = [item[7] for item in comp_list]
  df_stim_vs_gray = pd.DataFrame({'Session name': sesson_names, 'P_val': p_values, '% change wrt grey2': Percent_increase, '% responding (>6%)': perc_responding_V,
                                  '% tuned (OSI>0.5)':perc_tuned_V, '% responding and tuned':perc_responding_tuned_V, 'Mean tuning of responsive': avg_tuning_all_responding_V, 'Mean tuning responsive and tuned': avg_tuning_all_tuned_responding_V, 
                                  'Nr_segmented_cells': nr_segmented_cells}) 

  s = 0
  for k in nr_responsive_cells_dict:
    s = s+nr_responsive_cells_dict[k]
    nr_responsive_cells_dict[k] = [nr_responsive_cells_dict[k],s]       
  
  for col in zip(np.transpose(correlation_stats_tensor[:,:,0]), V_names_corrs):
      df_stim_vs_gray[col[1]] = col[0]

  return locals()


def Comparison_gray_stim(Fluorescence, logical_dict,session_name, omitplot = False):
  
  n_b = 200
  str_keys, list_keys = get_orientation_keys(logical_dict)
  Activity_arr = np.zeros((Fluorescence.shape[0],n_b,4))
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
  perc_diff_wGray2_vector = ((Activity_arr2[:,2] - Activity_arr2[:,5])/Activity_arr2[:,5])*100
  perc_diff_wGray = np.nanmean(((Activity_arr2[:,2] - Activity_arr2[:,3])/Activity_arr2[:,3])*100)
  perc_diff_wGray2 = np.nanmean(((Activity_arr2[:,2] - Activity_arr2[:,5])/Activity_arr2[:,5])*100)
  _, p_value = stats.wilcoxon(Activity_arr2[:,2] - Activity_arr2[:,3], alternative='greater')
  if omitplot==False:
    plt.title("P value "+str("{:.2e}".format(p_value))+', % diff '+str("{:.2}".format(perc_diff_wGray)))
    plt.savefig(session_name+'Fluorescence_periods_comparison.png')
    plt.show()

  #return Activity_arr,Activity_arr2
  return p_value,perc_diff_wGray2, perc_diff_wGray2_vector

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
  sveglio_yn = int(input('vuoi analizzare animali svegli o anestetizzati? (1=sveglio, 0=anestetizzato)'))
  psilo_type = int(input('pre vs psilo alta o bassa? (1=alta, 0=bassa)'))
  Metrics_list = '\n'.join([f'{i}: {metr}' for i, metr in enumerate(df.columns)])
  idx_Metrics = int(input('Which variable?\n'+Metrics_list))
  stat_of_interest = df[df.columns[idx_Metrics]].to_numpy()
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
    if session_info[-1]=='sveglio' and sveglio_yn==0:
      continue
    elif not(session_info[-1]=='sveglio') and sveglio_yn==1:
      continue
    if ('alta' in session_info[2]) and psilo_type==0:
      continue
    elif (not('alta' in session_info[2]) and not('pre' in session_info[2])) and psilo_type==1: #ora dovrebbe andare bene
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
  animal_ids = set([element.split('_')[0] for element in Exp_day_list])
  colors = [colorsys.hsv_to_rgb(i/len(animal_ids), 1, 1) for i in range(len(animal_ids))]
  color_dict = {id: colors[i] for i, id in enumerate(animal_ids)}
  for i,row in enumerate(tabella_comparazioni):
      animal = Exp_day_list[i].split('_')[0]
      plt.plot(row, color=color_dict[animal], marker = 'o', markersize=15)
  labels = ['pre1', 'pre2', 'psilo1', 'psilo2']
  proxy_artists = [plt.Line2D([0], [0], marker='o', color='w', label=f'{animal_id}', markersize=15, markerfacecolor=color)
                for animal_id, color in color_dict.items()]
  plt.legend(handles=proxy_artists, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=len(color_dict), frameon=False)
  plt.xticks(range(len(labels)), labels)
  plt.ylabel('')
  plt.show()

  return tabella_comparazioni,Exp_day_list


def stat_comparison_betw_cells(responding_cells_df_ALL): 
  psilo_ab = int(input('psilo alta o bassa? (1= alta, 0=bassa)'))
  pre_cells = responding_cells_df_ALL[responding_cells_df_ALL["responding cell name"].str.contains("pre")]
  pre_cells = pre_cells[~pre_cells["responding cell name"].str.contains("sveglio")]

  psilo_cells = responding_cells_df_ALL[responding_cells_df_ALL["responding cell name"].str.contains("psilo")]
  psilo_cells = psilo_cells[~psilo_cells["responding cell name"].str.contains("sveglio")]
  if psilo_ab == 0:
    l2 = 'Psilo_bassa'
    psilo_cells = psilo_cells[~psilo_cells["responding cell name"].str.contains("alta")]
  else:
    l2 = 'Psilo_alta'
    psilo_cells = psilo_cells[psilo_cells["responding cell name"].str.contains("alta")]

  _, p_value = stats.mannwhitneyu(pre_cells['Corr gray'], psilo_cells['Corr gray'])
  var_list = '\n'.join([f'{i}: {x}' for i, x in enumerate(responding_cells_df_ALL.columns)])
  idx_vars = ast.literal_eval(input('Which variables do you want to compare? (write in [])?\n'+var_list))
  for v in idx_vars:
    _, p_value = stats.mannwhitneyu(pre_cells[responding_cells_df_ALL.columns[v]], psilo_cells[responding_cells_df_ALL.columns[v]])
    plt.boxplot([pre_cells[responding_cells_df_ALL.columns[v]], psilo_cells[responding_cells_df_ALL.columns[v]]], labels=['Pre', l2])
    plt.ylabel(responding_cells_df_ALL.columns[v])
    plt.title('p value: '+str(p_value))
    plt.show()

