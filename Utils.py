import numpy as np
import os
import re
import pandas as pd
import glob
from scipy.stats import mode
from scipy.stats import zscore
from sklearn.decomposition import PCA
from scipy import stats
import shutil
import matplotlib.pyplot as plt
import Plotting_functions
from Plotting_functions import *
from Generic_tools.Generic_list_operations import *
import ast
import colorsys
from collections import Counter

def get_orientation_keys(Mean_SEM_dict):
  numeric_keys_int = [] #    # Creazione di una lista vuota chiamata 'numeric_keys_int' per memorizzare chiavi numeriche come interi.
  for key in Mean_SEM_dict.keys(): # Iterazione attraverso tutte le chiavi nel dizionario 'Mean_SEM_dict'.
      if key.isnumeric(): # Verifica se la chiave è una stringa numerica.
          numeric_keys_int.append(int(key)) # Se la chiave è numerica, la converte in un intero e la aggiunge a 'numeric_keys_int'.

  numeric_keys_int = sorted(numeric_keys_int) # Ordina la lista 'numeric_keys_int' in ordine crescente.
  numeric_keys = [str(num) for num in numeric_keys_int]  # Creazione di una nuova lista 'numeric_keys' che contiene le chiavi di numeric_keys_int convertite in stringhe.

  return numeric_keys, numeric_keys_int

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


def SEMf(Fluorescence_matrix):
   #funzione per calcolare la SEM nei dati di fluorescenza 
   Std = np.nanstd(Fluorescence_matrix, axis=0) # Calcolo della deviazione standard lungo l'asse delle colonne della matrice 'Fluorescence_matrix'.
   nr_neurons = Fluorescence_matrix.shape[0] # Determinazione del numero di neuroni considerando le righe della matrice. Andrebbe cambiato togliendo i nan
   SEM = Std/np.sqrt(nr_neurons) # Calcolo dell'errore standard della media (SEM)
   return SEM

def single_session_analysis(Session_folder='manual_selection', session_name='none',Force_reanalysis = False, change_existing_dict_files=True, PCA_yn = 1):
  nr_PCA_components=10  # Impostazione del numero di componenti PCA desiderate (predefinito a 10).
  getoutput=False
  if Session_folder=='manual_selection':
    getoutput=True
    from google.colab import drive
    drive.mount('/content/drive')
    #ricerda del folder della sessione di interesse
    Main_folder = '/content/drive/MyDrive/esperimenti2p_Tausani/'
    dir_list = os.listdir(Main_folder) # lista dei file e delle cartelle all'interno di Main_folder
    sbj = multioption_prompt(dir_list, in_prompt='Which subject?')
    sbj_list = '\n'.join([f'{i}: {sbj}' for i, sbj in enumerate(dir_list)]) # questa linea serve per creare il prompt di selezione del soggetto
    idx_sbj = int(input('Which subject?\n'+sbj_list))
    sbj_folder = os.path.join(Main_folder,sbj)
    #le seguenti 4 righe fanno lo stesso delle precedenti 4, solo per sessione e non per soggetto
    dir_list = os.listdir(sbj_folder)
    sess_list = '\n'.join([f'{i}: {sess}' for i, sess in enumerate(dir_list)])
    idx_session = int(input('Which session?\n'+sess_list))
    session_name = dir_list[idx_session]
    Session_folder = os.path.join(sbj_folder,session_name) 
    os.chdir(Session_folder)# cambia la directory corrente alla cartella della sessione selezionata

    #se vuoi rianalizzare cancella tutti i foldere delle analisi preesistenti
    if Force_reanalysis:
      if os.path.isdir(os.path.join(Session_folder, 'Analyzed_data')):
        shutil.rmtree(os.path.join(Session_folder,'Analyzed_data/'))
      if os.path.isdir(os.path.join(Session_folder, 'Plots')):
        shutil.rmtree(os.path.join(Session_folder,'Plots/'))

  df_list, StimVec_list,len_Fneu_list = Df_loader_and_StimVec(Session_folder, not_consider_direction = False)
  
  F_raw = np.load('F.npy')
  Fneu_raw = np.load('Fneu.npy')
  iscell = np.load('iscell.npy') #iscell[:,0]==1 sono cellule
  if getoutput==True: #da rimuovere
    stat = np.load('stat.npy', allow_pickle=True)
    stat = stat[iscell[:,0]==1]

  def single_session_processing(session_name,Session_folder,F,Fneu,iscell,df,StimVec,getoutput,change_existing_dict_files):
    cut = len(StimVec)
    if getoutput:
      plt.plot(np.mean(F,axis = 0))
      # Show the plot
      plt.show()
      plt.pause(0.1)
      cut = int(input('at which frame you want to cut the series (all = ' +str(len(StimVec))+ ')?'))
      StimVec = StimVec[:cut]
      df = df[df['N_frames']<cut] #taglia fuori END? da controllare
    F = F[iscell[:,0]==1,:cut]
    Fneu = Fneu[iscell[:,0]==1,:cut]

    F_neuSubtract = F - 0.7*Fneu
    F_neuSubtract[F_neuSubtract<0]=0
    #normalizzare?

    os.makedirs(os.path.join(Session_folder,'Analyzed_data/'), exist_ok=True); os.chdir(os.path.join(Session_folder,'Analyzed_data/'))
    logical_dict = Create_logical_dict(session_name,StimVec,df, change_existing_dict_files=change_existing_dict_files)
    # F0 = np.mean(F_neuSubtract[:,logical_dict['final gray']], axis = 1)[:, np.newaxis]
    # DF_F = (F_neuSubtract - F0)/ F0
    # DF_F_zscored = zscore(DF_F, axis=1)  
    F_to_use = F_neuSubtract
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

    Mean_SEM_dict_F = Create_Mean_SEM_dict(session_name,logical_dict, F_to_use, Fluorescence_type = 'F_neuSubtract', change_existing_dict_files=change_existing_dict_files)
    Cell_Max_dict_F = Create_Cell_max_dict(logical_dict, F_to_use, session_name, averaging_window ='mode', Fluorescence_type='F_neuSubtract', change_existing_dict_files=change_existing_dict_files)
    cell_OSI_dict = Create_OSI_dict(Cell_Max_dict_F,session_name, change_existing_dict_files=change_existing_dict_files)
    Cell_stat_dict = Create_Cell_stat_dict(logical_dict, F_to_use, session_name, averaging_window ='mode', Fluorescence_type='F_neuSubtract', OSI_alternative=False, change_existing_dict_files=change_existing_dict_files)
    

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
  for df, StimVec, len_Fneu in zip(df_list, StimVec_list,len_Fneu_list):
    F = F_raw[:,c:c+len_Fneu]
    Fneu = Fneu_raw[:,c:c+len_Fneu]
    c = len_Fneu
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
              if os.path.isdir(os.path.join(Session_folder, 'Analyzed_data')):
                shutil.rmtree(os.path.join(Session_folder,'Analyzed_data/'))
              if os.path.isdir(os.path.join(Session_folder, 'Plots')):
                shutil.rmtree(os.path.join(Session_folder,'Plots/'))

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

  def get_StimVec(df):
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
    return StimVec

  # use the glob module to find the Excel file with the specified extension
  excel_files = glob.glob(os.path.join(Session_folder, "*.xlsx"))
  #print(excel_files[0])
  len_Fneu = []
  df = []
  StimVec = []
  for n,ex_f in enumerate(excel_files): #pre e psilo sono sempre ordinati. No need di ordinare ad hoc
    df.append(pd.read_excel(ex_f))
    StimVec.append(get_StimVec(df[n]))
# SE SI VUOLE UNIFICARE IL DF (OBSOLETO)
#  df = pd.concat(df_list, ignore_index=True)
#  begin_idxs = df[df['Computer_time'] == 0.0].index
#  for i in begin_idxs:
#     if i>0:#not the beginning
#        df.iloc[i:,1]=df.iloc[i:,1]+df.iloc[i-1,1] #Computer time
#        df.iloc[i:,2]=df.iloc[i:,2]+df.iloc[i-1,2] #N frames
    curr_folder_name = os.path.basename(Session_folder) #prima pre, poi psilo
    pre_psilo_names = curr_folder_name.split('-')
    base =  os.path.join('/',*Session_folder.split('/')[:-1])

    for p in pre_psilo_names:
      SF = os.path.join(base, p)
      os.chdir(SF)
      Fneu = np.load('Fneu.npy')
      len_Fneu.append(Fneu.shape[1])

  os.chdir(Session_folder)
           
  return df, StimVec, len_Fneu


def Create_logical_dict(session_name,stimoli,df, change_existing_dict_files=True):
    def contains_plus_character(vector): #function to check if any element of df['Orientamenti'].unique() contains a '+' sign
      for string in vector:
          if '+' in string:
              return True
      return False
    SBAs = ['initial gray', 'initial black', 'after flash gray', 'final gray']
    logical_dict_filename = session_name+'_logical_dict.npz'
    if not(os.path.isfile(logical_dict_filename)) or change_existing_dict_files==True:
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

def Create_Mean_SEM_dict(session_name,logical_dict, Fluorescence,  Fluorescence_type = 'F', change_existing_dict_files=True):
    #Fluorescence_type can be set to F, Fneu, F_neuSubtract, DF_F, DF_F_zscored
    SBAs = ['initial gray', 'initial black', 'after flash gray', 'final gray']
    Mean_SEM_dict_filename = session_name+Fluorescence_type+'_Mean_SEM_dict.npz'
    if not(os.path.isfile(Mean_SEM_dict_filename)) or change_existing_dict_files==True:
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


def Create_Cell_max_dict(logical_dict, Fluorescence, session_name, averaging_window ='mode', Fluorescence_type='F',change_existing_dict_files = True):
  #Fluorescence_type can be set to F, Fneu, F_neuSubtract, DF_F, DF_F_zscored

  #averaging_window può anche essere settato come intero, che indichi il numero di frame da considerare

  if not(isinstance(averaging_window, str)):
      averaging_window = str(averaging_window)
  Cell_max_dict_filename = session_name+'_'+Fluorescence_type+'_Cell_max_dict_'+averaging_window+'.npz'
  if not(os.path.isfile(Cell_max_dict_filename)) or change_existing_dict_files==True:
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
                    Cells_maxs[cell,i] = (Avg_stim-Avg_PreStim)/Avg_PreStim #i.e.  (F - F0) / F0
                    Avg_stim_V[cell,i] = Avg_stim
                    Avg_PreStim_V[cell,i] = Avg_PreStim
                    #Cells_maxs[cell,i] = Avg_stim

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
    #R_pref  = np.mean([Tuning_curve_avgSem[0,idx_max],Tuning_curve_avgSem[0,idxs_4orth_ori[idx_max+4]]])
    R_pref  = Tuning_curve_avgSem[0,idx_max]
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


def Create_OSI_dict(Cell_Max_dict,session_name, OSI_alternative=False,change_existing_dict_files=True):
  OSI_dict_filename = session_name+'_'+str(Cell_Max_dict['Fluorescence_type'])+'_OSI_dict_'+str(Cell_Max_dict['averaging_window'])+'.npz'
  if not(os.path.isfile(OSI_dict_filename)) or change_existing_dict_files==True:
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
        OSI, preferred_or = OSIf(Tuning_curve_avgSem, numeric_keys_int, idxs_4orth_ori = idxs_4orth_ori,plus180or = True)
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

def Create_Cell_stat_dict(logical_dict, Fluorescence, session_name, averaging_window ='mode', Fluorescence_type='F', OSI_alternative=True, change_existing_dict_files=True):
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
    Pl180yn = True
    OSI_writing = 'OSI_classic' + ('+180' if Pl180yn == 1 else '')
    OSI_v = np.full((nr_cells), np.nan)
    PrefOr_v = np.full((nr_cells), np.nan)
  Cell_stat_dict_filename = session_name+'_'+Fluorescence_type+'_Cell_stat_dict_'+averaging_window+OSI_writing+'.npz'
  if not(os.path.isfile(Cell_stat_dict_filename)) or change_existing_dict_files==True:
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
                    #Avg_stim = np.mean(cell_trace[row[0]:(row[0]+averaging_window)])
                    #Cells_maxs[cell,i] = Avg_stim
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