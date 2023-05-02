import os
import csv
import math
import pandas as pd
import pickle
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold 
from comet_ml import Experiment
from joblib import dump, load

def save_cv_results(cmt_exp, cv_results):
    '''
    Entrada:
        cmt_exp: experimento comet
        cv_results: scikit-learn cross-validation results
    '''

    n_splits = len(cv_results['test_f1']) 
    for i in range(n_splits):
        metrics_step = {
            'F1-Score': cv_results['test_f1'][i],
            'ROC': cv_results['test_roc_auc'][i]
        }
        cmt_exp.log_metrics(metrics_step, step=i)

    return


def new_experiment(name=None, run=None, project=None, tags=None):
  '''
  name: str
    nombre del experimento
  run: str
    numero de corrida del experimento
  project: str
    nombre del proyecto al que pertenece el experimento
  tags: list()
    etiquetas a agregar al experimento
  '''

  if name is not None\
  and project is not None\
  and run is not None:
    # Create an experiment with your api key
    exp = Experiment(
        api_key="4758xmPFeCo55aLIV3xaKbPU2",
        project_name=project,
        workspace="naguileraleal"
    )

    exp_name = name+'-run-'+run
    exp.set_name(exp_name) # Nombre de este experimento
    
    if tags:
      exp.add_tags(tags) # Tags
    return (exp, exp_name)


def cargalo_en_comet_maestro(cv,proyecto, name, tags):
    exp = Experiment(api_key='v55FUAgeHHarLxPg0Fg72SCOe',  
                     project_name=proyecto, # Nombre del proyecto donde se registran los experimentos
                     auto_param_logging=False) 
    exp.set_name(name) # Nombre de este experimento
    exp.add_tags(tags) # Tags
    save_cv_results(exp, cv)

    exp.end()


def divideConjuntos(df:pd.DataFrame, n_splits=5, random_state=123, verbose=False):
    '''
    divide el dataframe que se pasa como argumento en grupos de test y validacion, separa las labels de los datos. 
    
    '''
    
    # SE FIJA CUANTO VALE LA SUMA DE LOS PESOS PARA CADA EVENTO
    mascara_signal = df['Label'] == 's'
    mascara_background = df['Label'] == 'b'
    valor_esperado_signal = np.sum(df[mascara_signal]['Weight'])
    valor_esperado_background = np.sum(df[mascara_background]['Weight'])
    if verbose:
        print('Suma de los pesos de los evento señal (valor esperado del evento en un año): ',valor_esperado_signal)
        print('Suma de los pesos de los evento background (valor esperado del evento en un año): ',valor_esperado_background)
    
    # SEPARA LAS LABELS DE LOS DATOS
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # GENERA LOS CJTOS DE TRAIN Y VAL, MANTENIENDO EN CADA UNO LA PROPORCION ENTRE LOS EVENTOS QUE TENÍA EL DF ORIGINAL
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    train_index, val_index = next(skf.split(X, y))
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    if verbose:
        print('Proporción de clases en el conjunto de entrenamiento:', y_train.value_counts(normalize=True))
        print('\nProporción de clases en el conjunto de validación:', y_val.value_counts(normalize=True))
    
    # GUARDA LOS VALORES DE LOS PESOS
    pesos_train, pesos_val = X_train['Weight'], X_val['Weight']
    
    return X_train, X_val, y_train, y_val, pesos_train, pesos_val, valor_esperado_signal, valor_esperado_background
    
    
def actualizaPesos(X_train,X_val,y_train,y_val,pesos_train,pesos_val,valor_esperado_signal,valor_esperado_background,verbose=False):
    '''
    en cada conjunto mantiene la suma de los pesos que tenia el df original
    '''
    
    # MULTIPLICO LOS PESOS PARA QUE SU SUMA EN CADA CONJUNTO SEA LA MISMA QUE DABA ANTES DE SEPARARLOS
    # PRIMERO CALCULO CUANTO DA LA SUMA AHORA
    suma_pesos_signal_train = np.sum(X_train[y_train=='s']['Weight'])
    suma_pesos_bg_train = np.sum(X_train[y_train=='b']['Weight'])
    suma_pesos_signal_val = np.sum(X_val[y_val=='s']['Weight'])
    suma_pesos_bg_val = np.sum(X_val[y_val=='b']['Weight'])
    if verbose:
        print('\nAntes de corregir:')
        print('\nSuma de los pesos de los evento señal en el cjto de train: ', suma_pesos_signal_train)
        print('Suma de los pesos de los evento background en el cjto de train: ', suma_pesos_bg_train)
        print('Suma de los pesos de los evento señal en el cjto de validación: ', suma_pesos_signal_val) 
        print('Suma de los pesos de los evento background en el cjto de validación: ', suma_pesos_bg_val)
    
    # DESPUES CALCULO LOS COEFS POR LOS QUE TENGO QUE MULTIPLICAR PARA QUE LOS PESOS SUMEN LO MISMO DE ANTES
    coeficiente_signal_train = valor_esperado_signal/suma_pesos_signal_train
    coeficiente_background_train = valor_esperado_background/suma_pesos_bg_train
    coeficiente_signal_val = valor_esperado_signal/suma_pesos_signal_val
    coeficiente_background_val = valor_esperado_background/suma_pesos_bg_val
    
    # ME DEFINO DFS PARA HACER LA MULTIPLICACION
    coeficientes_train = (y_train=='s') * coeficiente_signal_train + (y_train=='b') * coeficiente_background_train
    coeficientes_val = (y_val=='s') * coeficiente_signal_val + (y_val=='b') * coeficiente_background_val

    # MULTIPLICO
    X_train.loc[:, 'Weight'] = pesos_train * coeficientes_train
    X_val.loc[:, 'Weight'] = pesos_val * coeficientes_val
    
    if verbose:
        print('\nDespues de corregir:')
        suma_pesos_signal_train = np.sum(X_train[y_train=='s']['Weight'])
        suma_pesos_bg_train = np.sum(X_train[y_train=='b']['Weight'])
        suma_pesos_signal_val = np.sum(X_val[y_val=='s']['Weight'])
        suma_pesos_bg_val = np.sum(X_val[y_val=='b']['Weight'])
        print('\nSuma de los pesos de los evento señal en el cjto de train: ', suma_pesos_signal_train)
        print('Suma de los pesos de los evento background en el cjto de train: ', suma_pesos_bg_train)
        print('Suma de los pesos de los evento señal en el cjto de validación: ', suma_pesos_signal_val) 
        print('Suma de los pesos de los evento background en el cjto de validación: ', suma_pesos_bg_val)
        
    pesos_train, pesos_val = X_train['Weight'], X_val['Weight']
    X_train = X_train.drop(columns=['Weight'])
    X_val = X_val.drop(columns=['Weight'])
    
#     X_train = X_train.reset_index(drop=True)
#     y_train = y_train.reset_index(drop=True)
#     X_val = X_val.reset_index(drop=True)
#     y_val = y_val.reset_index(drop=True)
    
    return X_train, X_val, y_train, y_val, pesos_train, pesos_val



def drop999(df:pd.DataFrame):
    ''''dropea todas las columnas que tengan celdas con -999'''
    
    df_nan = df.replace(-999, np.nan)
    nan_columns = df.columns[df_nan.isna().any()].tolist()
    df_ret = df.drop(columns=nan_columns)

    return df_ret



def dropPRI(df:pd.DataFrame):
    '''dropea todas las columnas cuyo nombre comience con PRI'''
    
    pri_cols = [col for col in df.columns if col.startswith('PRI')]
    df_ret = df.drop(columns=pri_cols)

    return df_ret


def dropEventID(df:pd.DataFrame):
    try:
        event_id = df['EventId']
        df_ret = df.drop(columns=['EventId'])
        return event_id, df_ret
    except:
        print('no habia columna EventID, revisar lo que se devolvió')
        return df
    
    
def labels_a_binario(y_train, y_val):
    if (y_train=='s').any():
        y_t = (y_train=='s')
        y_v = (y_val=='s')
    else:
        y_t = y_train
        y_v = y_val
    return y_t, y_v




def create_solution_dictionary(solution):
    """ Read solution file, return a dictionary with key EventId and value (weight,label).
    Solution file headers: EventId, Label, Weight """

    # Este proceso toma tiempo: una vez generado el diccionario se salva a disco.
    # if os.path.isfile("{}_dict.pkl".format(solution)):
    #     with open("{}_dict.pkl".format(solution), 'rb') as f:
    #         solnDict = pickle.load(f)
    #         return (solnDict)


    solnDict = {}
    df = pd.read_csv(solution)

    for i, row in df.iterrows():
        solnDict[row.EventId] = (row.Class, row.Weight)

    f = open("{}_dict.pkl".format(solution),"wb")
    pickle.dump(solnDict,f)
    f.close()

    return solnDict

        
def check_submission(submission, Nelements):
    """ Check that submission RankOrder column is correct:
        1. All numbers are in [1,NTestSet]
        2. All numbers are unqiue
    """
    df = pd.read_csv(submission)

    if len(df['EventId'].unique()) != Nelements:
        print('RankOrder column must contain unique values')
        exit()
    else:
        return True

def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
    
    br = 10.0
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return math.sqrt(radicand)


def AMS_metric(solution, submission):
    """  Prints the AMS metric value to screen.
    Solution File header: EventId, Class, Weight
    Submission File header: EventId, Class
    """

    # solutionDict: key=eventId, value=(label, class)
    solutionDict = create_solution_dictionary(solution)

    numEvents = len(solutionDict)
    print(numEvents)

    signal = 0.0
    background = 0.0
    if check_submission(submission, numEvents):
        df = pd.read_csv(submission)
        #with open(submission, 'rb') as f:
        #    sub = csv.reader(f)
        #    sub.next() # header row
        for i, row in df.iterrows():
            if row[2] == 1: # only events predicted to be signal are scored
                if solutionDict[row[0]][0] == 1:
                    signal += float(solutionDict[row[0]][1])
                elif solutionDict[row[0]][0] == 0:
                    background += float(solutionDict[row[0]][1])
     
        print('signal = {0}, background = {1}'.format(signal, background))
        ams = AMS(signal, background)
        print('AMS = ' + str(ams))

        return(ams)
    
    
def crear_solution_file(eventID, clase, pesos, nombre_archivo):
    ''' se le pasan listas o series de pandas que tengan la info correspondiente a los parametros y se crea
    un archivo que permite calcular el AMS
    Solution File header: EventId, Class, Weight'''
    eventID_list = eventID.to_list()
#     clase_list = clase.to_list()
#     pesos_list = pesos.to_list()
    df_sol = pd.DataFrame(data={
        'EventId': eventID,
        'Class': clase,
        'Weight': pesos
    })
    df_sol.to_csv(nombre_archivo, index=False)
    
    
def crear_submission_file(eventID, clase, nombre_archivo):
    ''' se le pasan listas o series de pandas que tengan la info correspondiente a los parametros y se crea
    un archivo que permite calcular el AMS
    Submission File header: EventId, Class'''
    eventID_list = eventID.to_list()
    clase_list = clase.to_list()
    p = list(range(1,len(eventID_list)+1))
    df_sol = pd.DataFrame(data={
        'EventId': eventID_list,
        'RankOrder': p,
        'Class': clase_list,
    })
    df_sol.to_csv(nombre_archivo, index=False)
