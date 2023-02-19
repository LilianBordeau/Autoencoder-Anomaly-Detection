import itertools
import numpy as np
import pandas as pd
import pickle
import teradatasql
from statsmodels.tsa.seasonal import seasonal_decompose
import statistics
import math
from pyod.models.auto_encoder import AutoEncoder
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.models import Sequential
from pandarallel import pandarallel
import tensorflow as tf
pandarallel.initialize()

# Connexion
host     = ''
user     = ''
password = ''
url      = '{"host":"'+host+'","user":"'+user+'","password":"'+password+'"}'

# Connexion
connexion = teradatasql.connect(url)
curseur   = connexion.cursor()

# Paramètres
date_min = '2021-06-01'
date_max = 'CURRENT_DATE'

# Répertoire SQL
repertoire = r'SQL/'

# Mapping data
features = ['END_SALES_CONTEXT_SALES_CH',
            'CONCAT_TYPE_ORDER',
            'DEPARTEMENT',
            'LOGISTICS_DELIVERY_MODE',
            'DEVICE_BRAND_LABEL',
            'DEPARTEMENT_LIVRAISON']
date = ['RECORDED_DATE']

# Mapping Y
price_frequencies = ['fp_'+str(i) for i in range(12)]
count_frequencies = ['fs_'+str(i) for i in range(12)]
total_frequencies = price_frequencies + count_frequencies
means_frequencies = ['mean_count', 'mean_price']
means_frequencies_n = [frequency + '_n' for frequency in means_frequencies]
count_frequencies_n = [frequency + '_n' for frequency in count_frequencies]
price_frequencies_n = [frequency + '_n' for frequency in price_frequencies]
total_frequencies_n = [frequency + '_n' for frequency in total_frequencies]

def load_timeseries(user_login, date_min, date_max):
    """ charge les séries tmeporelles
    """
    
    query = open(repertoire + 'load_timeseries.sql', 'r').read()
    query = query.replace('USER_LOGIN', user_login)
    query = query.replace('DATE_MIN', date_min)
    query = query.replace('DATE_MIN', date_max)
    
    # Chargement
    df = pd.read_sql(query, connexion)
    
    # Prétraitement
    df[features] = df[features].fillna(-1)
    df['RECORDED_DATE'] = pd.to_datetime(df.RECORDED_DATE, dayfirst=True)
    df = df.sort_values(features+date)
    df = df.reset_index(drop=True)
    
    return df

def normalize_grouped(group):
    """ Normalise les series temporelles aggrégées par variables predictives
    """
    
    means = group[total_frequencies].mean()
    stds  = group[total_frequencies].std()
    group[total_frequencies_n] = group[total_frequencies].sub(means).div(stds)
    return group

def adjust_for_global_trend(group):
    """ Ajuste les series temporelles par rapport à la tendance globale de tous les pays
    """
    
    group[means_frequencies_t] = group[means_frequencies_n].sub(means.values)
    return group

def seasonal_decomposition(group):
    """ Décomposition seasonale
    On garde les series résiduelles.
    """
    
    group[means_frequencies_r] = group[means_frequencies_n].apply(lambda x: seasonal_decompose(x, model='additive', two_sided=True, period=7).resid)
    return group

def preprocessing_timeseries(df):
    """ Wrapper pour le prétraitement des sreies temproelles.
    """
    
    # Normalisation features
    df_n = df.groupby(features, dropna=False).parallel_apply(normalize_grouped)

    # Normalisation & moyenne mobile
    df_n['mean_count'] = df_n[count_frequencies].sum(axis=1) / df_n['somme_bool']
    df_n['mean_price'] = df_n[price_frequencies].sum(axis=1) / df_n['somme_bool']
    df_n['mean_count_n'] = df_n[count_frequencies_n].sum(axis=1) / df_n['somme_bool']
    df_n['mean_price_n'] = df_n[price_frequencies_n].sum(axis=1) / df_n['somme_bool']
    means_frequencies_r = [frequency + '_r' for frequency in means_frequencies]
    df_n['mean_count_r'] = df_n['mean_count_n'].rolling(window=15, min_periods=1).mean()
    df_n['mean_price_r'] = df_n['mean_price_n'].rolling(window=15, min_periods=1).mean()

    # Ajustement tendance globale
    means_frequencies_t = [frequency + '_t' for frequency in means_frequencies]
    df_t = df_n.groupby(features, dropna=False).parallel_apply(adjust_for_global_trend)
    
    # Scaling
    means_frequencies_s = [col+'_s' for col in means_frequencies]
    mins = df_n[means_frequencies_n].min().abs()
    df_n[means_frequencies_s] = df_n[means_frequencies_n].add(mins) + 1
    
    # Décomposition
    means_frequencies_r = [frequency + '_r' for frequency in means_frequencies]
    df_r = df_n.set_index('RECORDED_DATE').copy()
    df_r = df_r.groupby(features, dropna=False).parallel_apply(seasonal_decomposition)
    
    df_r = df_r.reset_index()
    cols = date + features + means_frequencies_r 
    df_res = df_r[cols].dropna().copy()
    print(df_res.shape, df_r.shape)
    
    return df_res, df_r

def reshape_timeseries(df):
    """ Mise en forme des séries temporelles pour entrée / sortie en réseau de neurones
    """
    
    lag = 5
    forecast_size = 1

    values_X = list()
    values_Y = list()
    range_date = df.RECORDED_DATE.unique().size

    for name, group in df_f.groupby(features, dropna=False):
        for i in list(range(lag, range_date))[::lag]:
            values_X.append(group.iloc[i-lag:i][means_frequencies_r].values)
            values_Y.append(group.iloc[i:i+forecast_size][means_frequencies_r].values)

    values_X = K.stack( values_X, axis=0 )
    values_Y = K.stack( values_Y, axis=0 )
    values_Y = K.squeeze(values_Y, axis=1 )

    print(values_X.shape, values_Y.shape)

    timesteps    = values_X.shape[1]
    input_dim    = values_X.shape[2]
    latent_dim   = 256

    print(timesteps, input_dim, latent_dim)
    
    return values_X, values_Y

def load_autoencoder():
    """ Chargement de l'autoencodeur précédemment entrainéé avec autoencodeur_training.py
    """
    
    sequence_autoencoder = load_model("Models/autoencodeur.h5")
    
    return sequence_autoencoder

# Forecast Autoencodeur

def autoencoder_forecast(group, model, lag, cols_res):
    """ Prévision avec l'autoencodeur.
    """
    
    global progress
    global progress_tot
    global col_res_binary
    global col_res_scores
        
    forecast = list()
    for i in range(0,range_date-lag)[::5]:
        forecast.append(group.iloc[i:i+lag][means_frequencies_r].values)
    forecast = np.stack(forecast, axis=0)
    
    #print(forecast.shape)
    
    preds = model.predict(forecast)
    preds = preds.reshape(-1, preds.shape[-1])
    #print(preds.shape)
    lag = group.shape[0] - preds.shape[0]
    preds = np.insert(preds, -1, np.zeros((lag,len(means_frequencies_r))), axis=0)
    
    group[cols_res] = pd.DataFrame(index=group.index, columns=cols_res, data=preds)

    print(progress,'/',progress_tot, end='\r', flush=True)
    progress += 1
    return group

def evaluate_autoencoder(df_f):
    """ Evalue les performances de l'autoencodeur.
    """
    
    # Erreurs individuelles
    df_f['mse_auto'] = df_f.parallel_apply(lambda x: mean_squared_error(x[means_frequencies_r], x[cols_res]), axis=1)
    df_f['mde_auto'] = df_f.parallel_apply(lambda x: median_absolute_error(x[means_frequencies_r], x[cols_res]), axis=1)
    df_f['mae_auto'] = df_f.parallel_apply(lambda x: mean_absolute_error(x[means_frequencies_r], x[cols_res]), axis=1)
    df_f['r2s_auto'] = df_f.parallel_apply(lambda x: r2_score(x[means_frequencies_r], x[cols_res]), axis=1)
    df_f['map_auto'] = df_f.parallel_apply(lambda x: mean_absolute_percentage_error(x[means_frequencies_r], x[cols_res]), axis=1)
    
    # Erreurs moyennes
    df_f['mse_mean'] = df_f[[x for x in df_f.columns if 'mse' in x]].mean(axis=1)
    df_f['mde_mean'] = df_f[[x for x in df_f.columns if 'mde' in x]].mean(axis=1)
    df_f['mae_mean'] = df_f[[x for x in df_f.columns if 'mae' in x]].mean(axis=1)
    df_f['r2s_mean'] = df_f[[x for x in df_f.columns if 'r2s' in x]].mean(axis=1)
    df_f['map_mean'] = df_f[[x for x in df_f.columns if 'map' in x]].mean(axis=1)
    
    return df_f

def insert_results(df_f):
    """ Insert les résultats de l'autoencodeur sur DAtalab
    """
    
    questionmarks = ','.join(len(df_f.columns) * ['?'])

    query = open(repertoire + 'create_anomaly_results.sql', 'r').read()
    
    df_f['RECORDED_DATE'] = pd.to_datetime(df_f['RECORDED_DATE'].astype(str))
    df_f[features] = df_f[features].replace(np.nan,'')
    
    values = df_f.values.tolist()

    insert = """{fn teradata_require_fastload}
                INSERT INTO DB_DATALAB_DAF.ML_ANOMALY_RESULTS ("""+questionmarks+""")"""

    with connexion.cursor() as cur:
            cur.execute(query)
            connexion.commit()
            cur.execute("{fn teradata_nativesql}{fn teradata_autocommit_off}")
            connexion.commit()
            cur.execute(insert, values)
            connexion.commit()

if __name__ == "__main__":
    
    df = load_timeseries(user_login, date_min, date_max)
    df_res, df_r = preprocessing_timeseries(df)
    df_f = df_r.fillna(0).copy()
    values_X, values_Y = reshape_timeseries(df_f)
    
    sequence_autoencoder = load_autoencoder()
    
    progress, progress_tot = 0, df_f.groupby(features).ngroups
    cols_res = [col+'_f_ae' for col in means_frequencies]
    df_f = df_f.groupby(features, dropna=False).apply(autoencoder_forecast, sequence_autoencoder, lag, cols_res)
    
    df_f = evaluate_autoencoder(df_f)
    insert_results(df_f)