import teradatasql
import itertools
import numpy as np
import pandas as pd
import pickle

# Connexion Teradata
host     = '10.43.67.32'
user     = 'u165983'
password = '7BB43ryXd6'
url      = '{"host":"'+host+'","user":"'+user+'","password":"'+password+'"}'

# Connexion
connexion = teradatasql.connect(url)
curseur   = connexion.cursor()

# Paramètres
taille_permutation = 2
seuil_cardinalite = 1000

# Variables à permuter et correspondance dans les donneés
features = {
        "CONCAT_TYPE_ORDER": "CONCAT_TYPE_ORDER",
        "CODE_BANQUE": "SUBSTR(IBAN, 5 , 5)",
        "END_SALES_CONTEXT_SALES_CH": "END_SALES_CONTEXT_SALES_CH",
#        "CODE_BANQUE_GUICHET": "SUBSTR(IBAN, 5 , 10)",
#        "CODE_POSTAL": "LPAD(TRIM(TITULAIRE_ZIP_CODE) ,5,'0')",
        "DEPARTEMENT":  "CAST(LPAD(TRIM(TITULAIRE_ZIP_CODE) ,5,'0') AS CHAR(2))",
        "DOMAINE_EMAIL": "UPPER(STRTOK(TITULAIRE_EMAIL, '@', 2))",
#        "END_SALES_CONTEXT_SALES_CH": "END_SALES_CONTEXT_SALES_CH",
#        "CONCAT_TYPE_ORDER": "CONCAT_TYPE_ORDER",
        "LOGISTICS_DELIVERY_MODE": "LOGISTICS_DELIVERY_MODE",
        "DELIVERY_POINT_ID": "DELIVERY_POINT_ID",
        "CODE_POINT_VENTE": "CODE_POINT_VENTE",
        "DEVICE_BRAND_LABEL": "DEVICE_BRAND_LABEL",
#        "CODE_POSTAL_LIVRAISON": "LPAD(TRIM(ZIPCODE) ,5,'0')",
        "DEPARTEMENT_LIVRAISON":  "CAST(LPAD(TRIM(ZIPCODE) ,5,'0') AS CHAR(2))"
     }

def permute_variables(variables, taille_permutation):
    """ Renvoie une liste des permutations pour une taille donnée.
    """
    
    permutations = list(itertools.combinations(features, taille_permutation))
    print('Permutation:', len(permutations), 'permutations')
    return permutations

def create_permutation_table():
    """ Créer la table pour stocker les permutations.
    """
    
    query = """
            CREATE SET TABLE DB_DATALAB_DAF.ML_ANOMALY_PERMUTATIONS 
            (
                   COL_1 VARCHAR(255) CHARACTER SET LATIN NOT CASESPECIFIC,
                   COL_2 VARCHAR(255) CHARACTER SET LATIN NOT CASESPECIFIC,
                   CARDINALITE INT
            ) UNIQUE PRIMARY INDEX(COL_1, COL_2); 
            """

    comment = """COMMENT ON TABLE DB_DATALAB_DAF.ML_ANOMALY_PERMUTATIONS 
    AS 'Table des permutations de variables avec leur cardinalité';"""

    with connexion.cursor() as cur:
        cur.execute(query)
        connexion.commit()
        cur.execute(comment)
        connexion.commit()

def insert_permutations_into_table():
    """ Insert les permutations avec leur cardinalité sur la table précédemment créée.
    """

    for i, permutation in enumerate(permutations):

        print(permutation[0],permutation[1])

        query = """INSERT INTO DB_DATALAB_DAF.ML_ANOMALY_PERMUTATIONS 
                   WITH R1 AS (
                   SELECT DISTINCT
                         """+features[permutation[0]]+""" AS F1,
                         """+features[permutation[1]]+""" AS F2
                   FROM DB_DATALAB_DAF.ML_SCORING_DATA)
                   SELECT '"""+permutation[0]+"""' AS COL_1, 
                          '"""+permutation[1]+"""' AS COL_2, 
                          COUNT(*) AS CARDINALITE 
                   FROM R1
                """
        
        with connexion.cursor() as cur:
            cur.execute(query)
            connexion.commit()

# Filtre les permutations au dela d'une certaine cardinalité

def filter_out_high_cardinalities():
    """ Filtre les permutations avec une trop grnade cardinalité.
    """
    
    perm_min_df = pd.read_sql("""SELECT * 
                                FROM DB_DATALAB_DAF.ML_ANOMALY_PERMUTATIONS
                                WHERE CARDINALITE < """+str(seuil_cardinalite)+"""
                                ORDER BY ID""", connexion)

    permutations_min = perm_min_df[['COL_1', 'COL_2']].values.tolist()
    features_min = list(set([item for sublist in permutations_min for item in sublist]))
    print(len(permutations_min),'/',len(permutations),'permutations')
    print(len(features_min),'/',len(features),'features')
    print('features éliminées:', set(features) - set(features_min))
    
    return perm_min_df, permutations_min, features_min

def create_timeseries_data_table():
    """ Créer la table pour le stockage des séries temperlllels.
    """
    
    sql_select_features = [features[feature] + ' AS ' + feature for feature in features_min]

    query = """
            CREATE MULTISET TABLE DB_DATALAB_DAF.ML_ANOMALY_FEATURES AS
            (
                SELECT DISTINCT """+','.join(sql_select_features)+"""
                FROM DB_DATALAB_DAF.ML_SCORING_DATA
            ) WITH DATA NO PRIMARY INDEX; 
            """

    comment = """COMMENT ON TABLE DB_DATALAB_DAF.ML_ANOMALY_FEATURES AS 'Table des variables retenues pour la création des séries temporelles';"""

    with connexion.cursor() as cur:
        cur.execute(query)
        connexion.commit()
        cur.execute(comment)
        connexion.commit()

def compute_timesteps_percent_for_each_timeseries():
    """ Calcule le pourcentage de pas de temps présent dans chaque permutation pour chaque serie temporelle.
    """
    
    for i, permutation in enumerate(permutations_min):
        print(i, permutation)

        query_alter = """
        ALTER TABLE DB_DATALAB_DAF.ML_ANOMALY_FEATURES
                    ADD st_"""+str(i)+""" FLOAT DEFAULT 0;
        """

        query_count = """
        SELECT """+features[permutation[0]]+""" AS """+permutation[0]+""",
               """+features[permutation[1]]+""" AS """+permutation[1]+""",
               Count(DISTINCT Cast(RECORDED_DATE AS DATE)) AS COUNT_TOT

        FROM DB_DATALAB_DAF.ML_SCORING_DATA
        GROUP BY 1, 2
        """

        #print(query)

        query_updat = """
        UPDATE S1
        FROM DB_DATALAB_DAF.ML_ANOMALY_FEATURES S1, ("""+query_count+""") S2
        SET st_"""+str(i)+""" = S2.COUNT_TOT
        WHERE S1."""+permutation[0]+""" = S2."""+permutation[0]+"""
          AND S1."""+permutation[1]+""" = S2."""+permutation[1]+""";
        """

        with connexion.cursor() as cur:
            cur.execute(query_alter)
            connexion.commit()
            cur.execute(query_count)
            connexion.commit()
            cur.execute(query_updat)
            connexion.commit()
            
        sql_total_steps = ['st_'+str(i) for i,p in enumerate(permutations_min)]
  
    query = """
        CREATE MULTISET TABLE DB_DATALAB_DAF.ML_ANOMALY_FEATURES_2 AS
        (
    SELECT T0.*,
    (CASE WHEN st_0 > 0 THEN 1 ELSE 0 END) AS bst_0,
    (CASE WHEN st_1 > 0 THEN 1 ELSE 0 END) AS bst_1,
    (CASE WHEN st_2 > 0 THEN 1 ELSE 0 END) AS bst_2,
    (CASE WHEN st_3 > 0 THEN 1 ELSE 0 END) AS bst_3,
    (CASE WHEN st_4 > 0 THEN 1 ELSE 0 END) AS bst_4,
    (CASE WHEN st_5 > 0 THEN 1 ELSE 0 END) AS bst_5,
    (CASE WHEN st_6 > 0 THEN 1 ELSE 0 END) AS bst_6,
    (CASE WHEN st_7 > 0 THEN 1 ELSE 0 END) AS bst_7,
    (CASE WHEN st_8 > 0 THEN 1 ELSE 0 END) AS bst_8,
    (CASE WHEN st_9 > 0 THEN 1 ELSE 0 END) AS bst_9,
    (CASE WHEN st_10 > 0 THEN 1 ELSE 0 END) AS bst_10,
    (CASE WHEN st_11 > 0 THEN 1 ELSE 0 END) AS bst_11,
    (bst_0 + bst_1 + bst_2 + bst_3 + bst_4 + bst_5 + bst_6
    + bst_7 + bst_8 + bst_9 + bst_10 + bst_11) AS somme_bool,
    (st_0 + st_1 + st_2 + st_3 + st_4 + st_5 + st_6
    + st_7 + st_8 + st_9 + st_10 + st_11) AS somme_tot,
    (somme_tot / somme_bool) AS avg_steps,
    (avg_steps / 367) AS avg_steps_percent

    FROM DB_DATALAB_DAF.ML_ANOMALY_FEATURES T0
            ) WITH DATA NO PRIMARY INDEX; 
        """

    query2 = """DROP TABLE DB_DATALAB_DAF.ML_ANOMALY_FEATURES;"""
    query3 = """RENAME TABLE DB_DATALAB_DAF.ML_ANOMALY_FEATURES_2 TO DB_DATALAB_DAF.ML_ANOMALY_FEATURES;"""

    with connexion.cursor() as cur:
        cur.execute(query)
        connexion.commit()
        cur.execute(query2)
        connexion.commit()
        cur.execute(query3)
        connexion.commit()

def plot_timesteps_distribution():
    """ Plot distribution % timesteps.
    """
    
    df_steps = pd.read_sql("SELECT avg_steps_percent FROM DB_DATALAB_DAF.ML_ANOMALY_FEATURES", connexion)
    df_steps.avg_steps_percent.hist(bins=100)

def add_id_to_permutation():
    """ Ajoute un id aux permutations?
    """
    
    query1 = """CREATE SET TABLE DB_DATALAB_DAF.ML_ANOMALY_PERMUTATIONS_2 AS
                (SELECT Row_Number() Over (ORDER BY CARDINALITE ASC) AS ID, T1.*
                FROM DB_DATALAB_DAF.ML_ANOMALY_PERMUTATIONS T1)
                WITH DATA UNIQUE PRIMARY INDEX(COL_1, COL_2);"""

    query2 = """DROP TABLE DB_DATALAB_DAF.ML_ANOMALY_PERMUTATIONS;"""

    query3 = """RENAME TABLE DB_DATALAB_DAF.ML_ANOMALY_PERMUTATIONS_2 TO DB_DATALAB_DAF.ML_ANOMALY_PERMUTATIONS;"""

    with connexion.cursor() as cur:
        cur.execute(query1)
        connexion.commit()
        cur.execute(query2)
        connexion.commit()
        cur.execute(query3)
        connexion.commit()

def create_table_permutation_stats():
    """ Table pour avoir les valeurs de Y pour chaque permutation au global (sans dates)
    """

    query = """CREATE MULTISET TABLE DB_DATALAB_DAF.ML_ANOMALIES_PERMUTATIONS_STATS (
                      ID_PERM INT,
                      VAL_1 VARCHAR(255) CHARACTER SET LATIN NOT CASESPECIFIC,
                      VAL_2 VARCHAR(255) CHARACTER SET LATIN NOT CASESPECIFIC,
                      COUNT_TOT INTEGER,
                      PRICE_SUM DECIMAL(38,0),
                      PRICE_AVG FLOAT,
                      PRICE_STD FLOAT,
                      PRICE_RSD FLOAT,
                      PRICE_MIN DECIMAL(38,0),
                      PRICE_MAX DECIMAL(38,0),
                      FRISK_TOT_SUM FLOAT,
                      FRISK_TOT_NEG FLOAT,
                      FRISK_TOT_POS FLOAT,
                      FRISK_TOT_AVG FLOAT,
                      FRISK_AVG_NEG FLOAT,
                      FRISK_AVG_POS FLOAT)
    UNIQUE PRIMARY INDEX (ID_PERM, VAL_1, VAL_2);"""

    with connexion.cursor() as cur:
        cur.execute(query)
        connexion.commit()

def populate_table_permutation_stats():
    """ Populate ML_ANOMALIES_PERMUTATIONS_STATS
    """
    
    for i, permutation in perm_min_df.iterrows():

        print(permutation['COL_1'], permutation['COL_2'])

        query = """
        INSERT INTO DB_DATALAB_DAF.ML_ANOMALIES_PERMUTATIONS_STATS
        SELECT """+str(permutation['ID'])+""" AS ID_PERM,
               """+features[permutation['COL_1']]+""" AS VAL_1,
               """+features[permutation['COL_2']]+""" AS VAL_2,
               Count(*) AS COUNT_TOT,
               Sum(TOTAL_PRICE_TTC) AS PRICE_SUM,
               Avg(TOTAL_PRICE_TTC) AS PRICE_AVG,
               StdDev_Pop(TOTAL_PRICE_TTC) AS PRICE_STD,
               (PRICE_STD / NullIfZero(PRICE_AVG)) AS PRICE_RSD,
               Min(TOTAL_PRICE_TTC) AS PRICE_MIN,
               Max(TOTAL_PRICE_TTC) AS PRICE_MAX,
               Sum(MTT_RK_12) AS FRISK_TOT_SUM,
               Sum(CASE WHEN MTT_RK_12 >= 0 THEN 0 ELSE mtt_rk_12 END) AS FRISK_TOT_NEG,
               Sum(CASE WHEN MTT_RK_12 >= 0 THEN mtt_rk_12 ELSE 0 END) AS FRISK_TOT_POS,
               Avg(MTT_RK_12) AS FRISK_TOT_AVG,
               Avg(CASE WHEN MTT_RK_12 >= 0 THEN 0 ELSE mtt_rk_12 END) AS FRISK_AVG_NEG,
               Avg(CASE WHEN MTT_RK_12 >= 0 THEN mtt_rk_12 ELSE 0 END) AS FRISK_AVG_POS

        FROM DB_DATALAB_DAF.ML_SCORING_DATA T1

        LEFT JOIN DB_DATALAB_DAF.ML_SCORING_FINANCIAL_RISK T2
        ON T1.IDNT_COMP_FACT = T2.IDNT_COMP_FACT

        GROUP BY 1, 2, 3"""

        with connexion.cursor() as cur:
            cur.execute(query)
            connexion.commit()

def create_table_permutation_stats_with_dates():
    """ Table pour avoir les valeurs de Y pour chaque permutation au global (avec dates)
    """
    
    query = """CREATE MULTISET TABLE DB_DATALAB_DAF.ML_ANOMALIES_PERMUTATIONS_STATS_DATES (
                      ID_PERM INT,
                      VAL_1 VARCHAR(255) CHARACTER SET LATIN NOT CASESPECIFIC,
                      VAL_2 VARCHAR(255) CHARACTER SET LATIN NOT CASESPECIFIC,
                      RECORDED_DATE DATE FORMAT 'YY/MM/DD',
                      COUNT_TOT INTEGER,
                      PRICE_SUM DECIMAL(38,0),
                      PRICE_AVG FLOAT,
                      PRICE_STD FLOAT,
                      PRICE_RSD FLOAT,
                      PRICE_MIN DECIMAL(38,0),
                      PRICE_MAX DECIMAL(38,0),
                      FRISK_TOT_SUM FLOAT,
                      FRISK_TOT_NEG FLOAT,
                      FRISK_TOT_POS FLOAT,
                      FRISK_TOT_AVG FLOAT,
                      FRISK_AVG_NEG FLOAT,
                      FRISK_AVG_POS FLOAT)
    UNIQUE PRIMARY INDEX (ID_PERM, VAL_1, VAL_2, RECORDED_DATE);"""

    with connexion.cursor() as cur:
        cur.execute(query)
        connexion.commit()

def populate_table_permutation_stats_with_dates():
    """ Populate ML_ANOMALIES_PERMUTATIONS_STATS_DATES
    """
    
    for i, permutation in perm_min_df.iterrows():

        print(permutation['COL_1'], permutation['COL_2'])

        query = """
        INSERT INTO DB_DATALAB_DAF.ML_ANOMALIES_PERMUTATIONS_STATS_DATES
        SELECT """+str(permutation['ID'])+""" AS ID_PERM,
               """+features[permutation['COL_1']]+""" AS VAL_1,
               """+features[permutation['COL_2']]+""" AS VAL_2,
               CAST(RECORDED_DATE AS DATE) AS RECORDED_DATE,
               Count(*) AS COUNT_TOT,
               Sum(TOTAL_PRICE_TTC) AS PRICE_SUM,
               Avg(TOTAL_PRICE_TTC) AS PRICE_AVG,
               StdDev_Pop(TOTAL_PRICE_TTC) AS PRICE_STD,
               (PRICE_STD / NullIfZero(PRICE_AVG)) AS PRICE_RSD,
               Min(TOTAL_PRICE_TTC) AS PRICE_MIN,
               Max(TOTAL_PRICE_TTC) AS PRICE_MAX,
               Sum(MTT_RK_12) AS FRISK_TOT_SUM,
               Sum(CASE WHEN MTT_RK_12 >= 0 THEN 0 ELSE mtt_rk_12 END) AS FRISK_TOT_NEG,
               Sum(CASE WHEN MTT_RK_12 >= 0 THEN mtt_rk_12 ELSE 0 END) AS FRISK_TOT_POS,
               Avg(MTT_RK_12) AS FRISK_TOT_AVG,
               Avg(CASE WHEN MTT_RK_12 >= 0 THEN 0 ELSE mtt_rk_12 END) AS FRISK_AVG_NEG,
               Avg(CASE WHEN MTT_RK_12 >= 0 THEN mtt_rk_12 ELSE 0 END) AS FRISK_AVG_POS

        FROM DB_DATALAB_DAF.ML_SCORING_DATA T1

        LEFT JOIN DB_DATALAB_DAF.ML_SCORING_FINANCIAL_RISK T2
        ON T1.IDNT_COMP_FACT = T2.IDNT_COMP_FACT

        GROUP BY 1, 2, 3, 4"""

        with connexion.cursor() as cur:
            cur.execute(query)
            connexion.commit()

def create_timeseries_data_table_with_dates():
    """ Créer la table pour le stockage des séries temperells. (avec dates)
    """
    
    sql_select_features = [features[feature] + ' AS ' + feature for feature in features_min]

    query = """
            CREATE MULTISET TABLE DB_DATALAB_DAF.ML_ANOMALY_FEATURES_WITH_DATES AS
            (
                SELECT DISTINCT CAST(RECORDED_DATE AS DATE) AS RECORDED_DATE,
                        """+','.join(sql_select_features)+"""
                FROM DB_DATALAB_DAF.ML_SCORING_DATA
            ) WITH DATA UNIQUE PRIMARY INDEX(RECORDED_DATE,"""+','.join(features_min)+"""); 
            """

    comment = """COMMENT ON TABLE DB_DATALAB_DAF.ML_ANOMALY_FEATURES_WITH_DATES AS 'Table des variables retenues pour la création des séries temporelles avec les dates';"""

    with connexion.cursor() as cur:
        cur.execute(query)
        connexion.commit()
        cur.execute(comment)
        connexion.commit()

def update_timeseries_data_table_with_dates():
    """ Ajoute les valeurs des séries temporelles (mean,max,avg,etc sur le prix et nombre (count)) pour chaque pas de temps
        Et le risque qui représente le degré d'anomalie en supervisé
        Attention : l'écart type est djéà normalisé par l amoyenne
    """
    
    for i, permutation in perm_min_df.iterrows():

        print(permutation['COL_1'], permutation['COL_2'])

        query_alter = """
        ALTER TABLE DB_DATALAB_DAF.ML_ANOMALY_FEATURES_WITH_DATES
                    ADD p"""+str(permutation['ID'])+"""_count_tot FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_sum FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_avg FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_std FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_rsd FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_min FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_max FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_tot_sum FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_tot_neg FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_tot_pos FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_tot_avg FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_avg_neg FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_avg_pos FLOAT DEFAULT 0;
        """

        query_count = """
        SELECT *    
        FROM DB_DATALAB_DAF.ML_ANOMALIES_PERMUTATIONS_STATS_DATES
        WHERE ID_PERM = """+str(permutation['ID'])+"""
        """

        query_updat = """
        UPDATE S1
        FROM DB_DATALAB_DAF.ML_ANOMALY_FEATURES_WITH_DATES S1, ("""+query_count+""") S2
        SET p"""+str(permutation['ID'])+"""_count_tot = S2.count_tot,
            p"""+str(permutation['ID'])+"""_price_sum = S2.price_sum,
            p"""+str(permutation['ID'])+"""_price_avg = S2.price_avg,
            p"""+str(permutation['ID'])+"""_price_std = S2.price_std,
            p"""+str(permutation['ID'])+"""_price_rsd = S2.price_rsd,
            p"""+str(permutation['ID'])+"""_price_min = S2.price_min,
            p"""+str(permutation['ID'])+"""_price_max = S2.price_max,
            p"""+str(permutation['ID'])+"""_frisk_tot_sum = S2.frisk_tot_sum,
            p"""+str(permutation['ID'])+"""_frisk_tot_neg = S2.frisk_tot_neg,
            p"""+str(permutation['ID'])+"""_frisk_tot_pos = S2.frisk_tot_pos,
            p"""+str(permutation['ID'])+"""_frisk_tot_avg = S2.frisk_tot_avg,
            p"""+str(permutation['ID'])+"""_frisk_avg_neg = S2.frisk_avg_neg,
            p"""+str(permutation['ID'])+"""_frisk_avg_pos = S2.frisk_avg_pos
        WHERE S1."""+permutation['COL_1']+""" = S2.VAL_1
          AND S1."""+permutation['COL_2']+""" = S2.VAL_2
          AND S1.RECORDED_DATE          = S2.RECORDED_DATE ;
        """

        with connexion.cursor() as cur:
            cur.execute(query_alter)
            connexion.commit()
            cur.execute(query_count)
            connexion.commit()
            cur.execute(query_updat)
            connexion.commit()

def update_timeseries_data_table():
    """ Ajoute les valeurs des séries temporelles (mean,max,avg,etc sur le prix et nombre (count))
        Et le risque qui représente le degré d'anomalie en supervisé
        Attention : l'écart type est djéà normalisé par l amoyenne
    """
    
    for i, permutation in perm_min_df.iterrows():

        print(permutation['COL_1'], permutation['COL_2'])

        query_alter = """
        ALTER TABLE DB_DATALAB_DAF.ML_ANOMALY_FEATURES
                    ADD p"""+str(permutation['ID'])+"""_count_tot FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_sum FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_avg FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_std FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_rsd FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_min FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_price_max FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_tot_sum FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_tot_neg FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_tot_pos FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_tot_avg FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_avg_neg FLOAT DEFAULT 0,
                    ADD p"""+str(permutation['ID'])+"""_frisk_avg_pos FLOAT DEFAULT 0;
        """

        query_count = """
        SELECT *    
        FROM DB_DATALAB_DAF.ML_ANOMALIES_PERMUTATIONS_STATS
        WHERE ID_PERM = """+str(permutation['ID'])+"""
        """

        query_updat = """
        UPDATE S1
        FROM DB_DATALAB_DAF.ML_ANOMALY_FEATURES S1, ("""+query_count+""") S2
        SET p"""+str(permutation['ID'])+"""_count_tot = S2.count_tot,
            p"""+str(permutation['ID'])+"""_price_sum = S2.price_sum,
            p"""+str(permutation['ID'])+"""_price_avg = S2.price_avg,
            p"""+str(permutation['ID'])+"""_price_std = S2.price_std,
            p"""+str(permutation['ID'])+"""_price_rsd = S2.price_rsd,
            p"""+str(permutation['ID'])+"""_price_min = S2.price_min,
            p"""+str(permutation['ID'])+"""_price_max = S2.price_max,
            p"""+str(permutation['ID'])+"""_frisk_tot_sum = S2.frisk_tot_sum,
            p"""+str(permutation['ID'])+"""_frisk_tot_neg = S2.frisk_tot_neg,
            p"""+str(permutation['ID'])+"""_frisk_tot_pos = S2.frisk_tot_pos,
            p"""+str(permutation['ID'])+"""_frisk_tot_avg = S2.frisk_tot_avg,
            p"""+str(permutation['ID'])+"""_frisk_avg_neg = S2.frisk_avg_neg,
            p"""+str(permutation['ID'])+"""_frisk_avg_pos = S2.frisk_avg_pos
        WHERE S1."""+permutation['COL_1']+""" = S2.VAL_1
          AND S1."""+permutation['COL_2']+""" = S2.VAL_2 ;
        """

        with connexion.cursor() as cur:
            cur.execute(query_alter)
            connexion.commit()
            cur.execute(query_count)
            connexion.commit()
            cur.execute(query_updat)
            connexion.commit()

def normalise_timeseries():
    """ Normalise les serires temporelels
    """

    sql_norm_p_sum = ['(p_sum_'+str(i)+' / NULLIFZERO(PRICE_SUM)) AS p_sum_'+str(i) for i,p in enumerate(permutations_min)]
    sql_norm_p_avg = ['(p_avg_'+str(i)+' / NULLIFZERO(PRICE_AVG)) AS p_avg_'+str(i) for i,p in enumerate(permutations_min)]
    #sql_norm_p_min = ['(p_min_'+str(i)+' / NULLIFZERO(PRICE_MIN)) AS p_min_'+str(i) for i,p in enumerate(permutations_min)]
    sql_norm_p_max = ['(p_max_'+str(i)+' / NULLIFZERO(PRICE_MAX)) AS p_max_'+str(i) for i,p in enumerate(permutations_min)]
    sql_norm_c_tot = ['(c_tot_'+str(i)+' / NULLIFZERO(COUNT_TOT)) AS c_tot_'+str(i) for i,p in enumerate(permutations_min)]

    query_norm = """
                 SELECT Cast(RECORDED_DATE AS DATE) AS RECORDED_DATE,
                        Cast(SUM(total_price_ttc) AS FLOAT) AS PRICE_SUM,
                        Cast(AVG(total_price_ttc) AS FLOAT) AS PRICE_AVG,
                        Cast(MIN(total_price_ttc) AS FLOAT) AS PRICE_MIN,
                        Cast(MAX(total_price_ttc) AS FLOAT) AS PRICE_MAX,
                        Cast(COUNT(*) AS FLOAT) AS COUNT_TOT
                 FROM DB_DATALAB_DAF.ML_SCORING_DATA
                 GROUP BY 1
                 """

    query = """
            CREATE MULTISET TABLE DB_DATALAB_DAF.ML_ANOMALY_PERMUTED_FEATURES_WITH_STEPS_NORM AS
            (
                SELECT T1.RECORDED_DATE,
                       """+','.join(features_min)+""",
                       """+',\n'.join(sql_norm_p_sum)+""",
                       """+',\n'.join(sql_norm_p_avg)+""",
                       """+',\n'.join(sql_norm_p_max)+""",
                       """+',\n'.join(sql_norm_c_tot)+"""

                FROM DB_DATALAB_DAF.ML_ANOMALY_PERMUTED_FEATURES_WITH_STEPS T1
                LEFT JOIN ("""+query_norm+""") T2
                ON T1.RECORDED_DATE = T2.RECORDED_DATE

            ) WITH DATA PRIMARY INDEX (RECORDED_DATE); 
            """

    comment = """COMMENT ON TABLE DB_DATALAB_DAF.ML_ANOMALY_PERMUTED_FEATURES_WITH_STEPS_NORM AS 'Table des permutations de variables pour la création des séries temporelles avec valeurs normalisée';"""

    #print(query)
    with connexion.cursor() as cur:
        cur.execute(query)
        connexion.commit()
        cur.execute(comment)
        connexion.commit()