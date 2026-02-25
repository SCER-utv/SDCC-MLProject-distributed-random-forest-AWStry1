import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
### [INIZIO MODIFICA AWS] Importazione librerie AWS e parsing URL ###
import boto3
from urllib.parse import urlparse
### [FINE MODIFICA AWS] ###
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.factories.ids_task_factory import IDSTaskFactory
from src.core.factories.taxi_task_factory import TaxiTaskFactory
import threading

class RandomForestManager:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.dataset_cache = {}
        # --- NUOVO: Lock per evitare Race Condition sui download ---
        self.download_lock = threading.Lock()

    # Helper per ottenere la strategia corretta dal bit gRPC"""
    def _get_ml_components(self, task_type):
       
        factory = TaxiTaskFactory() if task_type == 1 else IDSTaskFactory()
        return factory.create_ml_strategy()
    
    ### [INIZIO MODIFICA AWS] Funzione per estrarre il nome del bucket dinamicamente ###
    def _extract_bucket_from_s3_path(self, s3_path):

        # Estrae il nome del bucket senza hard-coding da 's3://mio-bucket/cartella/file.csv'
        parsed_url = urlparse(s3_path)
        if parsed_url.scheme != "s3":
            raise ValueError(f"Il percorso non è un URI S3 valido: {s3_path}")
        return parsed_url.netloc
    ### [FINE MODIFICA AWS] ###

    def train(self, model_id, subforest_id, dataset_path, seed, n_estimators, task_type, max_depth, max_features, criterion, skip_rows, num_rows):
        # 1. Componenti Polimorfici
        ml_strategy = self._get_ml_components(task_type)

        # [MODIFICA ETEROGENEA] 1. Parsing dei parametri
        # gRPC manda 0 per indicare "default/infinito" (int non può essere None)
        final_depth = None if max_depth == 0 else max_depth
        
        # gRPC manda stringhe, dobbiamo convertire "0.5" in float 0.5
        try:
            final_features = float(max_features)
        except ValueError:
            # Se fallisce (es. è "sqrt" o "log2"), teniamo la stringa
            final_features = max_features

        print(f" -> [Model] Training Config: Depth={final_depth}, Feat={final_features}, Crit={criterion}")

        ### [INIZIO MODIFICA AWS] Uso dell'URI S3 e estrazione Bucket ###
        full_path = dataset_path # Ad es: "s3://distributed-random-forest-bkt/shards/taxi/train_part_1.csv"
        try:
            target_bucket = self._extract_bucket_from_s3_path(full_path)
        except ValueError as e:
            print(f"!!! [ERRORE S3]: {e}")
            return 0
        # [FINE MODIFICA AWS] ###

        cache_key = f"{full_path}_{skip_rows}_{num_rows}"        
        
        # [MODIFICA ZERO-COPY] Cache key basata sull'offset, non solo sul path
        if cache_key in self.dataset_cache:
            X, y = self.dataset_cache[cache_key]
        else:
            # --- MODIFICA ZERO-COPY VELOCE (DOWNLOAD LOCALE + PARSING) ---
            print(f" -> [Worker] Download rapido S3 -> Disco locale...")
            
            # Estraiamo la chiave corretta per Boto3 usando urlparse
            parsed_url = urlparse(full_path)
            s3_key = parsed_url.path.lstrip('/')
            
            # Creiamo un file temporaneo locale univoco per questo worker
            local_temp_csv = f"/tmp/temp_dataset_{subforest_id}.csv"
            
            start_dl = time.time()
            s3_client = boto3.client('s3')
            # Questo download è multi-thread e sfrutta tutta la banda EC2!
            s3_client.download_file(target_bucket, s3_key, local_temp_csv)
            print(f"    [OK] Download 100% completato in {time.time() - start_dl:.1f}s")
            
            print(f" -> [Worker] Lettura da disco NVMe locale: Salto {skip_rows} righe, leggo {num_rows} righe...")
            start_parse = time.time()
            
            rows_to_skip = 0 if skip_rows == 0 else range(1, skip_rows + 1)
            
            # Ora Pandas legge dal disco locale: è incredibilmente più veloce!
            df = pd.read_csv(local_temp_csv, engine='c', dtype=np.float32, skiprows=rows_to_skip, nrows=num_rows)
            print(f" [OK] Parsing CSV locale completato in {time.time() - start_parse:.1f}s")
            
            # Pulizia immediata del disco per risparmiare spazio
            os.remove(local_temp_csv)
            # -------------------------------------------------------------
            
            target_col = 'Label' if 'Label' in df.columns else df.columns[-1]

            if task_type != 1: 
                 df[target_col] = df[target_col].astype(np.int8)

            X = df.drop(target_col, axis=1).values 
            y = ml_strategy.cast_target(df[target_col].values) 
            
            self.dataset_cache[cache_key] = (X, y)

        # [MODIFICA ETEROGENEA] 2. Passaggio Parametri alla Strategia
        # ATTENZIONE: Qui avevi 'max_depth=None' hardcoded! Ora usiamo le variabili.
        clf = ml_strategy.create_model(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            
            # USIAMO I PARAMETRI PARSATI
            max_depth=final_depth,      
            max_features=final_features,
            criterion=criterion,
            
            verbose=0
        )

        # Controllo per valori infiniti
        if not np.isfinite(X).all():
            print(f"!!! [DEBUG DATA ERROR] Valori non finiti trovati nello shard {subforest_id}")

            # Individua le colonne colpevoli
            cols_with_inf = np.where(np.isinf(X).any(axis=0))[0]
            print(f" -> Colonne con 'inf': {cols_with_inf}")

            # Individua righe con valori troppo grandi per float32
            max_float32 = np.finfo(np.float32).max
            overflows = np.where(np.abs(X) > max_float32)
            print(f" -> Indici di overflow: {overflows}")

        start_time = time.time()
        clf.fit(X, y)
        print(f"   -> [{subforest_id}] Training completato in {time.time() - start_time:.2f}s")

        ### [INIZIO MODIFICA AWS] Salvataggio Modello su S3 via Boto3 ###
        filename = f"{model_id}_{subforest_id}.joblib"
        
        ### [MODIFICA S3 PATH] Salvataggio Modello ordinato in S3 ###
        filename = f"{model_id}_{subforest_id}.joblib"
        os.makedirs(self.models_dir, exist_ok=True)
        local_path = os.path.join(self.models_dir, filename)
        joblib.dump(clf, local_path)

        s3_client = boto3.client('s3')
        dataset_folder = "taxi" if task_type == 1 else "higgs"
        
        # ORA CREIAMO LA SOTTOCARTELLA CON IL MODEL_ID!
        s3_key = f"models/{dataset_folder}/{model_id}/{filename}"
        
        print(f"-> Upload modello su s3://{target_bucket}/{s3_key} in corso...")
        s3_client.upload_file(local_path, target_bucket, s3_key)
        os.remove(local_path)
        ### [FINE MODIFICA S3 PATH] ###

        self.loaded_models[f"{model_id}_{subforest_id}"] = clf

        return n_estimators

    def predict_batch(self, model_id, subforest_id, flat_features, task_type):

        ml_strategy = self._get_ml_components(task_type)
        model_key = f"{model_id}_{subforest_id}"

        ### [INIZIO MODIFICA AWS] Recupero dinamico del bucket per l'Inferenza ###
        # Poiché predict_batch non riceve il path del dataset, peschiamo il bucket dalle variabili d'ambiente 
        # (Standard per il Cloud). Assicurati di lanciare: export AWS_S3_BUCKET=distributed-random-forest-bkt
        target_bucket = os.environ.get("AWS_S3_BUCKET")
        if not target_bucket:
            print("!!! [ERRORE S3]: Variabile d'ambiente AWS_S3_BUCKET non impostata sul Worker!")
            return []
        ### [FINE MODIFICA AWS] ###

        print(f"\n[DEBUG] Richiesta inferenza per {model_key}")

        ### [INIZIO MODIFICA AWS] Lazy Loading del modello da S3 ###
        if model_key not in self.loaded_models:
            
            # --- AGGIUNTA DEL LOCK ---
            with self.download_lock:
                # Doppia verifica (se un altro thread ha scaricato mentre aspettavamo)
                if model_key not in self.loaded_models:
            # -------------------------
            
                    filename = f"{model_key}.joblib"
                    dataset_folder = "taxi" if task_type == 1 else "higgs"
                    s3_key = f"models/{dataset_folder}/{model_id}/{filename}"
                    
                    # --- FIX PERMESSI: Usiamo self.models_dir invece di /tmp ---
                    os.makedirs(self.models_dir, exist_ok=True)
                    local_path = os.path.join(self.models_dir, filename)
                    
                    s3_client = boto3.client('s3')
                    print(f" -> [Worker] Download modello da s3://{target_bucket}/{s3_key} ...")
                    try:
                        # Scarica direttamente nella cartella del progetto
                        s3_client.download_file(target_bucket, s3_key, local_path)
                        
                        # Caricalo in RAM
                        self.loaded_models[model_key] = joblib.load(local_path)
                        
                        # Pulisce il file dal disco
                        os.remove(local_path)
                    except Exception as e:
                        print(f"!!! Errore: Download Modello {filename} da S3 fallito: {e}")
                        return []
        ### [FINE MODIFICA AWS] ###

        model = self.loaded_models[model_key]
        n_features = int(model.n_features_in_)

        # 2. Reshape Robusto (Risolve 'list object cannot be interpreted as integer')
        try:
            # Forziamo i dati in un array NumPy numerico 1D e poi modelliamo la matrice
            data_clean = np.asarray(flat_features, dtype=np.float32).ravel()
            input_matrix = data_clean.reshape(-1, n_features)
            n_rows = input_matrix.shape[0]
            print(f"[DEBUG] Reshape riuscito: {n_rows} righe x {n_features} colonne")
        except Exception as e:
            print(f"!!! [ERRORE RESHAPE]: {e}")
            return []

        try:
            return ml_strategy.format_tree_preds(model, input_matrix)
        except Exception as e:
            print(f"!!! [ERRORE ESTRAZIONE VOTI]: {e}")
            return []
