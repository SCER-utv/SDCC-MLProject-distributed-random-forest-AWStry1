import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
import sys
import boto3

# Aggiungiamo la root del progetto al path per poter importare 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config

# Carichiamo la configurazione per prendere il nome del bucket
CONFIG = load_config()
S3_BUCKET = CONFIG.get("s3_bucket", "distributed-random-forest-bkt") 

def prepare_zero_copy_dataset(name, is_regression):
    
    # 1. Definiamo i path SORGENTE (dove si trovano i dataset unici ottimizzati)
    # [NOTA]: Ho inserito i percorsi in cui tenevi i file "optimized". Se li hai spostati, modificali qui!
    sources = {
        "higgs": f"s3://{S3_BUCKET}/data/higgs/processed/higgs_optimized.csv",
        "taxi": f"s3://{S3_BUCKET}/data/taxi/processed/taxi_optimized.csv",
        "ids": f"s3://{S3_BUCKET}/data/ids/processed/ids_optimized.csv"
    }
    
    input_s3_uri = sources[name]
    
    # 2. Definiamo i path di DESTINAZIONE (allineati al tuo nuovo config.json!)
    train_s3_key = f"data/3_processed/{name}/{name}_train.csv"
    test_s3_key = f"data/3_processed/{name}/{name}_test.csv"
    
    print(f"\n[{name.upper()}] Inizio preparazione DataPrep per Zero-Copy...")
    print(f" -> Sorgente S3: {input_s3_uri}")
    print(f" -> Destinazione Train: s3://{S3_BUCKET}/{train_s3_key}")
    print(f" -> Destinazione Test:  s3://{S3_BUCKET}/{test_s3_key}")

    # File temporanei locali per appoggiare i dati prima dell'upload
    temp_train_path = f"/tmp/{name}_train.csv"
    temp_test_path = f"/tmp/{name}_test.csv"

    total_rows = 0
    chunk_size = 500000 

    print("\n -> Lettura a blocchi da S3 e Split in corso (potrebbe richiedere qualche minuto)...")
    
    # Pandas legge direttamente da S3 grazie a s3fs!
    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(input_s3_uri, chunksize=chunk_size, dtype=np.float32)):
            
            # Shuffle interno al chunk
            chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)
            total_rows += len(chunk)
            
            # SPLIT 80/20
            if is_regression:
                train_chunk, test_chunk = train_test_split(chunk, test_size=0.20, random_state=42)
            else:
                y_chunk_for_split = chunk['Label'].astype(int)
                train_chunk, test_chunk = train_test_split(chunk, test_size=0.20, random_state=42, stratify=y_chunk_for_split)

            # Scriviamo sui file temporanei locali
            mode = 'w' if chunk_idx == 0 else 'a'
            header = True if chunk_idx == 0 else False
            
            train_chunk.to_csv(temp_train_path, mode=mode, header=header, index=False)
            test_chunk.to_csv(temp_test_path, mode=mode, header=header, index=False)
            
            print(f"    - Elaborate {total_rows} righe...")
            
    except FileNotFoundError:
        print(f"\n[ERRORE FATALE] Pandas non trova il file sorgente S3 all'indirizzo:\n{input_s3_uri}")
        print("Assicurati che la stringa in 'sources' corrisponda alla posizione reale del dataset!")
        sys.exit(1)

    print(f"\n -> Split completato ({total_rows} righe). Upload su S3 in corso...")

    # 4. Upload finale su S3
    s3_client = boto3.client('s3')
    
    s3_client.upload_file(temp_train_path, S3_BUCKET, train_s3_key)
    print(f" Upload Train completato: s3://{S3_BUCKET}/{train_s3_key}")
    
    s3_client.upload_file(temp_test_path, S3_BUCKET, test_s3_key)
    print(f" Upload Test completato:  s3://{S3_BUCKET}/{test_s3_key}")

    # 5. Pulizia
    os.remove(temp_train_path)
    os.remove(temp_test_path)
    print("\n -> Data Prep Zero-Copy completata con successo! 🚀")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Prep: Split in Train/Test unici su S3")
    parser.add_argument('--dataset', type=str, required=True, choices=['taxi', 'higgs', 'ids'])
    args = parser.parse_args()

    is_regression = True if args.dataset == 'taxi' else False
    
    prepare_zero_copy_dataset(args.dataset, is_regression)
