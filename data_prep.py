import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
import sys
import boto3
import io

# Aggiungiamo la root del progetto al path per poter importare 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config

# Carichiamo la configurazione
CONFIG = load_config()
S3_BUCKET = CONFIG.get("s3_bucket") 
PATHS = CONFIG.get("paths")

if not S3_BUCKET or not PATHS:
    raise ValueError("Il file config.json deve contenere 's3_bucket' e 'paths'.")

def prepare_zero_copy_dataset(name, input_path, is_regression):
    
    # 1. Definiamo i percorsi di output unici su S3
    train_s3_key = f"datasets/{name}/train.csv"
    test_s3_key = f"datasets/{name}/test.csv"
    
    print(f"\n[{name.upper()}] Inizio preparazione DataPrep per Zero-Copy...")
    print(f" -> Sorgente: {input_path}")
    print(f" -> Destinazione Train: s3://{S3_BUCKET}/{train_s3_key}")
    print(f" -> Destinazione Test:  s3://{S3_BUCKET}/{test_s3_key}")

    if not os.path.exists(input_path):
        print(f" -> ERRORE: File sorgente locale non trovato in {input_path}.")
        print(" Assicurati di scaricare prima il file originale!")
        return

    s3_client = boto3.client('s3')
    
    # Creiamo buffer in RAM per fare upload in streaming su S3 senza salvare file temporanei
    # (Attenzione: se il file è molto grande, meglio scrivere su disco locale temporaneo)
    temp_train_path = f"/tmp/{name}_train.csv"
    temp_test_path = f"/tmp/{name}_test.csv"

    total_rows = 0
    chunk_size = 500000 

    # 2. Leggiamo, mischiamo e dividiamo a blocchi
    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size, dtype=np.float32)):
        
        # Shuffle interno al chunk
        chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)
        total_rows += len(chunk)
        
        # SPLIT 80/20
        if is_regression:
            train_chunk, test_chunk = train_test_split(chunk, test_size=0.20, random_state=42)
        else:
            y_chunk_for_split = chunk['Label'].astype(int)
            train_chunk, test_chunk = train_test_split(chunk, test_size=0.20, random_state=42, stratify=y_chunk_for_split)

        # 3. Scriviamo sui file temporanei locali
        mode = 'w' if chunk_idx == 0 else 'a'
        header = True if chunk_idx == 0 else False
        
        train_chunk.to_csv(temp_train_path, mode=mode, header=header, index=False)
        test_chunk.to_csv(temp_test_path, mode=mode, header=header, index=False)

    print(f" -> Elaborate e mescolate {total_rows} righe totali. Upload su S3 in corso...")

    # 4. Upload finale su S3
    s3_client.upload_file(temp_train_path, S3_BUCKET, train_s3_key)
    print(f" Upload completato: {train_s3_key}")
    
    s3_client.upload_file(temp_test_path, S3_BUCKET, test_s3_key)
    print(f" Upload completato: {test_s3_key}")

    # 5. Pulizia
    os.remove(temp_train_path)
    os.remove(temp_test_path)
    print("\n -> Data Prep Zero-Copy completata con successo! 🚀")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Prep: Split in Train/Test unici su S3")
    parser.add_argument('--dataset', type=str, required=True, choices=['taxi', 'higgs', 'ids', 'covertype'])
    args = parser.parse_args()

    is_regression = True if args.dataset == 'taxi' else False
    input_file = PATHS[args.dataset]["full"]
    
    prepare_zero_copy_dataset(args.dataset, input_file, is_regression)