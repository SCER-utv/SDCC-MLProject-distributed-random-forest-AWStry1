from concurrent.futures import ThreadPoolExecutor, as_completed

import grpc
import numpy as np
import boto3   # [NUOVO] Per Auto-Healing
import time    # [NUOVO] Per le pause
from src.network.proto import rf_service_pb2_grpc, rf_service_pb2
import socket
import threading # Aggiungi questo import in alto

class GrpcMaster:
    def __init__(self, config, strategy):
        
        self.config = config
        self.strategy = strategy
        self.workers = config['workers']
        self.channels = []
        self.stubs = []
        self.worker_assignments = {}
        # [MODIFICA FT] Lista dei worker "morti" da escludere
        self.dead_workers = set()
        self.recovery_lock = threading.Lock()
        self.is_recovering = {}
       

    def _spawn_new_worker(self, old_worker_address):
        if old_worker_address in self.is_recovering and self.is_recovering[old_worker_address] is not None:
             return self.is_recovering[old_worker_address]

        with self.recovery_lock:
            if old_worker_address in self.is_recovering and self.is_recovering[old_worker_address] is not None:
                return self.is_recovering[old_worker_address]

            self.is_recovering[old_worker_address] = None
            print(f"\n{'='*50}\n [AUTO-HEALING] CRASH RILEVATO: {old_worker_address}\n{'='*50}")
            
            ec2_client = boto3.client('ec2', region_name='us-east-1')
            ec2_resource = boto3.resource('ec2', region_name='us-east-1')
            old_ip = old_worker_address.split(':')[0]
            
            try:
                worker_num = self.workers.index(old_worker_address) + 1
                new_name = f"DRF-worker{worker_num}-autohealed"
            except ValueError:
                new_name = "worker_extra_autohealed"

            def _wait_for_port(target_ip, max_attempts=60): # 60 tentativi da 5s = 5 MINUTI DI ATTESA!
                print(f" [AUTO-HEALING] Inizio polling: busso alla porta 50051 di {target_ip}...")
                for attempt in range(max_attempts):
                    try:
                        with socket.create_connection((target_ip, 50051), timeout=2):
                            print(f" [AUTO-HEALING] >>> MIRACOLO! Porta 50051 APERTA al tentativo {attempt + 1}! <<<")
                            return True
                    except (socket.timeout, ConnectionRefusedError, OSError):
                        if attempt % 2 == 0: # Stampa solo ogni 10 secondi per non inondare il terminale
                            print(f"   ... Tentativo {attempt + 1}/{max_attempts}: Porta chiusa. Attendo...")
                        time.sleep(5)
                return False

            reboot_successful = False
            final_ip = None

            try:
                # --- FASE 1: TENTATIVO DI RIAVVIO / ACCENSIONE ---
                print(f" [AUTO-HEALING] Interrogo AWS per trovare l'IP {old_ip}...")
                response = ec2_client.describe_instances(Filters=[{'Name': 'private-ip-address', 'Values': [old_ip]}])
                
                if response['Reservations'] and response['Reservations'][0]['Instances']:
                    instance = response['Reservations'][0]['Instances'][0]
                    instance_id = instance['InstanceId']
                    instance_state = instance['State']['Name']
                    
                    print(f" [AUTO-HEALING] TROVATA SU AWS: Istanza {instance_id} | Stato: {instance_state.upper()}")
                    
                    try:
                        if instance_state == 'stopped':
                            print(f" [AUTO-HEALING] -> Inviato comando 'START_INSTANCES' ad AWS...")
                            ec2_client.start_instances(InstanceIds=[instance_id])
                            print(f" [AUTO-HEALING] -> Attendo che AWS accenda l'hardware...")
                            ec2_client.get_waiter('instance_running').wait(InstanceIds=[instance_id])
                            print(f" [AUTO-HEALING] -> Hardware ACCESO! Pausa di 30s per il boot di Linux...")
                            time.sleep(30)
                            
                        elif instance_state in ['running', 'pending']:
                            print(f" [AUTO-HEALING] -> Inviato comando 'REBOOT_INSTANCES' ad AWS...")
                            ec2_client.reboot_instances(InstanceIds=[instance_id])
                            print(f" [AUTO-HEALING] -> Pausa di 45s per far spegnere e riaccendere la macchina...")
                            time.sleep(45)

                        # --- NUOVO BLOCCO AGGIUNTO ---
                        elif instance_state == 'stopping':
                            print(f" [AUTO-HEALING] -> L'istanza si sta spegnendo. Attendo che sia completamente ferma...")
                            ec2_client.get_waiter('instance_stopped').wait(InstanceIds=[instance_id])
                            print(f" [AUTO-HEALING] -> Ora è 'stopped'. Inviato comando 'START_INSTANCES'...")
                            ec2_client.start_instances(InstanceIds=[instance_id])
                            ec2_client.get_waiter('instance_running').wait(InstanceIds=[instance_id])
                            print(f" [AUTO-HEALING] -> Hardware ACCESO! Pausa di 30s per il boot...")
                            time.sleep(30)
                        # -----------------------------
                        
                        # Polling ESTREMO sulla vecchia macchina
                        if _wait_for_port(old_ip, max_attempts=60):
                            reboot_successful = True
                            final_ip = old_ip
                        else:
                            print(f" [AUTO-HEALING] FALLIMENTO: Ho aspettato 5 minuti ma la porta su {old_ip} non si è mai aperta.")
                            
                    except Exception as e:
                        print(f" [AUTO-HEALING] Errore Boto3 durante la Fase 1: {e}")
                else:
                    print(f" [AUTO-HEALING] Macchina non trovata su AWS.")

                # --- FASE 2: CREAZIONE NUOVA ISTANZA (Se Fase 1 fallisce) ---
                if not reboot_successful:
                    print(f"\n [AUTO-HEALING] --- AVVIO PIANO B: CREAZIONE NUOVA ISTANZA ---")
                    AMI_ID = 'ami-0f860c4d616d1b6ac'  
                    SUBNET_ID = 'subnet-0a61f2346de4cd937'
                    SG_ID = 'sg-004dedd411b0fe130'     
                    KEY_NAME = 'distributed-random-forest-key'

                    startup_script = """#!/bin/bash
                    sudo -u ubuntu bash -c '
                    cd /home/ubuntu/SDCC-MLProject-distributed-random-forest-AWStry
                    source venv/bin/activate
                    export PYTHONPATH=$(pwd)
                    export AWS_S3_BUCKET=distributed-random-forest-bkt
                    nohup python src/worker.py 50051 > /home/ubuntu/worker_log.txt 2>&1 &
                    '
                    """

                    instances = ec2_resource.create_instances(
                        ImageId=AMI_ID, InstanceType='t3.large', SubnetId=SUBNET_ID,
                        SecurityGroupIds=[SG_ID], KeyName=KEY_NAME, UserData=startup_script,
                        MinCount=1, MaxCount=1, IamInstanceProfile={'Name': 'LabInstanceProfile'}
                    )
                    
                    new_instance = instances[0]
                    new_instance.create_tags(Tags=[{'Key': 'Name', 'Value': new_name}])

                    print(f" [AUTO-HEALING] Creazione {new_instance.id} in corso. Attesa boot...")
                    new_instance.wait_until_running()
                    new_instance.reload()
                    new_ip = new_instance.private_ip_address
                    
                    if _wait_for_port(new_ip, max_attempts=40):
                        final_ip = new_ip
                    else:
                        print(f" [AUTO-HEALING] ERRORE FATALE: Neanche la nuova istanza {new_ip} risponde.")
                        del self.is_recovering[old_worker_address]
                        return None

                print(" [AUTO-HEALING] Attesa di stabilizzazione del demone gRPC (15s)...")
                time.sleep(15)

                new_address = f"{final_ip}:50051"
                self.is_recovering[old_worker_address] = new_address
                return new_address

            except Exception as e:
                print(f" [AUTO-HEALING] Fallimento critico generale: {e}")
                if old_worker_address in self.is_recovering:
                     del self.is_recovering[old_worker_address]
                return None
            
    # [NUOVO METODO ZERO-COPY] Interroga S3 senza scaricare il file        
    def _get_total_rows_s3_select(self, bucket, key):
        print(f" [S3 Select] Lancio query SQL 'SELECT count(*)' su s3://{bucket}/{key}...")
        s3 = boto3.client('s3')
        try:
            resp = s3.select_object_content(
                Bucket=bucket, Key=key,
                ExpressionType='SQL', Expression='SELECT count(*) FROM S3Object',
                InputSerialization={'CSV': {'FileHeaderInfo': 'USE', 'AllowQuotedRecordDelimiter': False}},
                OutputSerialization={'CSV': {}}
            )
            for event in resp['Payload']:
                if 'Records' in event:
                    total_rows = int(event['Records']['Payload'].decode('utf-8').strip())
                    print(f" [S3 Select] Trovate {total_rows} righe!")
                    return total_rows
            return 0
        except Exception as e:
            print(f"[ERRORE S3 Select]: {e}")
            raise e        

    def connect(self):
        print("--- Connessione ai Worker ---")

        # [MODIFICA FT] Connessione robusta: se uno è spento all'avvio, lo ignoriamo subito
        # ridondante
        active_stubs = []
        active_workers = [] # [NUOVO]

        for addr in self.workers:
            try:
                ch = grpc.insecure_channel(addr)
                grpc.channel_ready_future(ch).result(timeout=2)
                stub = rf_service_pb2_grpc.RandomForestWorkerStub(ch)
                self.channels.append(ch)
                active_stubs.append(stub)
                active_workers.append(addr) # [NUOVO]
                self.stubs.append(stub)
                print(f"Connesso a {addr}")
            except Exception as e:
                print(f"Errore connessione {addr}: {e}")

         # [NUOVO] Mantiene allineati indirizzi e stubs!
        self.stubs = active_stubs
        self.workers = active_workers
        if not self.stubs:
            raise Exception("CRITICO: Nessun worker disponibile! Impossibile avviare il cluster.")


    def close(self):
        for ch in self.channels: ch.close()

    # [MODIFICA ZERO-COPY] Firma aggiornata: riceve i path dal Master principale
    def train(self, train_s3_uri, train_s3_key, target_bucket):
        # [POLIMORFISMO] Recuperiamo il bit del task dalla strategia
        task_bit = self.strategy.get_task_type()
        print(f"\n--- Avvio Training Distribuito ({self.strategy.__class__.__name__}) ---")

        # [MODIFICA ETEROGENEA] 1. DEFINIZIONE STRATEGIE
        # Creiamo profili diversi per i worker

        total_trees = self.config['total_trees']
        num_workers = len(self.stubs)

        if num_workers == 0:
            print("Nessun worker disponibile.")
            return
        
        # 1. Conta il numero di righe all'interno del dataset (fatto per garantire generalità per ogni dataset)
        try:
            total_rows = self._get_total_rows_s3_select(target_bucket, train_s3_key)
        except Exception:
            print("Fallimento critico S3 Select. Esco.")
            return
        
        # 2. Matematica pura per distribuire il carico
        rows_per_worker = total_rows // num_workers
        remainder_rows = total_rows % num_workers

        trees_per_worker = total_trees // num_workers
        remainder_trees = total_trees % num_workers

        tasks = []

        # [MODIFICA] Estraiamo la lista delle strategie passate dal master.py
        strategies = self.config.get('worker_strategies', [])
        current_skip = 0

        # 3. Creazione dinamica dei task
        for i in range(num_workers):
            n_estimators = trees_per_worker + (remainder_trees if i == num_workers - 1 else 0)
            n_rows = rows_per_worker + (remainder_rows if i == num_workers - 1 else 0)
            sub_id = f"part_{i + 1}"
            conf = strategies[i]

            print(f" -> Assegnazione {sub_id} (Worker {i+1}): {n_estimators} alberi | Righe: {n_rows} (Offset: {current_skip})")

            tasks.append({
                'worker_addr': self.workers[i], 
                'stub': self.stubs[i],
                'subforest_id': sub_id,
                'seed': i * 1000,
                'n_estimators': n_estimators,
                'dataset_path': train_s3_uri, # TUTTI LEGGONO LO STESSO IDENTICO FILE
                'max_depth': conf['max_depth'],      
                'max_features': str(conf['max_features']), 
                'criterion': conf['criterion'],
                'skip_rows': current_skip,    # <--- NUOVO
                'num_rows': n_rows            # <--- NUOVO
            })
            
            current_skip += n_rows 
            self.worker_assignments[sub_id] = (self.stubs[i], self.workers[i])

        def _execute_train_request(task):

            MAX_RETRIES = 1 
            current_stub = task['stub']
            current_addr = task['worker_addr']
            
            for attempt in range(MAX_RETRIES + 1):
                try:
                    req = rf_service_pb2.TrainRequest(
                        model_id=self.config['model_id'],
                        subforest_id=task['subforest_id'],
                        dataset_s3_path=task["dataset_path"], 
                        seed=task['seed'],
                        n_estimators=task['n_estimators'],
                        task_type=task_bit,  
                        max_depth=int(task['max_depth']),
                        max_features=str(task['max_features']),
                        criterion=str(task['criterion']),
                        skip_rows=task['skip_rows'], # <--- NUOVO
                        num_rows=task['num_rows']    # <--- NUOVO
                    )

                    print(f"Worker {task['subforest_id']} [{current_addr}] -> Inizio {task['n_estimators']} alberi...")
                    resp = current_stub.TrainSubForest(req, timeout=1200)
                    return task['subforest_id'], resp.success

                except grpc.RpcError as e:
                    print(f"\nCRASH RILEVATO: Worker {task['subforest_id']} ({current_addr}) è caduto durante il training!")
                    
                    if attempt < MAX_RETRIES:
                        new_addr = self._spawn_new_worker(current_addr)
                        
                        if new_addr:
                            try:
                                ch = grpc.insecure_channel(new_addr)
                                new_stub = rf_service_pb2_grpc.RandomForestWorkerStub(ch)
                            
                                current_stub = new_stub
                                current_addr = new_addr
                            
                                # Aggiorniamo la rubrica globale (Thread-safe)
                                self.worker_assignments[task['subforest_id']] = (current_stub, current_addr)
                            
                                print(f" RIPRISTINO COMPLETATO! {new_addr} riprova il training di {task['subforest_id']}...")
                                continue 
                            except Exception as ex:
                                print(f" Handshake fallito col nuovo nodo {new_addr}: {ex}")
                                return task['subforest_id'], False
                        else:
                            print(f" Worker {task['subforest_id']} perso definitivamente.")
                            return task['subforest_id'], False
                    else:
                        print(f" Worker {task['subforest_id']} ha esaurito i tentativi.")
                        return task['subforest_id'], False
                    
        completed_tasks = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_execute_train_request, t) for t in tasks]
            for f in as_completed(futures):
                sid, success = f.result()
                if success:
                    print(f"Task {sid} completato.")
                    completed_tasks += 1
                else:
                    print(f"Task {sid} FALLITO DEFINITIVAMENTE. Escludo worker.")
                    if sid in self.worker_assignments:
                        del self.worker_assignments[sid]
                    

    def predict_batch(self, batch_rows):
            flat_feats = [x for r in batch_rows for x in r]
            row_votes = [[] for _ in range(len(batch_rows))]
            task_bit = self.strategy.get_task_type()
            responded_ids = set()
            
            # Creiamo un ID univoco per questo specifico blocco di dati
            chunk_id = hex(id(batch_rows))[-6:]
    
            def _ask_worker(sub_id, initial_stub, initial_addr):
                MAX_RETRIES = 1
                current_stub = initial_stub
                current_addr = initial_addr
    
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        req = rf_service_pb2.PredictRequest(
                            model_id=self.config['model_id'],
                            subforest_id=sub_id,
                            features=flat_feats,
                            task_type=task_bit
                        )
                        
                        resp = current_stub.Predict(req, timeout=180)
                        
                        # --- STAMPA DI CONFERMA VISIVA ---
                        print(f" Worker {sub_id} ({current_addr}) ha completato il blocco [ID: {chunk_id}]")
                        return sub_id, resp
    
                    except grpc.RpcError as e:
                        print(f"\n [AUTO-HEALING] CRASH Worker {sub_id} ({current_addr}) sul blocco [ID: {chunk_id}]!")
                        
                        if attempt < MAX_RETRIES:
                            # Qui il thread morto chiama il ripristino. Il Worker Sano invece ha già fatto 'return' sopra!
                            new_addr = self._spawn_new_worker(current_addr)
                            
                            if new_addr:
                                ch = grpc.insecure_channel(new_addr)
                                try:
                                    grpc.channel_ready_future(ch).result(timeout=60)
                                    current_stub = rf_service_pb2_grpc.RandomForestWorkerStub(ch)
                                    current_addr = new_addr
                                    
                                    # Aggiorna la rubrica
                                    self.worker_assignments[sub_id] = (current_stub, current_addr)
                                    
                                    print(f" RIPRISTINO COMPLETATO! {new_addr} riprova il blocco [ID: {chunk_id}]...")
                                    continue 
                                except Exception as ex:
                                    print(f" Handshake fallito col nuovo nodo {new_addr}: {ex}")
                                    return sub_id, None
                            else:
                                return sub_id, None
                        else:
                            print(f" Worker {sub_id} perso definitivamente.")
                            return sub_id, None
                            
                    except Exception as e:
                        return sub_id, None
    
            active_workers = len(self.worker_assignments)
            if active_workers == 0:
                return [None] * len(batch_rows)
    
            with ThreadPoolExecutor(max_workers=active_workers) as executor:
                futures = {
                    executor.submit(_ask_worker, sid, stub, addr): sid
                    for sid, (stub, addr) in self.worker_assignments.items()
                }
    
                for f in as_completed(futures):
                    sid, resp = f.result()
                    if not resp: continue
    
                    responded_ids.add(sid)
                    worker_vals = self.strategy.extract_predictions(resp)
                    if not worker_vals: continue
    
                    n_trees_in_worker = len(worker_vals) // len(batch_rows)
                    if n_trees_in_worker == 0: continue
    
                    for i in range(len(batch_rows)):
                        start = i * n_trees_in_worker
                        end = start + n_trees_in_worker
                        row_votes[i].extend(worker_vals[start:end])
    
            return [self.strategy.aggregate(vals) for vals in row_votes]
