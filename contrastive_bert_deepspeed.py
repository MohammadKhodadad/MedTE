import torch
import tqdm
import deepspeed
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from utils.model_loader import Model
from utils.tokenizer_loader import load_tokenizer
from utils.loss import InfoNCELoss
from utils.contrastive_data_loader_ordered import get_contrastive_dataloader
from utils.dataloader.mimic import create_mimic_cl_data_from_csv
from utils.dataloader.pubmed import download_pubmed_cl
from utils.dataloader.trialsgov import create_trials_contrastive_learning_data
from utils.dataloader.medmcqa import create_medmcqa_contrastive_leanring_data
from utils.dataloader.medqa import create_medqa_contrastive_leanring_data
from utils.dataloader.medquad import create_medquad_contrastive_leanring_data
from utils.dataloader.wikipedia import create_wiki_cl
from utils.dataloader.curev1 import create_curev1_contrastive_learning_data
from utils.dataloader import filter_datasets
# from utils.optimizer import get_optimizer_and_scheduler



# print('Handling Wiki Data')
# create_wiki_cl()
# print('Handling Mimic Data')
# create_mimic_cl_data_from_csv('./data/discharge_processed.csv','./data/csvs','discharge_diagnosis',['chief_complaint','history_of_present_illness'])
# print('Handling PubMed Data')
# download_pubmed_cl('./data/csvs')
# print('Handling Trials Data')
# create_trials_contrastive_learning_data('./data/clinical_trials_all_studies.csv','./data/csvs')
# print('Handling medmcqa Data')
# create_medmcqa_contrastive_leanring_data('./data/csvs')
# print('Handling medqa Data')
# create_medqa_contrastive_leanring_data('./data/csvs')
# print('Handling medquad Data')
# create_medquad_contrastive_leanring_data('./data/csvs')
# print('Handling curev1 Data')
# create_curev1_contrastive_learning_data('./data/csvs')
# print('Handling Biorxiv Data')
# create_biorxiv_sentence_data('./data/csvs')
# print('Handling Medrxiv Data')
# create_medrxiv_sentence_data('./data/csvs')

# DeepSpeed configuration without ZeRO optimization and gradient accumulation
# DS_CONFIG = {
#     "train_batch_size": 128,  # Adjust batch size as per GPU memory
#     "fp16": {"enabled": True}
# }

EPOCHS = 100
SAVE_STEP = 500
WARM_UP_STEPS = 1000
TOTAL_STEPS = 100000
LEARNING_RATE = 0.00005
BATCH_SIZE = 800  # Ensure batch size is consistent
# GRAD_ACC_STEPS=BATCH_SIZE/16

DS_CONFIG = {
    "train_batch_size": BATCH_SIZE,  
    # "gradient_accumulation_steps": GRAD_ACC_STEPS,
    "bf16": {"enabled": True},
    # "zero_optimization": {
    #     "stage": 3,
    #     "offload_optimizer": {
    #         "device": "cpu",
    #         "pin_memory": True
    #     },
    #     "offload_param": {
    #         "device": "cpu",
    #         "pin_memory": True
    #     }
    #     },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.00005,  # Learning rate
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "gradient_checkpointing": True,
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.00005,  # Max LR after warmup
            "warmup_num_steps": 1000  # Warmup steps
        }
    },
}



# Load tokenizer and model
print('Data Loading...')
# tokenizer = load_tokenizer("bert-base-uncased")
# model = Model("/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/weights/mlm/step_130000")

# tokenizer = load_tokenizer("intfloat/e5-base")
# model = Model("intfloat/e5-base")

tokenizer = load_tokenizer("thenlper/gte-base")
model = Model("thenlper/gte-base",peft_r=None,grad_checkpointing=True)



# tokenizer = load_tokenizer("Alibaba-NLP/gte-base-en-v1.5")
# model = Model("Alibaba-NLP/gte-base-en-v1.5")


# tokenizer = load_tokenizer("sentence-transformers/all-mpnet-base-v2")
# model = Model("sentence-transformers/all-mpnet-base-v2")

# Load data with specified batch size
# filter_datasets(
#     input_dir  = './data/csvs',
#     reference_dir = './data/bench_data',
#     output_dir = './data/filtered_csvs'
# )
train_loader, test_loader = get_contrastive_dataloader('./data/csvs', tokenizer, batch_size=BATCH_SIZE,max_length=512)
# train_loader, test_loader = get_contrastive_dataloader('./data/Ali_csvs', tokenizer, batch_size=BATCH_SIZE,max_length=512)
# Loss function
criterion = InfoNCELoss()
# criterion = InfoNCELossChunked(chunk_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get optimizer and scheduler from utility function
# optimizer, scheduler = get_optimizer_and_scheduler(model, LEARNING_RATE, WARM_UP_STEPS, TOTAL_STEPS)

# DeepSpeed initialization
# model, optimizer, _, _ = deepspeed.initialize(
#     model=model,
#     model_parameters=model.parameters(),
#     config=DS_CONFIG,
#     optimizer=optimizer
# )
print("Deepspeed Initialization ...")
model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=DS_CONFIG,
)
print("Training...")

step = 0
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1} (Training)")
    
    for batch1, batch2 in progress_bar:
        batch1 = {key: batch1[key].to(device) for key in batch1.keys()}
        batch2 = {key: batch2[key].to(device) for key in batch2.keys()}
        
        outputs1 = model.encode(batch1, use_cls=False)
        outputs2 = model.encode(batch2, use_cls=False)
        loss = criterion(outputs1, outputs2)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # scheduler.step()
        
        model.backward(loss)
        # loss_list = criterion(outputs1, outputs2)
        # for loss_chunk in loss_list:
        #     model.backward(loss_chunk)
        model.step()
        total_loss += loss.item()
        step += 1
        
        if step % SAVE_STEP == 0:
            print(f'Saving model at step {step}')
            model.module.save_pretrained(f'./weights/contrastive_gte_15/ds_step_{step}/')
        
        avg_loss = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix({'Step': step, "Loss": avg_loss})
    
    print(f"Epoch {epoch + 1}, Step: {step}, Loss: {avg_loss}")
    
    model.eval()
    total_eval_loss = 0
    progress_bar = tqdm.tqdm(test_loader, desc=f"Epoch {epoch + 1} (Testing)")
    
    with torch.no_grad():
        for batch1, batch2 in progress_bar:
            batch1 = {key: batch1[key].to(device) for key in batch1.keys()}
            batch2 = {key: batch2[key].to(device) for key in batch2.keys()}
            
            outputs1 = model.encode(batch1, use_cls=False)
            outputs2 = model.encode(batch2, use_cls=False)
            
            loss = criterion(outputs1, outputs2)
            total_eval_loss += loss.item()
            
            avg_eval_loss = total_eval_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({"Eval Loss": avg_eval_loss})
    
    print(f"Epoch {epoch + 1}, Evaluation Loss: {avg_eval_loss}")