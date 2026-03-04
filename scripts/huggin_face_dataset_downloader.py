import os
from datasets import load_dataset

# THIS FILE SHOULD BE IN GENERAL REPOSITORY TO ACCESS DATA FOLDER

token = "" # insert huggin face reading token 

# 2) Download + cache the dataset
ds = load_dataset(
    "Pratheesh99/animal-faces-raw",
    token=token,          
)

# 3) Save into repository folder
out_dir = "./data/animal-faces-raw"
ds.save_to_disk(out_dir)

print(f"Saved to: {out_dir}")