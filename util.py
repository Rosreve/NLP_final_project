from config import TRAIN_CLAIMS_PATH, DEV_CLAIMS_PATH, EVIDENCE_PATH, OUTPUT_PATH
import json

def load_data():
    with open(TRAIN_CLAIMS_PATH, "r") as f:
        train_claims = json.load(f) 
    with open(DEV_CLAIMS_PATH, "r") as f:
        dev_claims = json.load(f) 
    with open(EVIDENCE_PATH, "r") as f:
        evidence = json.load(f)

    return train_claims, dev_claims, evidence