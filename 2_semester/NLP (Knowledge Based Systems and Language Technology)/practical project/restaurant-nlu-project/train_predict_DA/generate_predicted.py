# Moga Patricia & Ciobanu Sergiu-Tudor

import os
import sys
import json
from pathlib import Path
import argparse
import tqdm
current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = current_script_dir
while base_dir and not os.path.exists(os.path.join(base_dir, "dialtask")):
    parent = os.path.dirname(base_dir)
    if parent == base_dir:
        break
    base_dir = parent

sys.path.append(base_dir)

from dialtask.dialogue import Dialogue
from dialtask.nlu.stat_dstc import StatisticalNLU
from dialtask.nlu.start_dstc_bert import StatisticalNLUBert

class DummyLogger:
    def info(self, msg, *args):
        pass

BASE_DIR = Path(base_dir)
TEST_DATA_PATH = BASE_DIR / "data" / "dstc_extracted_DA" / "results_nlu_test.json"

def main(args=None):
    if args.model == "bert" and args.file == "predicted_bert.txt":
        nlu = StatisticalNLUBert()
    elif args.model == "lr" and args.file == "predicted.txt":
        nlu = StatisticalNLU()

    OUTPUT_PATH = BASE_DIR / "train_predict_DA" / args.file
    
    logger = DummyLogger()
    
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for item in tqdm.tqdm(test_data, desc="Generating Predictions"):

            dial = Dialogue()
            dial.user = item.get("usr", "")
            dial = nlu(dial, logger)
            if args.model == "bert":
                f_out.write(dial.nlu.to_cambridge_da_string() + "\n")
            else:
                final_da_string = str(dial.nlu).replace("'", "")
                f_out.write(final_da_string + "\n")

    print("Predictions successfully generated!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=False, default="lr", choices=["lr", "bert"])
    parser.add_argument('-file', type=str, required=False, default="predicted.txt", choices=["predicted.txt", "predicted_bert.txt"])
    args = parser.parse_args()
    main(args)