import pandas_path 
from pathlib import Path
from src.HatefulMemesModel import *

def main(hparams):
    hateful_memes_model = HatefulMemesModel(hparams=hparams)
    hateful_memes_model.fit()


###
##input\data
##
    

if __name__=='__main__':
    data_dir = Path.cwd() / "input" / "data" 
    img_tar_path = data_dir / "img.tar.gz"
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    test_path = data_dir / "test.jsonl"
    
    print("Current Dir: ", Path.cwd())
    #print(data_dir)
    #print(train_path)
    
    hparams = {
        
        # Required hparams
        "train_path": train_path,
        "dev_path": dev_path,
        "img_dir": data_dir,
        
        # Optional hparams
        "embedding_dim": 150,
        "language_feature_dim": 300,
        "vision_feature_dim": 300,
        "fusion_output_size": 256,
        "output_path": "model-outputs",
        "dev_limit": None,
        "lr": 0.00005,
        "max_epochs": 10,
        "n_gpu": 1,
        "batch_size": 4,
        # allows us to "simulate" having larger batches 
        "accumulate_grad_batches": 16,
        "early_stop_patience": 3,
    }
    main(hparams)