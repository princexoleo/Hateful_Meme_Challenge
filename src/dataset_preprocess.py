##################################

###### This class created for data visulaization but not compled yet #########

####################################




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_path  # Path style access for pandas
from tqdm import tqdm

## dataset download 
# url = "https://drivendata-competition-fb-hateful-memes-data.s3.amazonaws.com/Lnmwdnq3YcF7F3YsJncp.zip?AWSAccessKeyId=AKIAJYJLFLA7N3WRICBQ&Signature=PLVOjg3fmVHcb8Qvuiasj3ZJG7o%3D&Expires=1596241398"

# data_file = wget.download(url)
# from zipfile import ZipFile
# zip_file - 'test.zip'
# password = 'KexZs4tn8hujn1nK'
# with ZipFile(data_file) as zf:
#   zf.extractall(pwd=bytes(password,'utf-8'))
  
# printf("[Dataset] Exatract coompleted")

########### Dataset Preprocessing #############

def load_data():
    data_dir = Path.cwd().parent / "data" 

    img_tar_path = data_dir / "img.tar.gz"
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    test_path = data_dir / "test.jsonl"

    print("[DataDir]",data_dir)
    # print(img_tar_path)

    # if not (data_dir / "img").exists():
    #     with tarfile.open(img_tar_path) as tf:
    #         tf.extractall(data_dir)
    
    # load train sample
    train_samples_frame = pd.read_json(train_path, lines=True)
    # class imblances check
    train_samples_frame.label.value_counts()
    # Text exploration
    train_samples_frame.text.map(
        lambda text: len(text.split(" "))
    ).describe()



