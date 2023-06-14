
# Helper functions to read and preprocess data files from Matlab format
# Data science libraries
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Others
from pathlib import Path
from tqdm.auto import tqdm
import requests

def matfile_to_dic(folder_path):
    '''
    Read all the matlab files of the CWRU Bearing Dataset and return a 
    dictionary. The key of each item is the filename and the value is the data 
    of one matlab file, which also has key value pairs.
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
    Return:
        output_dic: 
            Dictionary which contains data of all files in the folder_path.
    '''
    output_dic = {}
    for _, filepath in enumerate(folder_path.glob('*.mat')):
        # strip the folder path and get the filename only.
        key_name = str(filepath).split('\\')[-1]
        output_dic[key_name] = scipy.io.loadmat(filepath)
    return output_dic


def remove_dic_items(dic):
    '''
    Remove redundant data in the dictionary returned by matfile_to_dic inplace.
    '''
    # For each file in the dictionary, delete the redundant key-value pairs
    for _, values in dic.items():
        del values['__header__']
        del values['__version__']    
        del values['__globals__']


def rename_keys(dic):
    '''
    Rename some keys so that they can be loaded into a 
    DataFrame with consistent column names
    '''
    # For each file in the dictionary
    for _,v1 in dic.items():
        # For each key-value pair, rename the following keys 
        for k2,_ in list(v1.items()):
            if 'DE_time' in k2:
                v1['DE_time'] = v1.pop(k2)
            elif 'BA_time' in k2:
                v1['BA_time'] = v1.pop(k2)
            elif 'FE_time' in k2:
                v1['FE_time'] = v1.pop(k2)
            elif 'RPM' in k2:
                v1['RPM'] = v1.pop(k2)


def label(filename):
    '''
    Function to create label for each signal based on the filename. Apply this
    to the "filename" column of the DataFrame.
    Usage:
        df['label'] = df['filename'].apply(label)
    '''
    if 'B' in filename:
        return 'B'
    elif 'IR' in filename:
        return 'IR'
    elif 'OR' in filename:
        return 'OR'
    elif 'Normal' in filename:
        return 'N'


def matfile_to_df(folder_path):
    '''
    Read all the matlab files in the folder, preprocess, and return a DataFrame
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
    Return:
        DataFrame with preprocessed data
    '''
    dic = matfile_to_dic(folder_path)
    remove_dic_items(dic)
    rename_keys(dic)
    df = pd.DataFrame.from_dict(dic).T
    df = df.reset_index().rename(mapper={'index':'filename'},axis=1)
    df['label'] = df['filename'].apply(label)
    return df.drop(['BA_time','FE_time', 'RPM', 'ans'], axis=1, errors='ignore')


def divide_signal(df, segment_length,seg_num =None ):
    '''
    This function divide the signal into segments, each with a specific number 
    of points as defined by segment_length. Each segment will be added as an 
    example (a row) in the returned DataFrame. Thus it increases the number of 
    training examples. The remaining points which are less than segment_length 
    are discarded.
    
    Parameter:
        df: 
            DataFrame returned by matfile_to_df()
        segment_length: 
            Number of points per segment.
    Return:
        DataFrame with segmented signals and their corresponding filename and 
        label
    '''
    dic = {}
    idx = 0
    for i in range(df.shape[0]):
        n_sample_points = len(df.iloc[i,1])
        seg_num = segment_length  if seg_num is None else seg_num
        n_segments = (n_sample_points -segment_length) // seg_num
    
        for segment in range(n_segments):
            dic[idx] = {
                'signal': df.iloc[i,1][seg_num * segment:segment * seg_num+ segment_length], 
                # 'label': df.iloc[i,2],
                'filename' : df.iloc[i,0]
            }
            idx += 1

    df_tmp = pd.DataFrame.from_dict(dic,orient='index')
    return pd.concat(
        [
            df_tmp[['filename']],
            # df_tmp[['label', 'filename']],
            pd.DataFrame(np.hstack(df_tmp["signal"].values).T),
        ],
        axis=1,
    )


def normalize_signal(df):
    '''
    Normalize the signals in the DataFrame returned by matfile_to_df() by subtracting
    the mean and dividing by the standard deviation.
    '''
    mean = df['DE_time'].apply(np.mean)
    std = df['DE_time'].apply(np.std)
    df['DE_time'] = (df['DE_time'] - mean) / std


def get_df_all(data_path, segment_length=512,seg_num=None ,normalize=False):
    '''
    Load, preprocess and return a DataFrame which contains all signals data and
    labels and is ready to be used for model training.
    
    Parameter:
        normal_path: 
            Path of the folder which contains matlab files of normal bearings
        DE_path: 
            Path of the folder which contains matlab files of DE faulty bearings
        segment_length: 
            Number of points per segment. See divide_signal() function
        normalize: 
            Boolean to perform normalization to the signal data
    Return:
        df_all: 
            DataFrame which is ready to be used for model training.
    '''
    df = matfile_to_df(data_path)

    if normalize:
        normalize_signal(df)
    df_processed = divide_signal(df, segment_length,seg_num)
     
    # map_label = {'N':0, 'B':1, 'IR':2, 'OR':3}
    map_label = {'12k_Drive_End_IR014_0_169.mat':5 , '12k_Drive_End_B021_0_222.mat':3 , '12k_Drive_End_OR007@6_0_130.mat': 7, '12k_Drive_End_IR021_0_209.mat': 6, 'normal_0_97.mat': 0, '12k_Drive_End_IR007_0_105.mat':4 , '12k_Drive_End_B007_0_118.mat':1 , '12k_Drive_End_B014_0_185.mat': 2, '12k_Drive_End_OR014@6_0_197.mat': 8, '12k_Drive_End_OR021@6_0_234.mat': 9}


    
    df_processed['label'] = df_processed['filename'].map(map_label)
    print(df_processed['label'].value_counts())
    return df_processed

def download(url:str, dest_dir:Path, save_name:str, suffix=None) -> Path:
    assert isinstance(dest_dir, Path), "dest_dir must be a Path object"
    if not dest_dir.exists():
        dest_dir.mkdir()
    if save_name == None: filename = url.split('/')[-1]
    else: filename = save_name+suffix
    file_path = dest_dir / filename
    if not file_path.exists():
        print(f"Downloading {file_path}")
        with open(f'{file_path}', 'wb') as f:
            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length'))
            with tqdm(total=total, unit='B', unit_scale=True, desc=filename) as pbar:
                for data in response.iter_content(chunk_size=1024*1024):
                    f.write(data)
                    pbar.update(1024*1024)
    else:
        return file_path
    return file_path



def awgn(audio, snr):
    #在audio y中 添加噪声 噪声强度SNR为int
    # audio_average_db = 10 * np.log10( np.mean(audio ** 2))
    # noise_average_db = audio_average_db - snr
    # noise_average_power = 10 ** (noise_average_db / 10)
    # mean_noise = 0 
    # noise = np.random.normal(mean_noise, np.sqrt(noise_average_power), len(audio))
    # return audio + noise
    snr = 10**(snr)*1.2589
    xpower = np.sum(audio**2)/len(audio)
    noise_average_power = xpower / snr
    return audio+ np.random.normal(0, np.sqrt(noise_average_power), len(audio))
