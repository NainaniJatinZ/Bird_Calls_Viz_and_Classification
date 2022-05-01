import numpy as np # linear algebra
import pandas as pd 
import streamlit as st
import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
from icrawler.builtin import GoogleImageCrawler
import os
import glob
from PIL import Image

st.title('Bird Call Classification and Visualization')

path_ex = False
uploaded_file = st.file_uploader("Choose an Audio Bird call sample")
if uploaded_file is not None:
     # To read file as bytes:
    #  bytes_data = uploaded_file.getvalue()
    #  st.write(bytes_data)
    with open(os.path.join("inputs", uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())
    path3 = "inputs/"+uploaded_file.name
    st.write(path3)
    path_ex = True


class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    
    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return ((resig, sr))
    
    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)
    
    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
    
    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec


def preprocess_audio(original_file_paths):
    for path in original_file_paths:
        # open audio
        audio = AudioUtil.open(path)  
        print(audio[1])
        #print(audio[0].shape)
        
        # rechannel into 2 channels
        rechannel = AudioUtil.rechannel(audio, 2)
        #print(rechannel[0].shape)
        
        # pad and/or truncate so audio is same length 
        trunc = AudioUtil.pad_trunc(rechannel, 10000)
        #print(trunc[0].shape)
        
        # time shift audio some random percentage 
        shifted = AudioUtil.time_shift(trunc, 100)
        #print(shifted[0].shape)
        
        # create and modify spectrogram to avoid overfitting
        spectro_gram = AudioUtil.spectro_gram(shifted)
        spectro_augment = AudioUtil.spectro_augment(spectro_gram)
        return spectro_augment


class SoundDS(Dataset):
    def __init__(self, df):
        self.df = df
        self.duration = 10000
        self.sr = 32000
        self.channel = 2
        self.shift_pct = 0.4
        self.sgram = []

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)    

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.df.loc[idx, 'file_path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'label']

        aud = AudioUtil.open(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        rechan = AudioUtil.rechannel(aud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        self.sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(self.sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id

class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=21)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x



labels = ['akiapo', 'aniani', 'apapan', 'barpet', 'elepai', 'ercfra',
       'hawama', 'hawcre', 'hawgoo', 'hawhaw', 'hawpet1', 'houfin',
       'iiwi', 'jabwar', 'maupar', 'omao', 'puaioh', 'skylar', 'warwhe1',
       'yefcan']
le = LabelEncoder()
le.fit(np.unique(labels))                        # need this


if path_ex:
    path1 = "inputs\XC138797.ogg"
    path2 = "inputs\XC140587.ogg"
    test_df = pd.DataFrame({'file_path': [path1, path2, path3], 'label' : [0, 0, 0]})
    st.write("Uploaded Audio Sample: ")
    audio_file = open(path3, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')

    aud = AudioUtil.open(path3)
    rechan = AudioUtil.rechannel(aud, 2)
    dur_aud = AudioUtil.pad_trunc(rechan, 10000)
    shift_aud = AudioUtil.time_shift(dur_aud, 0.4)
    sgram1 = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    st.write("Mel-Spectogram of Sample: ")
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(sgram1.numpy(), (1,2,0))[:,:,0])
    st.pyplot(fig)


    model2 = AudioClassifier()
    model2.load_state_dict(torch.load("model\saved_model (1).pth", map_location=torch.device('cpu')))

    myds = SoundDS(test_df)
    test_dl = torch.utils.data.DataLoader(myds, batch_size=16, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model2 = model2.to(device)

    for data in test_dl:
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Get predictions
        outputs = model2(inputs)
        # st.write(outputs)
        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        print(prediction)
        # st.write(prediction.numpy()[-1])


    final_pred = le.inverse_transform([prediction.numpy()[-1]])
    print(final_pred)
    tax = pd.read_csv("data\eBird_Taxonomy_v2021.csv")


    prim_name = tax.loc[tax['SPECIES_CODE']==str(final_pred[0])]['PRIMARY_COM_NAME']


    st.write("Predicted Species of the audio sample: "+prim_name.values[0])
    query = prim_name.values[0] + ' bird'


    import os, shutil
    folder = 'downloads'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))



    google_Crawler = GoogleImageCrawler(storage = {'root_dir': r'downloads'})
    google_Crawler.crawl(keyword = query, max_num = 3)
    resizedImages = []
    for i in range(1,4):
        image = Image.open('downloads/00000'+str(i)+".jpg")
        resizedImg = image.resize((225, 325), Image.ANTIALIAS)
        resizedImages.append(resizedImg)

    st.image(resizedImages)


    
