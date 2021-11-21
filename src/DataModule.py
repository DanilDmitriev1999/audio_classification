import librosa
import sklearn
import random
import torch
import numpy as np
from typing import List
from torch.utils.data import Dataset


class AudioPreProcessing:
    def __init__(self, sr:int = 16000):
        self.sr = sr
    
    def open(self, file: str) -> tuple:
        """Метод для чтения аудио

        Args:
            file (str): путь к аудио

        Returns:
            tuple: x - Time Domain Features, sr - sample rate
        """
        x, sr = librosa.load(file, self.sr)
        return (x, sr)
    
    @staticmethod
    def specto_gram(audio: np.ndarray, sr:int=16000, n_fft:int=2048, hop_length:int=1024) -> np.ndarray: 
        """Из Time Domain Features генерирует Спектограмму Мела

        Args:
            audio (np.ndarray): Time Domain Features
            sr (int, optional): sample rate. Defaults to 16000.
            n_fft (int, optional): длина окна для каждого временного отрезка. Defaults to 2048.
            hop_length (int, optional): количество сэмплов, по которым сдвигается окно на каждом шаге.. Defaults to 1024.

        Returns:
            np.ndarray: Спектограмма Мела
        """
        sgram = librosa.stft(audio)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        return mel_sgram
    
    @staticmethod
    def mfcc(audio: np.ndarray, sr:int=16000, n_fft:int=2048, hop_length:int=1024) -> np.ndarray:
        """Из Time Domain Features генерирует MFCC (Mel Frequency Cepstral Coefficients)

        Args:
            audio (np.ndarray): Time Domain Features
            sr (int, optional): sample rate. Defaults to 16000.
            n_fft (int, optional): длина окна для каждого временного отрезка. Defaults to 2048.
            hop_length (int, optional): количество сэмплов, по которым сдвигается окно на каждом шаге.. Defaults to 1024.

        Returns:
            np.ndarray: MFCC
        """
        mfcc = librosa.feature.mfcc(audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
        return mfcc
    
    @staticmethod
    def time_shift(audio: np.ndarray, shift: int) -> np.ndarray:
        """Аугментация: сдвигает аудио на некое порогове значение

        Args:
            audio (np.ndarray): Time Domain Features
            shift (int): на сколько надо сдвинуть

        Returns:
            np.ndarray: Time Domain Features со смещение
        """
        x_roll = np.roll(audio, shift)
        return x_roll
    
    @staticmethod
    def white_noise(audio: np.ndarray, percent: float = 0.01) -> np.ndarray:
        """Аугментация: Добавляет шума

        Args:
            audio (np.ndarray): Time Domain Features
            percent (float, optional): процент шума. Defaults to 0.01.

        Returns:
            np.ndarray: Time Domain Features с шумом
        """
        wn = np.random.randn(len(audio))
        x_wn = audio + percent * wn
        return x_wn
    
    @staticmethod
    def stretching(audio: np.ndarray, percent: float = 2.0) -> np.ndarray:
        """Аугментация: Ускорение, если percent, иначе замедление

        Args:
            audio (np.ndarray): Time Domain Features
            percent (float, optional): процент изменения скорости. Defaults to 2.0.

        Returns:
            np.ndarray: Time Domain Features с изменной скоростью
        """
        x_stretch = librosa.effects.time_stretch(audio, percent)
        return x_stretch
    
    def pipeline(self, files: List[str], augmentation: bool, n_augm: int=1) -> np.ndarray:
        """pipeline для обработки группы аудиофайлов. Сначала он загружает файлы без аугментации, а потом если 
           в параметр augmentation передали True проводит n_augm раз случайных аугментаций

        Args:
            files (List[str]): список путей и классов к файлам
            augmentation (bool): Делать ли аугметацию
            n_augm (int, optional): Сколько раз аугментировать исхлдные данные. Defaults to 1.

        Returns:
            np.ndarray: [description]
        """
        train_audio = []
        for audio_path, label in files:
            audio, _ = self.open(audio_path)
            mfcc = self.mfcc(audio)
            train_audio.append((mfcc, label))
        
        if augmentation:
            for _ in range(n_augm):
                for audio_path, label in files:
                    if (label == 'oksana' and random.random() > 0.5) or label == 'omazh':
                        audio, _ = self.open(audio_path)
                        shift = np.random.randint(5000, 25000)
                        if np.random.random() > 0.5:
                            audio = self.time_shift(audio, shift)
                            audio = self.white_noise(audio, 0.01)
                        stretch_percent = random.uniform(1.0, 3.0)
                        audio = self.stretching(audio, stretch_percent)

                        mfcc = self.mfcc(audio)
                        train_audio.append((mfcc, label))
        return train_audio

class AudioDataset(Dataset):
    def __init__(self,
                 data: List[tuple],
                 sr: int = 16000,
                 ):
        self.data = data
        self.sr = sr
    
    @staticmethod
    def prepare_label(label):
        if label == 'oksana':
            return 1
        return 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_img, label = self.data[idx]
        
        
        label = self.prepare_label(label)
        lenght = audio_img.shape[1]
        
        return audio_img, label, lenght