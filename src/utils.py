import os
from typing import List
import numpy as np

def get_files(path_dir: str) -> List[tuple]:
    """ Функция для получения путей ауди из папки

    Args:
        path_dir (str): путь, где находятся аудиофайлы

    Raises:
        RuntimeError: проверяется путь на валидность, путь должен включать в себя класс аудио

    Returns:
        List[tuple]: Возвращает полный путь каждого файла и его класс
    """
    label = path_dir.split('/')[-2]
    
    if label not in ('oksana', 'omazh'):
        raise RuntimeError('not a valid path, example: ./vox-test-audio/data/oksana/')
    
    files = os.listdir(path_dir)
    files = [(path_dir+file_name, label) for file_name in files]
    return files

def get_test_files(path_dir: str) -> List[tuple]:
    """Функция для получения путей ауди из папки, отличается от get_files только способом получения классов

    Args:
        path_dir (str): путь, где находятся аудиофайлы

    Returns:
        List[tuple]: Возвращает полный путь каждого файла и его класс
    """
    files = os.listdir(path_dir)
    files = [(path_dir+file_name, file_name.split('_')[0]) for file_name in files]
    return files

def sigmoid(z: list) -> list:
    """функция для вычисления сигмоиды по предсказаниями

    Args:
        z (list): список logits

    Returns:
        list: преобразованные logits
    """
    return 1 / (1 + np.exp(-z))