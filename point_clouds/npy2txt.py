import numpy as np
import os

def npy2txt(npy_path, txt_path):
    data = np.load(npy_path)
    np.savetxt(txt_path, data)

if __name__ == '__main__':
    for dirname in os.listdir('.'):
        if os.path.isdir(dirname):
            txt_dir = os.path.join(dirname, 'txt')
            os.makedirs(txt_dir, exist_ok=True)
            for filename in os.listdir(dirname):
                if filename.endswith('.npy'):
                    npy_path = os.path.join(dirname, filename)
                    txt_path = os.path.join(txt_dir, filename.replace('.npy', '.txt'))
                    npy2txt(npy_path, txt_path)
