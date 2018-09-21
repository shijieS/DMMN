'''
author: shijie Sun
email: shijieSun@chd.edu.cn
'''
import argparse
import os
import scipy.io as scio
import numpy as np
from tqdm import trange
from tqdm import tqdm

print('''
Usage: ua_convert_mat_2_mot --ua_root="UA-DETRAC root"
''')

parser = argparse.ArgumentParser(description='convert all the DETRAC-Train-Annotations-MAT format to mot17 format')
parser.add_argument('--ua_root', default="/home/shiyuan/ssj/dataset/UATRAC", help='UA-DETRAC data set root directory, such as ua, we will create one directory called gt')

args = parser.parse_args()

class ConvertMat2UA:
    @staticmethod
    def run(mat_folder, save_folder):
        if not os.path.exists(mat_folder):
            raise FileNotFoundError('cannot find file ' + mat_folder)

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            print('create {}'.format(save_folder))

        print('================= read files =================')
        print('search mat')

        mat_files_name = [f for f in os.listdir(mat_folder) if os.path.splitext(f)[1] == '.mat']
        mat_files_full_name = [os.path.join(mat_folder, f) for f in mat_files_name]

        t = trange(len(mat_files_full_name))
        for f, i, in zip(mat_files_full_name, t):
            t.set_description('process: {}'.format(os.path.basename(f)))
            file_name = os.path.join(save_folder, os.path.splitext(os.path.basename(f))[0]+'.txt')

            mat = scio.loadmat(f)['gtInfo'][0][0]

            X = mat[0]
            Y = mat[1]
            H = mat[2]
            W = mat[3]

            res = []
            for trackId, (xc, yc, hc, wc) in enumerate(zip(X.T, Y.T, H.T, W.T)):
                for frameId, (x, y, h, w) in enumerate(zip(xc, yc, hc, wc)):
                    if x != 0 and y != 0 and h!=0 and w!=0:
                        res += [[frameId, trackId, x-w/2.0, y-h, x+w/2.0, y]]

            res = np.array(res)

            np.savetxt(file_name, res, delimiter=',', fmt="%d,%d,%1.2f,%1.2f,%1.2f,%1.2f")

        print('=================Finished! Well Done=================')


if __name__ == '__main__':

    mat_folder = os.path.join(args.ua_root, 'DETRAC-Train-Annotations-MAT')
    save_folder = os.path.join(args.ua_root, 'DETRAC-Train-Annotations-MOT')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    ConvertMat2UA.run(mat_folder, save_folder)



