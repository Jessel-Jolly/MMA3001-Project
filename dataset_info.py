import glob
import warnings

import numpy as np

numbers_to_check = [0,1,2,3,4]
files_to_check = ["train", "valid", "test", "*"]

def get_instances(numbers, file):
    labels = ["Loose Meat:\t\t","Packaging Error:\t","Twisted Meat:\t\t","Unsealed:\t\t","Wrinkle:\t\t","Undamaged:\t\t"]
    names = glob.glob(f"Pork Rasher Error-Packaging-.v4i.yolov11/{file}/labels/*")
    count = np.zeros(len(numbers)+1, dtype=int)
    images = np.zeros(len(numbers)+1, dtype=int)
    numbers = np.array(numbers, dtype=str)
    for txt in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = np.loadtxt(txt, dtype=str).flatten()
            for index, number in enumerate(numbers):
                add = 1
                if data.size != 0:
                    for i in data:
                        if i == number:
                            count[index] += 1
                            images[index] += add
                            add = 0
                else:
                    count[-1] += 1
                    images[-1] += 1
    print('\n================================================================')
    print(f'\t\t\tFile: {file}')
    print(f'\t\t\tTotal Images: {len(names)}\n')
    print("   Type\t\t\t   Total Instances\t  Number of Images")
    for index, number in enumerate(numbers):
        print(f"{labels[int(number)]}\t{count[index]}\t\t\t{images[index]}")
    print('================================================================\n')

for f in files_to_check:
    get_instances(numbers_to_check, f)

