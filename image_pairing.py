import os
import pandas as pd
import itertools

from sklearn.model_selection import train_test_split

column_names = ["sample", "image 1", "image 2", "label"]

df = pd.DataFrame(columns = column_names)


def pos_pairing(lst):
    ret = []
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            if lst[j] == lst[i]:
                pass
            else:
                ret.append((lst[i], lst[j]))
    return ret

def pos_pairing2(lst, lst2):
    return [(x, y) for x in lst for y in lst2 if not x==y]

def neg_pairing(lst, lst2):
    return [(x, y) for x in lst for y in lst2]


fake_path = './ds/fake'
real_path = './ds/real'

path_lst = [real_path, fake_path]


real_sub_dir = [name for name in os.listdir(real_path) if not  name.startswith(".")]
real_sub_dir.sort()
fake_sub_dir = [name for name in os.listdir(fake_path) if not  name.startswith(".")]
fake_sub_dir.sort()

#print(path_lst)

#for sub in path_lst:
#    sub_dir = [name for name in os.listdir(sub) if not  name.startswith(".")]
#    sub_dir.sort()

for directory in real_sub_dir:
    pos_images = [name for name in os.listdir(f'{real_path}/{directory}') if not  (name.startswith(".") or name.startswith("Thumb"))]
    neg_images = [name for name in os.listdir(f'{fake_path}/{directory}') if not  (name.startswith(".") or name.startswith("Thumb"))]
    pos_pairs = itertools.combinations(pos_images, 2)
    for i in pos_pairs:
        new_row = {'sample':directory, 'image 1':f'{real_path}/{directory}/{i[0]}', 'image 2':f'{real_path}/{directory}/{i[1]}', 'label':1}
        df = df.append(new_row, ignore_index=True)
    neg_pairs = itertools.product(pos_images,neg_images)
    for i in neg_pairs:
        new_neg_row = {'sample':directory, 'image 1':f'{real_path}/{directory}/{i[0]}', 'image 2':f'{fake_path}/{directory}/{i[1]}', 'label':0}
        df = df.append(new_neg_row, ignore_index=True)

# UNCOMMENT TO SAVE CSV
df.to_csv('ds.csv', encoding='utf-8')
df.columns = ['image1','image2','label']
y = df.label
X = df.drop('label', axis=1)
df.reset_index(drop=True, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

X =  pd.concat([X_train, y_train], axis=1)
X_test =  pd.concat([X_test, y_test], axis=1)

X.to_csv('train_data.csv', encoding='utf-8', index=False)
X_test.to_csv('test_data.csv', encoding='utf-8', index=False)
