import glob
import os
import random
import tqdm
from PIL import Image

base_dir = './part1/Dataset_Part1'
target_dir = './dataset'

filenames = sorted(glob.glob(os.path.join(base_dir, '*/*.JPG'))
filenames = list(filter(lambda x: 'Label' not in x, filenames))

os.makedirs(os.path.join(target_dir, 'train', exist_ok=True)
os.makedirs(os.path.join(target_dir, 'test', exist_ok=True)

nb_train = int(len(filenames) * 0.8)

random.shuffle(filenames)

train = filenames[:nb_train]
test = filenames[nb_train:]

for i, f in enumerate(tqdm.tqdm(train)):
    img = Image.open(f).convert('RGB')
    img = img.resize((512, 512))
    filename = os.path.join(target_dir, 'train/{0:08d}.jpg'.format(count))
    img.save(filename)

for i, f in enumerate(tqdm.tqdm(test)):
    img = Image.open(f).convert('RGB')
    img = img.resize((512, 512))
    filename = os.path.join(target_dir, 'test/{0:08d}.jpg'.format(count))
    img.save(filename)
