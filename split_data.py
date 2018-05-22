import argparse
import random
import json
import os

parser = argparse.ArgumentParser(description='Splitting json dataset')
parser.add_argument('--dataset', dest='dataset', type=str, required=True, help='path of the dataset json file')
parser.add_argument('--valdim', dest='valdim', type=float, default=0.2, help='dimension of the validation set')
parser.add_argument("--trainvaldir", dest="traindir", type=str, default="dataset/trainval/", help="Trainval folder")
args = parser.parse_args()

dataset = args.dataset
valdim = args.valdim
trainvaldir = args.traindir

assert (valdim <= 1.0 and valdim >= 0.0)
assert (os.path.isfile(dataset))

if not os.path.exists(trainvaldir):
    os.makedirs(traindir)

with open(dataset, 'r') as dfile:
	print('Reading json dataset')
	json_data = json.load(dfile)
	json_data = list(json_data.items())
	random.shuffle(json_data)

	val_size = int(len(json_data) * valdim)

	val_data= dict(json_data[:val_size])
	train_data = dict(json_data[val_size:])

	with open(trainvaldir + 'train.json', 'w') as train_file:
		print('Writing training dataset file')
		json.dump(train_data, train_file)

	with open(trainvaldir + 'val.json', 'w') as val_file:
		print('Writing validation dataset file')
		json.dump(val_data, val_file)

print("Operation completed")
