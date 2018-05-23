# Vision

Generate the dataset:
~~~~
python3 generate_dataset.py
~~~~

Split the dataset in train and val:
~~~~
python3 split_data.py
~~~~

Train the model:
~~~~
python3 train.py
~~~~

Train the extended model which includes COCO:
~~~~
python3 train_coco.py --dataset=dataset/coco --model=coco --download=True train
~~~~
