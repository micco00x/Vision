# Vision

Generate the dataset:
~~~~
python3 generate_dataset.py
~~~~

Split the dataset in train and val:
~~~~
python3 split_data.py --dataset=dataset/trainval/dataset.json
~~~~

Train the model:
~~~~
python3 activity.py train
~~~~

Train the extended model which includes COCO (note that there's no need to type
`--download=True` if the COCO dataset has already been downloaded previously):
~~~~
python3 activity.py train --extended=True --download=True
~~~~

Evaluate the last trained model on the extended dataset:
~~~~
python3 activity.py evaluate --extended=True --model=last
~~~~
