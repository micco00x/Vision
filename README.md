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
python3 activity.py train
~~~~

Train the extended model which includes COCO (note that there's no need to type
`--download=True` if the model has already been downloaded previously):
~~~~
python3 activity.py train --extended=True --download=True
~~~~
