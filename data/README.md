# data/

Put the unzipped dataset files here:

- `train.csv`
- `test.csv`

`full_data_flightdelay.csv` and the `raw_data/` folder are not needed to train or run the app.
They are the source the train and test files were built from.

`make_demo_data.py` writes `demo_train.csv` and `demo_test.csv` here. Those are synthetic and only
exist to smoke-test the pipeline.
