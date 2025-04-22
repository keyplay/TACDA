# Target-specific Adaptation and Consistent Degradation Alignment for Cross-Domain Remaining Useful Life Prediction

This is a PyTorch implementation of this domain adaptation method for remaining useful Life prediction on time series data.


## Requirmenets:
- Python3.x
- Pytorch==1.7
- Numpy
- Sklearn
- Pandas

## Datasets
### Download Datasets
We used NASA turbofan engines dataset
- [CMAPPS](https://catalog.data.gov/dataset/c-mapss-aircraft-engine-simulator-data)

### Prepare Datasets
- run_the data/data_preprocessing.py to apply the preprocessings.
- Output the data form each domain in tuple format train_x, train_y, test_x, test_y
- Put the data in the data folder

## Train the model
To pre train model:

```
python pretrain_main.py 
```
To train model for 1st round domain adaptation:

```
python main_cross_domains.py     
```

To train model for 2nd round domain adaptation:

```
python main_cross_domains_two_step.py     
```


