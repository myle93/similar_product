# similar_product
## Generate training data
Use `generate_training_data.ipynb` and run the code as instructed in the notebook.  
Instead you can use our example dataset [TrainingData](https://github.com/myle93/similar_product/tree/master/ditto/data/amazon).

## Train Ditto
If you generated your own training data, first you have to specify the path to these data in the [Config file](https://github.com/myle93/similar_product/blob/master/ditto/configs.json).  
Train the model as instructed in [Train Ditto](https://github.com/myle93/similar_product/blob/master/ditto/README.md)


## Train Ditto with different hyperparameters
To train and test Ditto model with different hyperparameters (e.g. different language models, sentence length, ...) you can use the command lines in [Config Ditto](https://github.com/myle93/similar_product/blob/master/table_2.md).  
Remember to change the value of the `--task` argument, if you use your own generated dataset.

## Train Siamese models
Use `table_3.ipynb` and run the code as instructed in the notebook.  
Instead you can skip this step and use our trained model [Siamese models](https://github.com/myle93/similar_product/tree/master/Model).

## Train and test multi-modal models
In this step you will combine the Ditto and Siamese model and test the combined one on test data.
Use `table_1.ipynb` and run the code as instructed in the notebook.  

## Visualize the result
Use `Figure_2.ipynb` and run the code as instructed in the notebook.  