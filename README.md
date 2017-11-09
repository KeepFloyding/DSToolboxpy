# DSToolboxpy
A data science toolbox useful for reading, manipulating and analyzing data using python.

It contains an easy and quick manner of transforming clickstream data into a machine learning format with the prepareTimesSeries function.
It also has an sklearn classifier can be easily tested through k-folds cross-validation using the testClassifier function. 

## Getting started

### Prerequisites

* pandas
* numpy
* sklearn

### Installing

The library can be added as a module to python using 

```
python setup.py install 
```

### Testing 

Now you should be able to import the library with 

```
import DSToolbox
```

## Running

A sample dataset has been provided that contains user_ids, clickstream actions, dates of the events and their registration date. Running testToolbox.py will import the dataset, manipulate it and save it as csvs after running each function. 

First the users and their actions in the dataset will be grouped by discretised event legs. Then the data will be manipulated again to assig the actions in each event_leg as seperate features. 

This can then be used with the testClassifier function to test a machine learning algorithm on that dataset. 







