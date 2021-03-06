
# st

`st` is a small tool for doing data science work at the command
line. One core goal is to adhere to Unix principles regarding
input/output. This tool is intended for use in Unix pipelines.

I spend a lot of time at the command line doing DS work, and this tool
largely replaces many of the simple scripts I used to use.

1. [Installing](#installing)
2. [Usage](#usage)
3. [Summary statistics](#summary-statistics)
4. [K-Quintiles](#k-quintiles)
5. [Model Evaluation](#model-evaluation)
6. [XGBoost](#xgboost)
7. [Correlation Matrix](#correlation-matrix)
8. [Extract Features](#extract-features)

# Installing

Building locally requires the rust tool chain (https://rustup.rs/).

When using cargo to install, make sure ~/.cargo/bin is in your $PATH.

## OS X

```
> # install rustup toolchain

> # haven't confirmed, I think you'll need
> brew install libomp
> cargo install --path .
```

## Debian (probably Ubuntu too)

```
> # install rustup toolchain

# this is needed for xgboost
> sudo apt-get install llvm-dev libclang-dev clang
> cargo install --path .
```

# Usage

```
> st --help
st 0.1
stat information and processing

USAGE:
    st <SUBCOMMAND>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

SUBCOMMANDS:
    cor-matrix    Computes the Pearson correlation coefficient
    eval          evaluation metrics to score an output, confusion matrix and other helpful probablities. Note: all
                  classes need to be 0..N
    extract       data transformations and feature generation tools
    graph         very simple cli graphing
    help          Prints this message or the help of the given subcommand(s)
    quintiles     k-quintile from a single vector (default k = 5)
    summary       summary statistics from a single vector
    xgb           train, predict, and understand xgboost models
```

The --help works after any subcommand to display that subcommands
info, flags, or options.

## Summary statistics

```
> cat tests/iris.csv | awk -F',' '{print $1}' |st summary -h
n          min        max        mean       median     mode       sd         var
150        4.3        7.9        5.8433332  5.8        5          0.8253013  0.68112224
```

Or transpose the output.

```
> cat tests/iris.csv | awk -F',' '{print $1}' |st summary -ht
N       150
min     4.3
max     7.9
mean    5.8433332
med     5.8
mode    5
stdev   0.8253013
var     0.68112224
```

## k-quintiles

Simple way to get k-quintiles with the -q (5-quintile) and -Q k (where
k is user defined) flags.

```
> cat tests/iris.csv | awk -F',' '{print $1}' |st quintiles -h -k 10
10%      4.8
20%      5
30%      5.3
40%      5.6
50%      5.8
60%      6.1
70%      6.3
80%      6.6
90%      6.9
```

## Model Evaluation

Model evaluation is super important, and this subcommand contains some
common tools for understanding your model.

Note: All classes are assumed to labeled 0..N. This starting at 0 and
growing up is assumed in all the calculations within this
section. Ensure your data is in this format or face the panics.

Additionally, all data passed into this subcommand is expected to be a
list of line separated tuples of the form `predicted, actual`. Again
`actual` must be 0..N. Predicted in this case is an int or a (0,1)
value. In the case of a (0,1) range, this is rounded at 0.5 up or down
to the nearest int. To specify the threshold use the `-t` flag.

```bash
> st eval iris_results.csv
-       0       1       2
0       7       1       0
1       0       9       0
2       0       0       8
```

There is a `-v` flag which will provide the TPR and FPR rates for each
class.

```bash
> st eval -v iris_results.csv
-       2       1       0
2       7       1       0
1       0       9       0
0       0       0       8

class   tpr     fpr     tnr     fnr
0       0.875   0.000   0.944   0.125
1       1.000   0.062   1.000   0.000
2       1.000   0.000   1.000   0.000
```

For binary and softmax objective functions. There is also a Bayes
estimator of the models effective performance. This requires passing
in a list of base rates of occurrence for a specific class. The length
of the list much match the number of classes.

Apply Bayes formula, given a natual rate of occurance for the target
class. In the example below, the natural rate of class_1 is very
low. This answers the question: if the model predicts class_N, what is
the probability that the predicted input is of class_N.

```bash
> st eval -v -b '0.99,0.01' results.csv
-       1       0
1       313     7
0       42      338

class   tpr     fpr     tnr     fnr
0       0.978   0.111   0.980   0.022
1       0.889   0.022   0.882   0.111

0: Pr(class_0 | positive) = 0.99885994
1: Pr(class_1 | positive) = 0.291144
```

A ROC curve in table form. For this the expected input is a list of
tuples of `prediction, actual` where prediction is a range (0,1).

```bash
> st eval --table results.csv
-       1       0
1       57492   2465
0       2508    57535

t       prec    f1      tpr     fpr
0.05    0.7785  0.9975  0.8745  0.2837
0.10    0.8343  0.9944  0.9074  0.1974
0.15    0.8688  0.9910  0.9259  0.1497
0.20    0.8922  0.9876  0.9375  0.1193
0.25    0.9087  0.9843  0.9449  0.0989
0.30    0.9215  0.9797  0.9498  0.0834
0.35    0.9330  0.9748  0.9534  0.0701
0.40    0.9429  0.9695  0.9560  0.0587
0.45    0.9516  0.9641  0.9578  0.0491
0.50    0.9589  0.9582  0.9585  0.0411
0.55    0.9653  0.9521  0.9587  0.0342
0.60    0.9714  0.9442  0.9576  0.0278
0.65    0.9760  0.9355  0.9553  0.0230
0.70    0.9808  0.9246  0.9519  0.0181
0.75    0.9852  0.9108  0.9465  0.0137
0.80    0.9898  0.8916  0.9381  0.0092
0.85    0.9935  0.8628  0.9235  0.0057
0.90    0.9962  0.8167  0.8976  0.0032
0.95    0.9985  0.7174  0.8349  0.0011
```

## XGBoost

XGBoost is built in to `st`. A simple workflow with the iris dataset is below.
XGBoost is usually the first model I start with when analyzing a dataset. Even
if XGBoost isn't the final model I'll be using in production, it is super easy
to train and most of all, interpret (well trees in general).

We're going to perform a binary prediction, however, there are three classes in
this set. So we need to ensure we're using a multi-class predictive objective.

```bash
> cat tests/iris.csv |sed -e '1,1d' |tr -d '"' | awk -F',' '{print $5}' |sort |uniq
setosa
versicolor
virginica
```

This command does a number of things. The first `sed` command strips off the
header line as that gets in the way of the final random shuffle that happens.
The next few `sed` commands changes the string labels to integers that XGBoost
can understand. We are left with two categories encoded with 0, 1, and 2. The
final command shuffles the entire dataset and prepares it for the train/test
split.

```bash
> cat tests/iris.csv |sed -e '1,1d' |tr -d '"' | sed -e 's/setosa/0/g' | sed -e 's/versicolor/1/g' | sed -e 's/virginica/2/g' | sort -R > tests/iris_normalized.csv

> head tests/iris_normalized.csv
5.1,3.5,1.4,0.2,0
4.9,3,1.4,0.2,0
4.7,3.2,1.3,0.2,0
4.6,3.1,1.5,0.2,0
```

Split the dataset into a training set and a testing set. Since the dataset is
random, we can just take the first N lines for the test, and the remainder as
the training set.

```bash
> head -n 25 tests/iris_normalized.csv > tests/iris_test.csv
> cat tests/iris_normalized.csv | sed -e '1,25d' > tests/iris_train.csv
```

Next, now that the data is cleaned, we can train the model with XGBoost using
all default parameters. The -y flag indicates which column is to be used as the
predictor value. After the model is trained and saved, we can use it on our
test set.

Training parameters can be tuned, such as eta and max depth. See `st xgboost
train --help` for more options.

```bash
> cat tests/iris_train.csv | st xgb train -n 3 -y 4 -m out.model -o multi:softmax
merror = 0.008
```

Now we can use the model to predict some values. Get the test set and use the
predict subcommand. The predicted value for the test set is added as the first
column of the output.

```bash
> cat tests/iris_test.csv | st xgb predict -m out.model
1,7,3.2,4.7,1.4,1
1,6.6,3,4.4,1.4,1
1,5,2.3,3.3,1,1
1,5.6,3,4.1,1.3,1
2,6.7,3.3,5.7,2.1,2
0,4.3,3,1.1,0.1,0
0,5.4,3.9,1.3,0.4,0
```

Tree based models are great for understanding the results. You can use the
"importance" subcommand to try and understand the model and how specific
features impact the model. All features are labeled "fx" where "x" is the
column number.


```bash
> st xgb importance -t gain out.model
f2 = 0.49456635
f3 = 0.4888009
f1 = 0.011651897
f0 = 0.004981032
```

## Correlation Matrix

Computes the Pearson correlation coefficient matrix. In the example
below, the `-y` flag is used because the CSV file still contains the
string labels in column 4.

```bash
> st cor-matrix -y 4 tests/iris_cleaned.csv
-       0       1       2       3
0:      1.00
1:      -0.12   1.00
2:      0.87    -0.43   1.00
3:      0.82    -0.37   0.96    1.00
```


## Extract Features

Frequently, a normalized byte histogram is desired from some input. This will
output a 256 sized series to stdout, where each index is the decimal mapping
for that specific byte.

```bash
> cat README.md | st extract byte-histogram
0,0,0,0,0,0,0,0,0,0,0.027048063825647013, ...
```

Reducing the dimensionality of the data using the hash-trick is built in under
the extract subcommand. Use the -F flag to set the delimiter.

```bash
> echo 'foo,bar,baz,raw,norm,etc' | st extract hash-trick -k 10 -b
0,0,1,1,0,0,1,1,0
```

Sometimes it is useful to use bit entropy as a feature.

```bash
> st extract entropy /bin/bash
6.337345458355953
```

