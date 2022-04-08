
# st

`st` is a small tool for doing data science work at the command line. I spend a
great deal of my time ssh'ing into various servers and often need to calculate
simple statistics. Additionally, machine learning, at least exploration, can
benefit from more command line tooling. 

Largely, this is built off of my workflow, but I've tried to make it as useful
generally as possible.

One core goal with this project is to try and adhere to the Unix philosophy with
regard to text files and piping data around. Additionally, we want to do those
things quickly which is one of the reasons Rust is the language chosen for this
project.

1. [Installing](#installing)
2. [Usage](#usage)
3. [Summary statistics](#summary-statistics)
4. [K-Quintiles](#k-quintiles)
5. [Model Evaluation](#model-evaluation)
6. [XGBoost](#xgboost)
7. [Extract Features](#extract-features)
8. [Graphing](#graphing)

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
    eval         evaluation metrics to score an output, confusion matrix and other helpful probablities. Note: all
                 classes need to be 0..N
    extract      data transformations and feature generation tools
    graph        very simple cli graphing
    help         Prints this message or the help of the given subcommand(s)
    quintiles    k-quintile from a single vector (default k = 5)
    sample       sample a vector, with or without replacement
    summary      summary statistics from a single vector
    xgboost      train, predict, and understand xgboost models
```

The --help works after any subcommand to display that subcommands info,
flags, or options.

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

Simple way to get k-quintiles with the -q (5-quintile) and -Q k (where k is
user defined) flags.

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

Model evaluation is super important, and this subcommand contains some common
tools for understanding your model.

Note: All classes are assumed to labeled 0..N. This starting at 0 and growing
up is assumed in all the calculations within this section. Ensure your data is
in this format or face the panics.

Additionally, all data passed into this subcommand is expected to be a list of
line separated tuples of the form `predicted, actual`. Again `actual` must be
0..N. Predicted in this case is an int or a (0,1) value. In the case of a (0,1)
range, this is rounded at 0.5 up or down to the nearest int. To specify the
threshold use the `-t` flag.

```bash
> st eval iris_results.csv
-       0       1       2
0       7       1       0
1       0       9       0
2       0       0       8
```

There is a `-v` flag which will provide the TPR and FPR rates for each class.

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

For binary and softmax objective functions. There is also a Bayes estimator of
the models effective performance. This requires passing in a list of base rates
of occurrence for a specific class. The length of the list much match the number
of classes. 

Apply Bayes formula, given a natual rate of occurance for the target class. In
the example below, the natural rate of class_1 is very low. This answers the
question: if the model predicts class_N, what is the probability that the
predicted input is of class_N.

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
> cat tests/iris_train.csv | st xgboost train -n 3 -y 4 -m out.model -o multi:softmax
merror = 0.008
```

Now we can use the model to predict some values. Get the test set and use the
predict subcommand. The predicted value for the test set is added as the first
column of the output.

```bash
> cat tests/iris_test.csv | st xgboost predict -m out.model
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
> st xgboost importance -t gain out.model
f2 = 0.49456635
f3 = 0.4888009
f1 = 0.011651897
f0 = 0.004981032
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

## Graphing

Quick histogram and line plots.

```
> cat tests/iris.csv | awk -F',' '{print $1}' |st graph -h -t histo
 62.00 ┤                      ╭──╮
 59.60 ┤                     ╭╯  ╰───────╮
 57.20 ┤                   ╭─╯           ╰───────╮
 54.80 ┤                  ╭╯                     ╰────╮
 52.40 ┤                ╭─╯                           ╰╮
 50.00 ┤               ╭╯                              ╰─╮
 47.60 ┤              ╭╯                                 ╰╮
 45.20 ┤            ╭─╯                                   ╰╮
 42.80 ┤           ╭╯                                      ╰─╮
 40.40 ┤         ╭─╯                                         ╰╮
 38.00 ┤        ╭╯                                            ╰╮
 35.60 ┤      ╭─╯                                              ╰─╮
 33.20 ┤     ╭╯                                                  ╰╮
 30.80 ┤    ╭╯                                                    ╰─╮
 28.40 ┤  ╭─╯                                                       ╰╮
 26.00 ┤ ╭╯                                                          ╰╮
 23.60 ┼─╯                                                            ╰─╮
 21.20 ┤                                                                ╰╮
 18.80 ┤                                                                 ╰╮
 16.40 ┤                                                                  ╰─╮
 14.00 ┤                                                                    ╰

cat tests/iris.csv | awk -F',' '{print $1}' |st graph -h -t line
 7.31 ┤                                                           ╭╮
 7.16 ┤                                                     ╭╮    ││
 7.01 ┤                                                     ││    ││
 6.86 ┤                       ╭╮                            ││  ╭╮│╰╮
 6.71 ┤                       ││         ╭╮             ╭╮  ││  │││ │  ╭╮╭╮
 6.57 ┤                       ││         ││   ╭╮        ││  ││  │││ │  ││││
 6.42 ┤                       │╰╮   ╭╮   ││   ││     ╭─╮│╰─╮│╰╮ │││ │  │││╰╮
 6.27 ┤                      ╭╯ │   ││  ╭╯│   ││     │ ││  ││ │╭╯╰╯ │╭╮│╰╯ │
 6.12 ┤                      │  ╰─╮╭╯│ ╭╯ ╰╮  ││     │ ││  ╰╯ ││    ╰╯╰╯   │
 5.97 ┤                      │    ││ │ │   │  ││╭╮ ╭╮│ ││     ││           │
 5.82 ┤                      │    ││ ╰╮│   │╭╮││││ │╰╯ ││     ││           ╰
 5.67 ┤      ╭╮              │    ││  ╰╯   │││││││╭╯   ││     ╰╯
 5.52 ┤      ││              │    ││       ╰╯╰╯╰╯││    ││
 5.37 ┤      ││      ╭╮      │    ╰╯             ││    ╰╯
 5.22 ┤      │╰─╮  ╭╮││      │                   ││
 5.07 ┼╮╭╮   │  │  ││││╭╮ ╭╮ │                   ╰╯
 4.92 ┤│││ ╭╮│  │╭─╯╰╯╰╯╰╮│╰─╯
 4.77 ┤││╰─╯││  ╰╯       ││
 4.62 ┤╰╯   ││           ││
 4.47 ┤     ││           ╰╯
 4.32 ┤     ╰╯
```



