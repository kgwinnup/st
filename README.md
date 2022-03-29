
# st

`st` is a small CLI tool for doing simple data science work. It is not designed
to replace a DS notebook, but is useful for some exploratory analysis if your
like me and spend a lot of time on the command line.

I am not 100% satisfied with the CLI interface but I also couldn't think of a
better/easier way to do it. For now the interface directed by subcommands and
within each subcommand is either a new subcommand or flags/options.

The -h/--help works after any subcommand to display that subcommands flags or
options.

```
st 0.1
stat information and processing

USAGE:
    st <SUBCOMMAND>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

SUBCOMMANDS:
    graph
    help         Prints this message or the help of the given subcommand(s)
    quintiles
    sample
    summary
    xgboost
```

## installing

Building locally requires the rust tool chain (https://rustup.rs/). 

```
cargo build --release
```

Or install it to the cargo bin directory (make sure it is in your $PATH).

```
cargo install --path .
```

# examples

```
cat tests/iris.csv | awk -F',' '{print $1}' |st summary -h
n          min        max        mean       median     mode       sd         var
150        4.3        7.9        5.8433332  5.8        5          0.8253013  0.68112224
```

Or transpose the output

```
cat tests/iris.csv | awk -F',' '{print $1}' |st summary -ht
N       150
min     4.3
max     7.9
mean    5.8433332
med     5.8
mode    5
stdev   0.8253013
var     0.68112224
```

Quick histogram and line plots

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

Simple way to get k-quintiles with the -q (5-quintile) and -Q k (where k is
user defined) flags.

```
> cat tests/iris.csv | awk -F',' '{print $1}' |st quintiles -h -q 10
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

XGBoost is also built in to `st`. A simple workflow with the iris dataset is
below.

We're going to perform a binary prediction, however, there are three classes in
this set. Since this is binary, we need to convert the three classes into two.
We could label two of the classes with the same label, or we can just remove
one of the classes. For this example, I will remove a class.

```bash
> cat tests/iris.csv |tr -d '"' | awk -F',' '{print $5}' |sort |uniq
Species
setosa
versicolor
virginica
```

This command does a number of things. The first `sed` command strips off the
header line as that gets in the way of the final random shuffle that happens.
The next few `sed` commands changes the string labels to integers that XGBoost
can understand. We are left with two categories encoded with 0 and 1. The final
command shuffles the entire dataset and prepares it for the train/test split.

```bash
cat tests/iris.csv |sed -e '1,1d' |tr -d '"' |grep -v virginica | sed -e 's/setosa/0/g' | sed -e 's/versicolor/1/g' | sed -e 's/virginica/2/g' | sort -R > tests/iris_normalized.csv

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
head -n 25 tests/iris_normalized.csv > tests/iris_test.csv
cat tests/iris_normalized.csv | sed -e '1,25d' > tests/iris_train.csv
```

Next, now that the data is cleaned, we can train the model with XGBoost using
all default parameters. The -y flag indicates which column is to be used as the
predictor value. After the model is trained and saved, we can use it on our
test set.

```bash
> cat tests/iris_train.csv | ./target/debug/st xgboost train binary -y 4 -m out.model
rmse = 0.015777


> cat tests/iris_test.csv | ./target/debug/st xgboost predict binary -m out.model
0.015754879,4.6,3.2,1.4,0.2,0
0.9842001,6.3,2.3,4.4,1.3,1
0.015754879,4.6,3.4,1.4,0.3,0
0.9842001,5.8,2.6,4,1.2,1
0.015754879,5.1,3.7,1.5,0.4,0
```

