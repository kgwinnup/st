
# st

`st` is a small cli tool for doing simple data science work. It is not designed
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

building locally requires the rust toolchain (https://rustup.rs/). 

```
cargo build --release
```

or install it to the cargo bin directory (make sure it is in your $PATH).

```
cargo install --path .
```

# examples

```
cat tests/iris.csv | awk -F',' '{print $1}' |st summary -h
n          min        max        mean       median     mode       sd         var
150        4.3        7.9        5.8433332  5.8        5          0.8253013  0.68112224
```

or transpose the output

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

quick histogram and line plots

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

simple way to get k-quintiles with the -q (5-quintile) and -Q k (where k is user defined) flags.

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

