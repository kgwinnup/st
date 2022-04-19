use murmur3::murmur3_32;
use series;
use std::io::prelude::*;
use std::io::Cursor;
use std::path::PathBuf;
use structopt::StructOpt;
use xgb;

fn print_quintiles(input: &mut [f64], k: u32) {
    if input.len() < (k as usize) {
        eprintln!(
            "insufficient data for {} quintiles, data has only {} rows",
            k,
            input.len()
        );
        return;
    }

    input.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut ks = vec![];
    let size = input.len() / (k as usize);

    for i in 1..k {
        ks.push(input[(i as usize) * size] as f32);
    }

    for i in 0..k - 1 {
        let perc = (((i + 1) * size as u32) as f32) / input.len() as f32;
        let perc_format = format!("{}%", (perc * 100.0) as u32);
        println!("{:<8} {:<8}", perc_format, ks[i as usize]);
    }
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "st",
    about = "stat information and processing",
    version = "0.1"
)]
struct Opt {
    #[structopt(subcommand)]
    cmd: Command,
}

#[derive(StructOpt, Debug)]
enum XgbOptions {
    #[structopt(about = "train a new binary or multiclass model")]
    Train {
        #[structopt(short, long, help = "predictor column")]
        ycol: usize,

        #[structopt(short, long, help = "max depth", default_value = "6")]
        depth: u32,

        #[structopt(short, long, help = "eta", default_value = "0.3")]
        eta: f32,

        #[structopt(short = "m", long, help = "path to save model")]
        model_out: String,

        #[structopt(
            short,
            long,
            help = "objective function: binary:logistic, multi:softmax, multi:softprob"
        )]
        objective: String,

        #[structopt(
            short,
            long,
            help = "number of classes to predict",
            default_value = "1"
        )]
        nclasses: u32,

        #[structopt(
            short,
            long,
            help = "how many boosted rounds, default is 10",
            default_value = "10"
        )]
        rounds: u32,

        #[structopt(short = "h", long = "with-header", help = "with header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(about = "use an xgboost model against some input")]
    Predict {
        #[structopt(short, long, help = "predictor column", default_value = "1000000")]
        ycol: usize,

        #[structopt(short, long, help = "path to model")]
        model_in: String,

        #[structopt(short = "h", long = "with-header", help = "with header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(about = "dump model importance statistics, and text version of the model itself")]
    Importance {
        #[structopt(short = "d", help = "dump model in text format")]
        dump_model: bool,

        #[structopt(
            short,
            long = "type",
            help = "importance type: gain, cover, freq",
            default_value = "gain"
        )]
        typ: String,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },
}

#[derive(StructOpt, Debug)]
enum ExtractOptions {
    #[structopt(about = "create a normalized byte histogram of the input")]
    ByteHistogram {
        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(about = "calculate the bits of entropy of the input")]
    Entropy {
        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(
        about = "given a comma separted list of strings, apply the hash-trick with k buckets"
    )]
    HashTrick {
        #[structopt(short, long, help = "number of buckets")]
        kbuckets: usize,

        #[structopt(short, long, help = "use 1 or 0 only in the buckets")]
        binary: bool,

        #[structopt(
            short = "F",
            long,
            help = "delimeter used to split the items (default = ',')",
            default_value = ","
        )]
        delimiter: String,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },
}

#[derive(StructOpt, Debug)]
enum Command {
    #[structopt(about = "summary statistics from a single vector")]
    Summary {
        #[structopt(short)]
        transpose: bool,

        #[structopt(short = "h", long = "with-header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(about = "k-quintile from a single vector (default k = 5)")]
    Quintiles {
        #[structopt(
            short = "k",
            help = "k-quintile, for some input k",
            default_value = "5"
        )]
        quintiles: u32,

        #[structopt(short = "h", long = "with-header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(about = "train, predict, and understand xgboost models")]
    Xgb(XgbOptions),

    #[structopt(about = "Computes the Pearson correlation coefficient")]
    CorMatrix {
        #[structopt(short, long, help = "predictor column", default_value = "1000000")]
        ycol: usize,

        #[structopt(short = "h", long = "with-header", help = "with header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(
        about = "evaluation metrics to score an output, confusion matrix and other helpful probablities. Note: all classes need to be 0..N"
    )]
    Eval {
        #[structopt(
            short,
            long,
            help = "if value is (0,1) set a threshold at which the value will be converted to 1"
        )]
        threshold: Option<f32>,

        #[structopt(short, long, parse(from_occurrences), help = "show verbose output")]
        verbose: u32,

        #[structopt(
            short,
            long,
            help = "Use bayes theorem to estimate the effective probability using a estimate of the true rate of occurance for each class. This value expects a string of floats, one for each class in the dataset. E.g. -b '0.1, 0.2, 0.3'"
        )]
        bayes: Option<String>,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(about = "data transformations and feature generation tools")]
    Extract(ExtractOptions),
}

fn main() {
    let opt = Opt::from_args();

    match opt.cmd {
        Command::Summary {
            transpose,
            with_header,
            input,
        } => {
            let raw_inputs = series::get_input(input);
            let data = series::to_vector(&raw_inputs, with_header);
            let series = series::Series::new(data);

            if transpose {
                series.summary_t();
            } else {
                series.summary();
            }
        }

        Command::Quintiles {
            quintiles,
            with_header,
            input,
        } => {
            let raw_inputs = series::get_input(input);
            let mut data = series::to_vector(&raw_inputs, with_header);
            print_quintiles(&mut data, quintiles);
        }

        Command::Eval {
            threshold,
            verbose,
            bayes,
            input,
        } => {
            let raw_inputs = series::get_input(input);
            let tuples = series::to_tuple(&raw_inputs);

            let bases: Vec<f32> = if let Some(s) = bayes {
                match series::str_to_vector(&s, ",") {
                    Ok(xs) => xs,
                    Err(_) => {
                        eprintln!("error parsing -b list");
                        std::process::exit(1);
                    }
                }
            } else {
                vec![]
            };

            let matrix = series::confusion_matrix(&tuples, threshold);
            let stats = series::confusion_matrix_stats(&matrix);

            let size = matrix.len();

            // convert the matrix into a formatted string for stdout
            let mut header = String::new();
            let mut body = String::new();
            header.push_str("Confusion Matrix\n");
            header.push_str("Predicted on y-axis, Actual on x-axis\n");
            header.push('\n');
            header.push_str(&format!("{:<8}", "-"));

            for i in 0..size {
                header.push_str(&format!("{:<8}", size - 1 - i));

                body.push_str(&format!("{:<8}", size - 1 - i));

                for j in 0..size {
                    body.push_str(&format!("{:<8}", matrix[i][j]));
                }

                body.push('\n');
            }

            // print matrix to stdout
            println!("{}", header);
            println!("{}", body);

            let mut bayes_calc_str = String::new();
            let mut verbose_str = String::new();

            verbose_str.push_str(&format!(
                "{:<8}{:<8}{:<8}{:<8}{:<8}\n",
                "class", "tpr", "fpr", "tnr", "fnr"
            ));

            for stat in stats {
                if verbose > 0 {
                    verbose_str.push_str(&format!(
                        "{:<8}{:<8.3}{:<8.3}{:<8.3}{:<8.3}\n",
                        stat.label, stat.tpr, stat.fpr, stat.tnr, stat.fnr
                    ));
                }

                if !bases.is_empty() {
                    if bases.len() != size {
                        eprintln!(
                            "invalid number of baseline values, it must match the number of classes"
                        );
                        std::process::exit(1);
                    }

                    let prob_positive =
                        (stat.tpr * bases[stat.label]) + (stat.fpr * (1.0 - bases[stat.label]));

                    let prob_class_given_positive = (stat.tpr * bases[stat.label]) / prob_positive;

                    bayes_calc_str.push_str(&format!(
                        "{}: Pr(class_{}|positive) = {}\n",
                        stat.label, stat.label, prob_class_given_positive
                    ));
                }
            }

            if verbose > 0 {
                print!("{}", verbose_str);
                println!("");
            }

            if !bases.is_empty() {
                println!("Bayes estimates with baseline rates\n");
                print!("{}", bayes_calc_str);
                println!("");
            }

            if verbose > 1 && matrix.len() == 2 {
                let output = series::threshold_table_stats(&tuples);

                println!("ROC table\n");
                println!("{:<8}{:<8}{:<8}{:<8}{:<8}", "t", "prec", "f1", "tpr", "fpr");

                for row in output {
                    println!(
                        "{:<8.2}{:<8.4}{:<8.4}{:<8.4}{:<8.4}",
                        row[0], row[1], row[2], row[3], row[4]
                    );
                }
            }
        }

        Command::Xgb(XgbOptions::Train {
            ycol,
            depth,
            eta,
            model_out: output,
            objective,
            nclasses,
            rounds,
            with_header,
            input,
        }) => {
            let raw_inputs = series::get_input(input);
            let (xdata, ydata) = series::to_matrix(&raw_inputs, ycol, with_header);

            let training_set = xgb::to_xgboost_dataset(&xdata, Some(ydata));

            xgb::train(
                &training_set,
                &objective,
                nclasses,
                depth,
                eta,
                rounds,
                &output,
            );
        }

        Command::Xgb(XgbOptions::Predict {
            ycol,
            model_in,
            with_header,
            input,
        }) => {
            let inputs = series::get_input(input);
            let (xdata, ydata) = series::to_matrix(&inputs, ycol, with_header);
            let test_set = xgb::to_xgboost_dataset(&xdata, None);

            let predicted = xgb::predict(&model_in, &test_set);
            let mut buf = String::new();

            for (index, row) in xdata.iter().enumerate() {
                let mut xs = String::new();
                let size = row.len();

                for (index, item) in row.iter().enumerate() {
                    if index < size - 1 {
                        xs.push_str(&format!("{},", item));
                    } else {
                        xs.push_str(&format!("{}", item));
                    }
                }

                if ydata.is_empty() {
                    buf.push_str(&format!("{},{}\n", predicted[index], xs));
                } else {
                    buf.push_str(&format!("{},{},{}\n", predicted[index], ydata[index], xs));
                };

                if index % 1000 == 0 {
                    std::io::stdout().write(&buf.as_bytes()).unwrap();
                    buf = String::new();
                }
            }

            if !buf.is_empty() {
                std::io::stdout().write(&buf.as_bytes()).unwrap();
            }
        }

        Command::Xgb(XgbOptions::Importance {
            input,
            typ,
            dump_model,
        }) => {
            let bytes = if let Some(path) = input {
                match std::fs::read(path) {
                    Ok(contents) => contents,

                    Err(e) => {
                        eprintln!("{}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                let mut input = vec![];
                let mut stdin = std::io::stdin();
                let _ = stdin.read_to_end(&mut input).unwrap();
                input
            };

            xgb::dump_model(&bytes, dump_model, &typ);
        }

        Command::CorMatrix {
            ycol,
            with_header,
            input,
        } => {
            let input = series::get_input(input);
            let (xdata, _) = series::to_matrix(&input, ycol, with_header);
            let matrix = series::correlation_matrix(&xdata);

            let size = matrix.len();

            print!("{:<8}", "-");
            for i in 0..size {
                print!("{:<8}", i);
            }
            println!("");

            for i in 0..size {
                print!("{:<8}", i);
                for j in 0..size {
                    if j < i + 1 {
                        print!("{:<8.2}", matrix[i][j]);
                    }
                }
                println!("");
            }
        }

        Command::Extract(ExtractOptions::ByteHistogram { input }) => {
            let input = series::get_input_bytes(input);
            let histo = series::to_byte_histogram(&input);
            let length = histo.len();
            let mut output = String::new();
            for (index, b) in histo.iter().enumerate() {
                if index < length - 1 {
                    output.push_str(&format!("{},", b));
                } else {
                    output.push_str(&format!("{}", b));
                }
            }

            println!("{}", output);
        }

        Command::Extract(ExtractOptions::Entropy { input }) => {
            let input = series::get_input_bytes(input);
            let en = series::entropy(&input);
            println!("{}", en);
        }

        Command::Extract(ExtractOptions::HashTrick {
            kbuckets,
            binary,
            delimiter,
            input,
        }) => {
            let s = series::get_input(input);
            let mut buckets_out: Vec<u32> = Vec::with_capacity(kbuckets);
            for _ in 0..(kbuckets - 1) {
                buckets_out.push(0);
            }

            for item in s.split(&delimiter) {
                let hash_result = murmur3_32(&mut Cursor::new(item), 0).unwrap();
                let index = hash_result % (kbuckets as u32 - 1);
                if binary {
                    buckets_out[index as usize] = 1;
                } else {
                    buckets_out[index as usize] += 1;
                }
            }

            let mut out_str = String::new();

            for (index, val) in buckets_out.iter().enumerate() {
                if index < kbuckets - 2 {
                    out_str.push_str(&format!("{},", val));
                } else {
                    out_str.push_str(&format!("{}", val));
                }
            }
            println!("{}", out_str);
        }
    }
}
