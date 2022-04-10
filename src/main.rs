use murmur3::murmur3_32;
use st_core;
use std::collections::HashMap;
use std::io::prelude::*;
use std::io::Cursor;
use std::path::PathBuf;
use structopt::StructOpt;
use xgboost::{parameters, Booster, DMatrix};

pub fn print_summary(input: &mut [f64], prec: u32) {
    let (sd, var, mean) = st_core::stdev_var_mean(input);
    let m = st_core::mode(input, prec);
    let med = st_core::median(input);
    let min = input[0];
    let max = input[input.len() - 1];

    println!(
        "{:<11}{:<11}{:<11}{:<11}{:<11}{:<11}{:<11}{:<11}",
        "n", "min", "max", "mean", "median", "mode", "sd", "var"
    );
    println!(
        "{:<11}{:<11}{:<11}{:<11}{:<11}{:<11}{:<11}{:<11}",
        input.len(),
        min as f32,
        max as f32,
        mean as f32,
        med as f32,
        m as f32,
        sd as f32,
        var as f32
    );
}

pub fn print_summary_t(input: &mut [f64], prec: u32) {
    let (sd, var, mean) = st_core::stdev_var_mean(input);
    let m = st_core::mode(input, prec);
    let med = st_core::median(input);
    let min = input[0];
    let max = input[input.len() - 1];

    println!("{:<8}{}", "N", input.len());
    println!("{:<8}{}", "min", min as f32);
    println!("{:<8}{}", "max", max as f32);
    println!("{:<8}{}", "mean", mean as f32);
    println!("{:<8}{}", "med", med as f32);
    println!("{:<8}{}", "mode", m as f32);
    println!("{:<8}{}", "stdev", sd as f32);
    println!("{:<8}{}", "var", var as f32);
}

pub fn print_quintiles(input: &mut [f64], k: u32) {
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

fn to_xgboost_dataset(xdata: &Vec<Vec<f64>>, ydata: Option<Vec<f32>>) -> DMatrix {
    let rows = xdata.len();
    let mut xdata2 = vec![];

    for row in xdata {
        for item in row {
            xdata2.push(*item as f32);
        }
    }

    match DMatrix::from_dense(&xdata2, rows) {
        Ok(mut x) => {
            if let Some(y) = ydata {
                let _ = x.set_labels(&y);
            }
            x
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
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
enum TreeOptions {
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
enum Command {
    #[structopt(about = "summary statistics from a single vector")]
    Summary {
        #[structopt(short)]
        transpose: bool,

        #[structopt(
            long,
            default_value = "1",
            help = "if inputs are floats, for bucketing purposes they are converted to ints"
        )]
        precision: u32,

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
    Xgboost(TreeOptions),

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

        #[structopt(short, long, help = "show verbose output")]
        verbose: bool,

        #[structopt(
            long,
            help = "if results are binary predictions in (0,1), output a table for each threshold and that thresholds metrics."
        )]
        table: bool,

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

#[derive(Default, Debug)]
struct XgboostNode {
    name: String,
    gain: f32,
    cover: f32,
}

fn parse_node(node_str: &str) -> XgboostNode {
    let mut name = String::new();

    let mut in_name = false;
    let mut in_map = false;
    let mut in_key_name = false;
    let mut key_name = String::new();
    let mut val = String::new();

    let mut node = XgboostNode::default();

    for c in node_str.chars() {
        match c {
            ' ' => {
                in_map = true;
                in_key_name = true;
                node.name = name.to_string();
            }

            '[' => {
                in_name = true;
            }

            '<' => {
                in_name = false;
            }

            '=' => {
                in_key_name = false;
            }

            ',' => {
                in_key_name = true;

                let temp: Result<f32, _> = val.parse();
                match temp {
                    Ok(f) => {
                        if key_name == "gain" {
                            node.gain = f;
                        } else if key_name == "cover" {
                            node.cover = f;
                        }

                        val = String::new();
                        key_name = String::new();
                    }
                    Err(e) => {
                        eprintln!("{}", e);
                        std::process::exit(1);
                    }
                }
            }

            c => {
                if in_name {
                    name.push(c);
                } else if in_map && in_key_name {
                    key_name.push(c);
                } else if in_map && !in_key_name {
                    val.push(c);
                }
            }
        }
    }

    let temp: Result<f32, _> = val.parse();
    match temp {
        Ok(f) => {
            if key_name == "gain" {
                node.gain = f;
            } else if key_name == "cover" {
                node.cover = f;
            }
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }

    node
}

fn importance(model_dump: String, typ: &str) {
    let mut gain_map = HashMap::new();
    let mut cover_map = HashMap::new();
    let mut freq_map = HashMap::new();

    let mut total_gain = 0.0;
    let mut total_cover = 0.0;
    let mut total_freq = 0;

    for line in model_dump.split("\n").into_iter() {
        if line.contains("leaf") {
            continue;
        }

        let space_split = line.split(" ").collect::<Vec<&str>>();
        if space_split.len() != 2 {
            continue;
        }

        let node = parse_node(&line);

        total_freq += 1;

        if let Some(val) = freq_map.get_mut(&node.name) {
            *val += 1;
        } else {
            freq_map.insert(node.name.to_string(), 1);
        }

        if let Some(val) = gain_map.get_mut(&node.name) {
            *val += node.gain;
            total_gain += node.gain;
        } else {
            gain_map.insert(node.name.to_string(), node.gain);
            total_gain += node.gain;
        }

        if let Some(val) = cover_map.get_mut(&node.name) {
            *val += node.cover;
            total_cover += node.cover;
        } else {
            cover_map.insert(node.name.to_string(), node.cover);
            total_cover += node.cover;
        }
    }

    if typ == "gain" {
        let mut list = vec![];
        for (k, v) in gain_map {
            list.push((k, v / total_gain));
        }
        list.sort_by(|(_, v1), (_, v2)| v2.partial_cmp(v1).unwrap());

        for (name, val) in list {
            println!("{} = {}", name, val);
        }
    }

    if typ == "cover" {
        let mut list = vec![];
        for (k, v) in cover_map {
            list.push((k, v / total_cover));
        }
        list.sort_by(|(_, v1), (_, v2)| v2.partial_cmp(v1).unwrap());

        for (name, val) in list {
            println!("{} = {}", name, val);
        }
    }

    if typ == "freq" {
        let mut list = vec![];
        for (k, v) in freq_map {
            list.push((k, v / total_freq));
        }
        list.sort_by(|(_, v1), (_, v2)| v2.partial_cmp(v1).unwrap());

        for (name, val) in list {
            println!("{} = {}", name, val);
        }
    }
}

fn main() {
    let opt = Opt::from_args();

    match opt.cmd {
        Command::Summary {
            transpose,
            precision,
            with_header,
            input,
        } => {
            let raw_inputs = st_core::get_input(input);
            let mut data = st_core::to_vector(&raw_inputs, with_header);
            if transpose {
                print_summary_t(&mut data, precision)
            } else {
                print_summary(&mut data, precision)
            }
        }

        Command::Quintiles {
            quintiles,
            with_header,
            input,
        } => {
            let raw_inputs = st_core::get_input(input);
            let mut data = st_core::to_vector(&raw_inputs, with_header);
            print_quintiles(&mut data, quintiles);
        }

        Command::Eval {
            threshold,
            verbose,
            table,
            bayes,
            input,
        } => {
            let raw_inputs = st_core::get_input(input);
            let tuples = st_core::to_tuple(&raw_inputs);

            let bases: Vec<f32> = if let Some(s) = bayes {
                match st_core::str_to_vector(&s, ",") {
                    Ok(xs) => xs,
                    Err(_) => {
                        eprintln!("error parsing -b list");
                        std::process::exit(1);
                    }
                }
            } else {
                vec![]
            };

            let matrix = st_core::confusion_matrix(&tuples, threshold);
            let stats = st_core::confusion_matrix_stats(&matrix);

            let size = matrix.len();

            // convert the matrix into a formatted string for stdout
            let mut header = String::new();
            let mut body = String::new();
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
                if verbose {
                    verbose_str.push_str(&format!(
                        "{:<8}{:<8.3}{:<8.3}{:<8.3}{:<8.3}\n",
                        stat.label, stat.tpr, stat.fpr, stat.tnr, stat.fnr
                    ));
                }

                if !bases.is_empty() {
                    if bases.len() != size {
                        eprintln!(
                            "invalid number of --base values, it must match the number of classes"
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

            if verbose {
                print!("{}", verbose_str);
                println!("");
            }

            if !bases.is_empty() {
                print!("{}", bayes_calc_str);
                println!("");
            }

            if table {
                let output = st_core::threshold_table_stats(&tuples);

                println!("{:<8}{:<8}{:<8}{:<8}{:<8}", "t", "prec", "f1", "trp", "fpr");

                for row in output {
                    println!(
                        "{:<8.2}{:<8.4}{:<8.4}{:<8.4}{:<8.4}",
                        row[0], row[1], row[2], row[3], row[4]
                    );
                }
            }
        }

        Command::Xgboost(TreeOptions::Train {
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
            let raw_inputs = st_core::get_input(input);
            let (xdata, ydata) = st_core::to_matrix(&raw_inputs, ycol, with_header);

            let training_set = to_xgboost_dataset(&xdata, Some(ydata));

            let objective_fn = match objective.as_str() {
                "binary:logistic" => parameters::learning::Objective::BinaryLogistic,
                "multi:softmax" => parameters::learning::Objective::MultiSoftmax(nclasses),
                "multi:softprob" => parameters::learning::Objective::MultiSoftprob(nclasses),
                _ => {
                    eprintln!("invalid objective function");
                    std::process::exit(1);
                }
            };

            let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
                .objective(objective_fn)
                .build()
                .unwrap();

            let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
                .max_depth(depth)
                .eta(eta)
                .build()
                .unwrap();

            let booster_params = parameters::BoosterParametersBuilder::default()
                .booster_type(parameters::BoosterType::Tree(tree_params))
                .learning_params(learning_params)
                .verbose(false)
                .build()
                .unwrap();

            let training_params = parameters::TrainingParametersBuilder::default()
                .dtrain(&training_set)
                .booster_params(booster_params)
                .boost_rounds(rounds)
                .build()
                .unwrap();

            let bst = Booster::train(&training_params).unwrap();
            for (k, v) in bst.evaluate(&training_set).unwrap() {
                eprintln!("{} = {}", k, v);
            }

            let _ = bst.save(output).unwrap();
        }

        Command::Xgboost(TreeOptions::Predict {
            ycol,
            model_in: model,
            with_header,
            input,
        }) => {
            let inputs = st_core::get_input(input);
            let (xdata, ydata) = st_core::to_matrix(&inputs, ycol, with_header);
            let test_set = to_xgboost_dataset(&xdata, None);

            let bst = Booster::load(model).unwrap();
            let predicted = bst.predict(&test_set).unwrap();

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

        Command::Xgboost(TreeOptions::Importance {
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

            if dump_model {
                let bst = Booster::load_buffer(&bytes).unwrap();
                println!("{}", bst.dump_model(true, None).unwrap());
            } else {
                let bst = Booster::load_buffer(&bytes).unwrap();
                let model = bst.dump_model(true, None).unwrap();
                importance(model, &typ);
            }
        }

        Command::Extract(ExtractOptions::ByteHistogram { input }) => {
            let input = st_core::get_input_bytes(input);
            let histo = st_core::to_byte_histogram(&input);
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
            let input = st_core::get_input_bytes(input);
            let en = st_core::entropy(&input);
            println!("{}", en);
        }

        Command::Extract(ExtractOptions::HashTrick {
            kbuckets,
            binary,
            delimiter,
            input,
        }) => {
            let s = st_core::get_input(input);
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
