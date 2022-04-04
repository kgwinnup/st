use murmur3::murmur3_32;
use st_stat;
use std::collections::HashMap;
use std::io::prelude::*;
use std::io::Cursor;
use std::path::PathBuf;
use structopt::StructOpt;
use xgboost::{parameters, Booster, DMatrix};

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

        #[structopt(short, long, help = "eta", default_value = "3")]
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

        #[structopt(short, long)]
        foo: Vec<u32>,

        #[structopt(short = "h", long = "with-header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(about = "k-quintile from a single vector (default k = 5)")]
    Quintiles {
        #[structopt(short, help = "k-quintile, for some input k", default_value = "5")]
        quintiles: u32,

        #[structopt(short = "h", long = "with-header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(about = "very simple cli graphing")]
    Graph {
        #[structopt(short)]
        typ: String,

        #[structopt(short = "h", long = "with-header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(about = "sample a vector, with or without replacement")]
    Sample {
        #[structopt(short)]
        size: u32,

        #[structopt(short = "r", help = "sample with replacement")]
        replace: bool,

        #[structopt(short = "h", long = "with-header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(about = "train, predict, and understand xgboost models")]
    Xgboost(TreeOptions),

    #[structopt(
        about = "evaluation metrics to score an output, confusion matrix, f1, recall, fpr"
    )]
    Eval {
        #[structopt(
            short,
            long,
            help = "output confustion matrix for some threshold [0,1]"
        )]
        confusion_matrix: Option<f32>,

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
        Command::Sample {
            size,
            replace,
            with_header,
            input,
        } => {
            let raw_inputs = st_input::get_input(input);
            st_stat::sample(&raw_inputs, with_header, size, replace);
        }

        Command::Summary {
            transpose,
            precision,
            foo,
            with_header,
            input,
        } => {
            println!("{:?}", foo);
            let raw_inputs = st_input::get_input(input);
            let mut data = st_input::to_vector(&raw_inputs, with_header);
            if transpose {
                st_stat::print_summary_t(&mut data, precision)
            } else {
                st_stat::print_summary(&mut data, precision)
            }
        }

        Command::Quintiles {
            quintiles,
            with_header,
            input,
        } => {
            let raw_inputs = st_input::get_input(input);
            let mut data = st_input::to_vector(&raw_inputs, with_header);
            st_stat::print_quintiles(&mut data, quintiles);
        }

        Command::Graph {
            typ,
            with_header,
            input,
        } => {
            let raw_inputs = st_input::get_input(input);
            let mut data = st_input::to_vector(&raw_inputs, with_header);
            let name = typ.to_lowercase();

            if name.starts_with("line") {
                st_stat::print_line(&data);
            } else if name.starts_with("histo") {
                st_stat::print_histo(&mut data, 1);
            } else {
                eprintln!("invalid graph type");
                std::process::exit(1);
            }
        }

        Command::Eval {
            input,
            confusion_matrix,
        } => {
            let raw_inputs = st_input::get_input(input);
            let tuples = st_input::to_tuple(&raw_inputs);

            if let Some(t) = confusion_matrix {
                let mut ttp: f32 = 0.0;
                let mut tfp: f32 = 0.0;
                let mut tfn: f32 = 0.0;
                let mut ttn: f32 = 0.0;

                for (p, a) in tuples.iter() {
                    if *p >= t && *a == 1.0 {
                        ttp += 1.0;
                        continue;
                    }

                    if *p >= t && *a == 0.0 {
                        tfp += 1.0;
                        continue;
                    }

                    if *p < t && *a == 1.0 {
                        tfn += 1.0;
                        continue;
                    }

                    if *p < t && *a == 0.0 {
                        ttn += 1.0;
                        continue;
                    }
                }

                println!("{:6} {:6}", ttp, tfp);
                println!("{:6} {:6}", tfn, ttn);
            } else {
                println!(
                    "{:8} {:8} {:8} {:8} {:8}",
                    "t", "prec", "f1", "recall", "fpr"
                );

                let mut threshold: f32 = 0.0;
                loop {
                    if threshold > 1.0 {
                        break;
                    }

                    let mut ttp: f32 = 0.0;
                    let mut tfp: f32 = 0.0;
                    let mut tfn: f32 = 0.0;
                    let mut ttn: f32 = 0.0;

                    for (p, a) in tuples.iter() {
                        if *p >= threshold && *a == 1.0 {
                            ttp += 1.0;
                            continue;
                        }

                        if *p >= threshold && *a == 0.0 {
                            tfp += 1.0;
                            continue;
                        }

                        if *p < threshold && *a == 1.0 {
                            tfn += 1.0;
                            continue;
                        }

                        if *p < threshold && *a == 0.0 {
                            ttn += 1.0;
                            continue;
                        }
                    }

                    let precision = ttp / (ttp + tfp);
                    let recall = ttp / (ttp + tfn);
                    let f1 = 2.0 * (recall * precision) / (recall + precision);
                    let fpr = tfp / (tfp + ttn);

                    println!(
                        "{:.2} {:8.4} {:8.4} {:8.4} {:8.4}",
                        threshold, precision, f1, recall, fpr
                    );
                    threshold += 0.05;
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
            let raw_inputs = st_input::get_input(input);
            let (xdata, ydata) = st_input::to_matrix(&raw_inputs, ycol, with_header);

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
            let inputs = st_input::get_input(input);
            let (xdata, ydata) = st_input::to_matrix(&inputs, ycol, with_header);
            let test_set = to_xgboost_dataset(&xdata, None);

            let bst = Booster::load(model).unwrap();
            let predicted = bst.predict(&test_set).unwrap();

            for (index, row) in xdata.iter().enumerate() {
                let mut xs = String::new();
                let size = row.len();

                for (index, item) in row.iter().enumerate() {
                    if index < size - 2 {
                        xs.push_str(&format!("{},", item));
                    } else {
                        xs.push_str(&format!("{}", item));
                    }
                }

                if ydata.is_empty() {
                    println!("{},{}", predicted[index], xs);
                } else {
                    println!("{},{},{}", predicted[index], ydata[index], xs);
                };
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
            let input = st_input::get_input(input);
            let histo = st_input::to_byte_histogram(input.as_bytes());
            let length = histo.len();
            let mut output = String::new();
            for (index, b) in histo.iter().enumerate() {
                if index < length - 2 {
                    output.push_str(&format!("{},", b));
                } else {
                    output.push_str(&format!("{}", b));
                }
            }

            println!("{}", output);
        }

        Command::Extract(ExtractOptions::HashTrick {
            kbuckets,
            binary,
            delimiter,
            input,
        }) => {
            let s = st_input::get_input(input);
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
