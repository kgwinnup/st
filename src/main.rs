use st_stat;
use std::collections::HashMap;
use std::io::prelude::*;
use std::path::PathBuf;
use structopt::StructOpt;
use xgboost::{parameters, Booster};

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
        buckets: usize,

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

        #[structopt(short, long, help = "max depth")]
        depth: Option<u32>,

        #[structopt(short, long, help = "eta")]
        eta: Option<f32>,

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

        #[structopt(short = "h", long = "with-header", help = "with header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    #[structopt(about = "use an xgboost model against some input")]
    Predict {
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
            with_header,
            input,
        }) => {
            let raw_inputs = st_input::get_input(input);
            let (training_set, _) = st_input::to_matrix(&raw_inputs, ycol, with_header, false);

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

            let eta_val = eta.unwrap_or(0.3);
            let depth_val = depth.unwrap_or(6);

            let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
                .max_depth(depth_val)
                .eta(eta_val)
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
                .build()
                .unwrap();

            let bst = Booster::train(&training_params).unwrap();
            for (k, v) in bst.evaluate(&training_set).unwrap() {
                eprintln!("{} = {}", k, v);
            }

            let _ = bst.save(output).unwrap();
        }

        Command::Xgboost(TreeOptions::Predict {
            model_in: model,
            with_header,
            input,
        }) => {
            let inputs = st_input::get_input(input);
            let (test_set, lines) = st_input::to_matrix(&inputs, 1000000, with_header, true);

            let bst = Booster::load(model).unwrap();
            let predict = bst.predict(&test_set).unwrap();

            for (index, line) in lines.iter().enumerate() {
                println!("{},{}", predict[index], line);
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
                if index < length - 1 {
                    output.push_str(&format!("{},", b));
                } else {
                    output.push_str(&format!("{}", b));
                }
            }

            println!("{}", output);
        }

        Command::Extract(ExtractOptions::HashTrick { buckets, input }) => {}
    }
}
