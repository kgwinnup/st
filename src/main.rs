use itertools::Itertools;
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rasciigraph;
use std::collections::HashMap;
use std::io::prelude::*;
use std::path::PathBuf;
use structopt::StructOpt;
use xgboost::{parameters, Booster, DMatrix};

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
enum TrainOptions {
    Binary {
        #[structopt(short, long, help = "predictor column")]
        ycol: usize,

        #[structopt(short, long, help = "max depth")]
        depth: Option<u32>,

        #[structopt(short, long, help = "eta")]
        eta: Option<f32>,

        #[structopt(short = "m", long = "model", help = "path to save model")]
        model: String,

        #[structopt(short = "h", long = "with-header", help = "with header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },
}

#[derive(StructOpt, Debug)]
enum PredictOptions {
    Binary {
        #[structopt(short, long, help = "path to model")]
        model: String,

        #[structopt(short = "h", long = "with-header", help = "with header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },
}

#[derive(StructOpt, Debug)]
enum TreeOptions {
    Train(TrainOptions),
    Predict(PredictOptions),
}

#[derive(StructOpt, Debug)]
enum Command {
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

    Quintiles {
        #[structopt(short, help = "k-quintile, for some input k", default_value = "5")]
        quintiles: u32,

        #[structopt(short = "h", long = "with-header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

    Graph {
        #[structopt(short)]
        typ: String,

        #[structopt(short = "h", long = "with-header")]
        with_header: bool,

        #[structopt(parse(from_os_str))]
        input: Option<PathBuf>,
    },

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

    Xgboost(TreeOptions),
}

fn median(input: &mut [f64]) -> f64 {
    if input.len() == 0 {
        return 0.0;
    }

    if input.len() == 1 {
        return input[0];
    }

    if input.len() == 2 {
        return input[0] + input[1] / 2.0;
    }

    input.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = input.len() / 2;
    input[mid]
}

fn mode(input: &[f64], prec: u32) -> f64 {
    let mut counts: HashMap<u32, u32> = HashMap::new();

    for val in input.iter() {
        let temp = val * (prec as f64);
        *counts.entry(temp as u32).or_insert(1) += 1;
    }

    let out = counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(val, _)| val)
        .expect("cannot compute mode of zero numbers");

    out as f64 / prec as f64
}

fn mean(input: &[f64]) -> f64 {
    if input.len() == 0 {
        return 0.0;
    }

    if input.len() == 1 {
        return input[0];
    }

    input.iter().sum::<f64>() as f64 / input.len() as f64
}

fn stdev_var_mean(input: &[f64]) -> (f64, f64, f64) {
    if input.len() == 0 {
        return (0.0, 0.0, 0.0);
    }

    if input.len() == 1 {
        return (0.0, 0.0, 0.0);
    }

    let u = mean(input);

    let mut sum = 0.0;

    for i in input.iter() {
        sum += (i - u).powf(2.0);
    }

    (
        (sum / input.len() as f64).sqrt(),
        (sum / input.len() as f64),
        u,
    )
}

fn print_summary(input: &mut [f64], prec: u32) {
    let (sd, var, mean) = stdev_var_mean(input);
    let m = mode(input, prec);
    let med = median(input);
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

fn print_summary_t(input: &mut [f64], prec: u32) {
    let (sd, var, mean) = stdev_var_mean(input);
    let m = mode(input, prec);
    let med = median(input);
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

fn print_line(input: &[f64]) {
    let config = rasciigraph::Config::default()
        .with_height(20)
        .with_width(70)
        .with_offset(0);
    let plot = rasciigraph::plot(input.to_vec(), config);
    println!("{}", plot);
}

fn print_histo(input: &mut [f64], prec: u32) {
    let mut histo: HashMap<u32, u32> = HashMap::new();

    for val in input.iter() {
        let temp = val * (prec as f64);
        *histo.entry(temp as u32).or_insert(1) += 1;
    }

    let mut new_vec = vec![];

    let keys = histo.keys().sorted();
    for k in keys {
        if let Some(v) = histo.get(k) {
            new_vec.push(*v as f64);
        }
    }

    let config = rasciigraph::Config::default()
        .with_height(20)
        .with_width(70)
        .with_offset(0);
    let plot = rasciigraph::plot(new_vec, config);
    println!("{}", plot);
}

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

fn handle_sampling(raw_inputs: &str, with_header: bool, size: u32, with_replacement: bool) {
    let mut lines = vec![];
    let mut header = String::new();

    for (index, line) in raw_inputs.split("\n").enumerate() {
        if index == 0 && with_header {
            header = line.to_string();
            continue;
        }

        if line == "\n" || line == "" {
            continue;
        }

        lines.push(line);
    }

    if !header.is_empty() {
        println!("{}", header);
    }

    if with_replacement {
        if size <= 0 {
            eprintln!("invalid sample with replacement size, must be a positive number");
            std::process::exit(1);
        }

        let mut rng = rand::thread_rng();
        let roll = Uniform::from(0..lines.len());

        let mut count = 0;

        loop {
            if count > size {
                break;
            }

            let index = roll.sample(&mut rng);
            println!("{}", lines[index as usize]);
            count += 1;
        }
    } else {
        if size <= 0 || size > lines.len() as u32 {
            eprintln!("invalid sampling without replacement. n must be within the magnitude of the input data set");
            std::process::exit(1);
        }

        let mut rng = rand::thread_rng();
        lines.shuffle(&mut rng);

        for (index, line) in lines.iter().enumerate() {
            if index as u32 > size - 1 {
                break;
            }
            println!("{}", line);
        }
    }
}

fn vectorize_column(raw_inputs: &str, with_header: bool) -> Vec<f64> {
    let mut data = vec![];

    for (index, line) in raw_inputs.split("\n").enumerate() {
        if index == 0 && with_header {
            continue;
        }

        if line == "\n" || line == "" {
            continue;
        }

        let temp: Result<f64, _> = line.trim().parse();
        match temp {
            Ok(f) => data.push(f),
            Err(_) => {
                eprintln!("error converting to float: {} at line {}", line, index);
                std::process::exit(1);
            }
        }
    }

    data
}

fn to_matrix_1y(
    raw_inputs: &str,
    ycol: usize,
    with_header: bool,
    output_lines: bool,
) -> (DMatrix, Vec<&str>) {
    let mut xdata = Vec::new();
    let mut ydata = Vec::new();

    let mut rows = 0;

    let mut lines = vec![];

    for (index, line) in raw_inputs.split("\n").enumerate() {
        if index == 0 && with_header {
            continue;
        }

        if line == "\n" || line == "" {
            continue;
        }

        let split = line.split(",");

        for (index, val) in split.enumerate() {
            let temp: Result<f32, _> = val.trim().parse();
            match temp {
                Ok(f) => {
                    if index == ycol {
                        ydata.push(f)
                    } else {
                        xdata.push(f)
                    }
                }
                Err(_) => {
                    eprintln!("error converting to float: {} at line {}", line, index);
                    std::process::exit(1);
                }
            }
        }

        if output_lines {
            lines.push(line);
        }

        rows += 1;
    }

    match DMatrix::from_dense(&xdata, rows) {
        Ok(mut x) => {
            let _ = x.set_labels(&ydata);
            (x, lines)
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}

fn get_input(input: Option<PathBuf>) -> String {
    if let Some(path) = input {
        match std::fs::read_to_string(path) {
            Ok(contents) => contents,

            Err(e) => {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }
    } else {
        let mut input = String::new();
        let stdin = std::io::stdin();

        for line in stdin.lock().lines() {
            let buf = line.expect("failed to read line");
            input.push_str(&buf);
            input.push('\n');
        }

        input
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
            let raw_inputs = get_input(input);
            handle_sampling(&raw_inputs, with_header, size, replace);
        }

        Command::Summary {
            transpose,
            precision,
            foo,
            with_header,
            input,
        } => {
            println!("{:?}", foo);
            let raw_inputs = get_input(input);
            let mut data = vectorize_column(&raw_inputs, with_header);
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
            let raw_inputs = get_input(input);
            let mut data = vectorize_column(&raw_inputs, with_header);
            print_quintiles(&mut data, quintiles);
        }

        Command::Graph {
            typ,
            with_header,
            input,
        } => {
            let raw_inputs = get_input(input);
            let mut data = vectorize_column(&raw_inputs, with_header);
            let name = typ.to_lowercase();

            if name.starts_with("line") {
                print_line(&data);
            } else if name.starts_with("histo") {
                print_histo(&mut data, 1);
            } else {
                eprintln!("invalid graph type");
                std::process::exit(1);
            }
        }

        Command::Xgboost(TreeOptions::Train(TrainOptions::Binary {
            ycol,
            depth,
            eta,
            model: output,
            with_header,
            input,
        })) => {
            let raw_inputs = get_input(input);
            let (training_set, _) = to_matrix_1y(&raw_inputs, ycol, with_header, false);

            let eta_val = eta.unwrap_or(0.3);
            let depth_val = depth.unwrap_or(6);

            let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
                .max_depth(depth_val)
                .eta(eta_val)
                .build()
                .unwrap();

            let booster_params = parameters::BoosterParametersBuilder::default()
                .booster_type(parameters::BoosterType::Tree(tree_params))
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

        Command::Xgboost(TreeOptions::Predict(PredictOptions::Binary {
            model,
            with_header,
            input,
        })) => {
            let inputs = get_input(input);
            let (test_set, lines) = to_matrix_1y(&inputs, 1000000, with_header, true);

            let bst = Booster::load(model).unwrap();
            let predict = bst.predict(&test_set).unwrap();

            for (index, line) in lines.iter().enumerate() {
                println!("{},{}", predict[index], line);
            }
        }
    }
}
