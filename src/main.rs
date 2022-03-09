use itertools::Itertools;
use rasciigraph;
use std::collections::HashMap;
use std::io::prelude::*;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "stat", about = "quick stat information")]
struct Opt {
    #[structopt(short, long, default_value = "1")]
    precision: u32,

    #[structopt(short, long)]
    transpose: bool,

    #[structopt(short = "l", long = "line")]
    line: bool,

    #[structopt(short = "h", long = "histo")]
    histo: bool,

    #[structopt(short = "H", long = "with-header")]
    with_header: bool,

    #[structopt(parse(from_os_str))]
    input: Option<PathBuf>,
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
        input.len() - 1,
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

    println!("{:<8}{}", "N", input.len() - 1);
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

fn main() {
    let opt = Opt::from_args();

    let raw_inputs = if let Some(path) = opt.input {
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
    };

    let mut data = vec![];

    for (index, line) in raw_inputs.split("\n").enumerate() {
        if index == 0 && opt.with_header {
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

    if opt.line {
        print_line(&data);
    } else if opt.histo {
        print_histo(&mut data, opt.precision);
    } else if opt.transpose {
        print_summary_t(&mut data, opt.precision)
    } else {
        print_summary(&mut data, opt.precision)
    }
}
