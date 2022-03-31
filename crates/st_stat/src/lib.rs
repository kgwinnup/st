use itertools::Itertools;
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rasciigraph;
use std::collections::HashMap;

pub fn median(input: &mut [f64]) -> f64 {
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

pub fn mode(input: &[f64], prec: u32) -> f64 {
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

pub fn mean(input: &[f64]) -> f64 {
    if input.len() == 0 {
        return 0.0;
    }

    if input.len() == 1 {
        return input[0];
    }

    input.iter().sum::<f64>() as f64 / input.len() as f64
}

/// (stdev, var, mean)
pub fn stdev_var_mean(input: &[f64]) -> (f64, f64, f64) {
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

pub fn print_line(input: &[f64]) {
    let config = rasciigraph::Config::default()
        .with_height(20)
        .with_width(70)
        .with_offset(0);
    let plot = rasciigraph::plot(input.to_vec(), config);
    println!("{}", plot);
}

pub fn print_histo(input: &mut [f64], prec: u32) {
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

pub fn sample(raw_inputs: &str, with_header: bool, size: u32, with_replacement: bool) {
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

pub fn print_summary(input: &mut [f64], prec: u32) {
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

pub fn print_summary_t(input: &mut [f64], prec: u32) {
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
