use std::collections::HashMap;
use std::path::PathBuf;
use structopt::StructOpt;
use xgboost::{parameters, Booster, DMatrix};

pub fn to_xgboost_dataset(xdata: &Vec<Vec<f64>>, ydata: Option<Vec<f32>>) -> DMatrix {
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

pub fn dump_model(model: &[u8], dump_model: bool, typ: &str) {
    if dump_model {
        let bst = Booster::load_buffer(model).unwrap();
        println!("{}", bst.dump_model(true, None).unwrap());
    } else {
        let bst = Booster::load_buffer(model).unwrap();
        let model = bst.dump_model(true, None).unwrap();
        importance(model, &typ);
    }
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

pub fn importance(model_dump: String, typ: &str) {
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

pub fn predict(model: &str, test_set: &DMatrix) -> Vec<f32> {
    let bst = Booster::load(model).unwrap();
    bst.predict(&test_set).unwrap()
}

pub fn train(
    training_set: &DMatrix,
    objective: &str,
    nclasses: u32,
    depth: u32,
    eta: f32,
    rounds: u32,
    output: &str,
) {
    let objective_fn = match objective {
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
