use std::collections::HashMap;

use std::str::FromStr;

#[derive(Default)]
pub struct Series {
    pub data: Vec<f64>,
    pub mean: f64,
    pub median: f64,
    pub stdev: f64,
    pub var: f64,
    pub min: f64,
    pub max: f64,
}

impl Series {
    pub fn new(input: Vec<f64>) -> Self {
        Series {
            data: input,
            ..Series::default()
        }
    }

    pub fn stats(&mut self) {
        if self.data.is_empty() {
            return;
        }

        self.median = if self.data.len() > 1 {
            let mut temp = self.data.clone();
            temp.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = self.data.len() as f64 / 2.0;
            self.data[mid as usize]
        } else {
            0.0
        };

        self.min = self.data[0];
        self.max = self.data[0];

        for x in &self.data {
            if *x > self.max {
                self.max = *x;
                continue;
            }

            if *x < self.min {
                self.min = *x;
            }
        }

        self.mean = self.data.iter().sum::<f64>() as f64 / self.data.len() as f64;

        let sum: f64 = self
            .data
            .iter()
            .map(|x| (*x - self.mean).powf(2.0))
            .collect::<Vec<f64>>()
            .iter()
            .sum();

        self.var = sum / self.data.len() as f64;
        self.stdev = (sum / self.data.len() as f64).sqrt();
    }

    pub fn summary(mut self) {
        self.stats();

        println!(
            "{:<11}{:<11}{:<11}{:<11}{:<11}{:<11}{:<11}",
            "n", "min", "max", "mean", "median", "sd", "var"
        );
        println!(
            "{:<11}{:<11.4}{:<11.4}{:<11.4}{:<11.4}{:<11.4}{:<11.4}",
            self.data.len(),
            self.min,
            self.max,
            self.mean,
            self.median,
            self.stdev,
            self.var
        );
    }

    pub fn summary_t(mut self) {
        self.stats();

        println!("{:<8}{:<8}", "N", self.data.len());
        println!("{:<8}{:<8.4}", "min", self.min);
        println!("{:<8}{:<8.4}", "max", self.max);
        println!("{:<8}{:<8.4}", "mean", self.mean);
        println!("{:<8}{:<8.4}", "med", self.median);
        println!("{:<8}{:<8.4}", "stdev", self.stdev);
        println!("{:<8}{:<8.4}", "var", self.var);
    }
}

pub fn confusion_matrix(tuples: &[(f32, f32)], threshold: Option<f32>) -> Vec<Vec<u32>> {
    let mut classes = HashMap::new();

    for (_, c) in tuples {
        // f32 is not hashable, convert to string
        classes.insert(format!("{}", c), 1);
    }

    let size = classes.keys().len();
    let mut matrix = vec![];

    // create the confusion matrix
    for _ in 0..size {
        matrix.push(vec![0; size]);
    }

    // intended to use only with binary (0,1) ranges. Not softprob (yet).
    for (p, actual_class_col) in tuples {
        let predicted_class_row = if let Some(t) = threshold {
            (*p + (1.0 - t)) as usize
        } else if size == 2 {
            (*p + 0.5) as usize
        } else {
            *p as usize
        };

        matrix[predicted_class_row][*actual_class_col as usize] += 1;
    }

    // reverse each row so they are in descending order
    // after this the matrix is in descending order from top/left to bottom/right
    // the purpose for ordering this way is for a binary prediction this defautt layout
    // matches a confusion matrix
    // TP FP
    // FN TN
    for i in 0..size {
        matrix[i].reverse();
    }
    matrix.reverse();

    matrix
}

pub struct CMatrixStats {
    pub label: usize,
    pub fpr: f32,
    pub tpr: f32,
    pub fnr: f32,
    pub tnr: f32,
}

pub fn confusion_matrix_stats(matrix: &[Vec<u32>]) -> Vec<CMatrixStats> {
    let size = matrix.len();

    let mut counts = HashMap::new();
    for i in 0..size {
        // TP FN FP TN
        counts.insert(i, vec![0.0, 0.0, 0.0, 0.0]);
    }

    let mut total = 0.0;

    for i in 0..size {
        let mut fn_i = 0.0;

        for j in 0..size {
            // total the entire matrix
            total += matrix[i][j] as f32;

            // TP
            if i == j {
                let data = counts.get_mut(&i).unwrap();
                data[0] = matrix[i][j] as f32;
                continue;
            }

            let data = counts.get_mut(&i).unwrap();
            // FP
            data[2] += matrix[j][i] as f32;

            // FN
            fn_i += matrix[i][j] as f32;
        }

        let data = counts.get_mut(&i).unwrap();
        // FN
        data[1] = fn_i;
    }

    let mut stats = vec![];

    for k in 0..size {
        let v = counts.get_mut(&k).unwrap();
        // TN
        v[3] = total - v[0] - v[1] - v[2];

        stats.push(CMatrixStats {
            label: k,
            fpr: v[2] / (v[2] + v[3]),
            tpr: v[0] / (v[0] + v[1]),
            fnr: v[1] / (v[1] + v[0]),
            tnr: v[3] / (v[3] + v[1]),
        })
    }

    stats.sort_by(|a, b| a.label.cmp(&b.label));

    stats
}

pub fn threshold_table_stats(tuples: &[(f32, f32)]) -> Vec<Vec<f32>> {
    let mut out = vec![];
    let mut t = 0.05;

    loop {
        if t > 1.0 {
            break;
        }

        let mut ttp: f32 = 0.0;
        let mut tfp: f32 = 0.0;
        let mut tfn: f32 = 0.0;
        let mut ttn: f32 = 0.0;

        for (p, a) in tuples {
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

        let precision = ttp / (ttp + tfp);
        let tpr = ttp / (ttp + tfn);
        let f1 = 2.0 * (tpr * precision) / (tpr + precision);
        let fpr = tfp / (tfp + ttn);

        out.push(vec![t, precision, tpr, f1, fpr]);

        t += 0.05;
    }

    out
}

pub fn correlation_matrix(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if input.is_empty() {
        eprintln!("input must be a non empty set");
        std::process::exit(1);
    }

    let rows = input.len();
    let cols = input[0].len();

    let mut out = vec![];

    let mut sets = HashMap::new();
    for i in 0..cols {
        // add a row for each class in the out vector
        out.push(vec![0.0; cols]);
        sets.insert(i, Series::new(vec![]));
    }

    // populate the sets map
    for row in input {
        for (index, col) in row.iter().enumerate() {
            let series = sets.get_mut(&index).unwrap();
            series.data.push(*col);
        }
    }

    // now calculate basic stats of each set
    for i in 0..cols {
        let series = sets.get_mut(&i).unwrap();
        series.stats();
    }

    // for each index calculate its cor with every other index
    for i in 0..cols {
        for j in i..cols {
            let mut sum = 0.0;
            let mut xs = 0.0;
            let mut ys = 0.0;

            let set_x = sets.get(&i).unwrap();
            let set_y = sets.get(&j).unwrap();

            for k in 0..rows {
                sum += set_x.data[k] * set_y.data[k];
                xs += set_x.data[k].powf(2.0);
                ys += set_y.data[k].powf(2.0);
            }

            sum -= (rows as f64) * set_x.mean * set_y.mean;
            let dem = (xs - (rows as f64) * set_x.mean.powf(2.0)).sqrt()
                * (ys - (rows as f64) * set_y.mean.powf(2.0)).sqrt();

            let r_xy = sum / dem;
            out[j][i] = r_xy;
        }
    }

    out
}

pub fn str_to_vector<F: FromStr>(s: &str, sep: &str) -> Result<Vec<F>, <F as FromStr>::Err> {
    let mut out = vec![];
    for i in s.split(sep).into_iter() {
        match i.trim().parse() {
            Ok(f) => {
                out.push(f);
            }

            Err(e) => {
                return Err(e);
            }
        }
    }

    Ok(out)
}

pub fn to_tuple(input: &str) -> Vec<(f32, f32)> {
    let mut data = vec![];

    for line in input.split('\n') {
        if line == "\n" || line.is_empty() {
            continue;
        }

        let cols = line.split(',').collect::<Vec<&str>>();
        if cols.len() != 2 {
            eprintln!("warning: invalid column count for parse_tuple: {}", line);
            continue;
        }

        let temp: Result<f32, _> = cols[0].trim().parse();
        let v1 = match temp {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        };

        let temp: Result<f32, _> = cols[1].trim().parse();
        let v2 = match temp {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        };

        data.push((v1, v2));
    }

    data
}

pub fn to_vector(raw_inputs: &str, with_header: bool) -> Vec<f64> {
    let mut data = vec![];

    for (index, line) in raw_inputs.split('\n').enumerate() {
        if index == 0 && with_header {
            continue;
        }

        if line == "\n" || line.is_empty() {
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

/// to_matrix parses a input and builds a Matrix. If 'ycol' is a valid column
/// index, that column will be held out and used as the labels for the Matrix.
pub fn to_matrix(raw_inputs: &str, ycol: usize, with_header: bool) -> (Vec<Vec<f64>>, Vec<f32>) {
    let mut xdata = Vec::new();
    let mut ydata = Vec::new();

    for (index, line) in raw_inputs.split('\n').enumerate() {
        if index == 0 && with_header {
            continue;
        }

        if line == "\n" || line.is_empty() {
            continue;
        }

        let split = line.split(',');

        let mut row = vec![];

        for (index, val) in split.enumerate() {
            let temp: Result<f64, _> = val.trim().parse();
            match temp {
                Ok(f) => {
                    if index == ycol {
                        ydata.push(f as f32)
                    } else {
                        row.push(f)
                    }
                }
                Err(_) => {
                    eprintln!("error converting to float: {} at line {}", line, index);
                    std::process::exit(1);
                }
            }
        }

        xdata.push(row);
    }

    (xdata, ydata)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_corrm() {
        let input = vec![
            vec![45.0, 38.0, 10.0],
            vec![37.0, 31.0, 15.0],
            vec![42.0, 26.0, 17.0],
            vec![35.0, 28.0, 21.0],
            vec![39.0, 33.0, 12.0],
        ];

        let m = correlation_matrix(&input);

        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[1][0], 0.5184570956392384);
        assert_eq!(m[2][0], -0.7018864176470834);
        assert_eq!(m[2][1], -0.860940956122431);
    }
}
