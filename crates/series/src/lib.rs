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
        let mut s = Series::default();
        s.data = input;
        s
    }

    pub fn stats(&mut self) {
        if self.data.len() == 0 {
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
