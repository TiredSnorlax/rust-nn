use std::{
    error::Error,
    fmt::Display,
    fs::File,
    io::{BufRead, BufReader},
};

use matrix::matrix::Matrix;

#[derive(Debug, PartialEq, Clone)]
pub enum DataType {
    Binary(f64),
    OneHot(Vec<f64>),
    Continuous(f64),
    String(String),
}

impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::Binary(v) => write!(f, "B({})", v),
            DataType::OneHot(items) => {
                write!(f, "OH({})", items.iter().position(|x| *x == 1.).unwrap())
            }
            DataType::Continuous(v) => write!(f, "F({})", v),
            DataType::String(s) => write!(f, "S({})", s),
        }
    }
}

impl DataType {
    pub fn into_vec(&self) -> Result<Vec<f64>, Box<dyn Error>> {
        match self {
            DataType::Binary(v) => Ok(vec![*v]),
            DataType::OneHot(items) => Ok(items.clone()),
            DataType::Continuous(v) => Ok(vec![*v]),
            DataType::String(_) => Err("Unable to convert string into float values".into()),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum FeatureTypes<'a> {
    // first string for 0, second string for 1
    Binary(&'a str, &'a str),
    // number of categories
    OneHot(usize, Vec<&'a str>),
    Continuous,
    String,
}

pub struct Dataframe {
    pub features: Vec<Vec<DataType>>,
    pub feature_names: Vec<String>,
    pub targets: Vec<DataType>,
    pub target_name: String,
}

impl Dataframe {
    pub fn from_file(
        file_path: &str,
        names: Vec<&str>,
        target_index: usize,
        feature_types: Vec<FeatureTypes>,
        sep: &str,
        drop_unknown: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let mut names: Vec<String> = names.iter().map(|s| s.to_string()).collect();

        let mut rows: Vec<Vec<DataType>> = Vec::new();
        let mut targets: Vec<DataType> = Vec::new();

        'outer: for line in reader.lines() {
            let line = line?;
            let strings: Vec<String> = match sep {
                " " => line.split_whitespace().map(|s| s.to_string()).collect(),
                _ => line.split(sep).map(|s| s.to_string()).collect(),
            };
            let mut row: Vec<DataType> = Vec::with_capacity(names.len());
            for i in 0..names.len() {
                let res = Self::convert_to_feature(&feature_types[i], &strings[i], &names[i]);
                if res.is_err() && drop_unknown {
                    continue 'outer;
                } else if res.is_err() && !drop_unknown {
                    return Err(res.unwrap_err());
                } else {
                    row.push(res.unwrap())
                }
            }

            let target = row.swap_remove(target_index);
            targets.push(target);
            rows.push(row);
        }

        // Remove the target after processing the features as the logic is dependent on the original indexing
        let target_name = names.swap_remove(target_index);
        Ok(Dataframe {
            features: rows,
            targets,
            feature_names: names,
            target_name,
        })
    }

    pub fn drop_col(&mut self, col_index: usize) {
        for row in &mut self.features {
            row.remove(col_index);
        }
        self.feature_names.remove(col_index);
    }

    fn convert_to_feature(
        feature_type: &FeatureTypes,
        data: &str,
        feature_name: &str,
    ) -> Result<DataType, Box<dyn Error>> {
        let data_type = match feature_type {
            FeatureTypes::Binary(class_zero, class_one) => {
                if data == *class_zero {
                    DataType::Binary(0.0)
                } else if data == *class_one {
                    DataType::Binary(1.0)
                } else {
                    return Err(format!(
                        "Found value ({}) in {} that cannot be converted to binary ({}/{})). Reconsider datatype for this feature",
                        data, feature_name, class_zero, class_one
                    ).into());
                }
            }
            FeatureTypes::OneHot(num_cat, categories) => {
                if let Some(index) = categories.iter().position(|c| *c == data) {
                    let mut encoding = vec![0.0; *num_cat];
                    encoding[index] = 1.;
                    DataType::OneHot(encoding)
                } else {
                    return Err(format!(
                        "Found value ({}) in {} that cannot be converted to one-hot encoding ({} categories). Check number of categories again.",
                        data, feature_name, num_cat
                    ).into());
                }
            }
            FeatureTypes::Continuous => {
                if let Ok(value) = data.parse::<f64>() {
                    DataType::Continuous(value)
                } else {
                    return Err(format!(
                        "Found value ({}) in {} that cannot be converted into float. Recheck possible values for this feature.",
                        data, feature_name
                    ).into());
                }
            }
            FeatureTypes::String => DataType::String(data.to_string()),
        };

        Ok(data_type)
    }

    pub fn show_example(&self, index: usize) {
        let features = &self.features[index];
        let target = &self.targets[index];

        let labels = &self.feature_names;

        println!(
            "{}, Target ({}): {}",
            features
                .iter()
                .zip(labels)
                .map(|(f, l)| format!("{}: {}", l, f))
                .collect::<Vec<_>>()
                .join(", "),
            self.target_name,
            target,
        )
    }

    /// Split dataframe into 2 dataframes by ratio using random sampling.
    pub fn split(&self, ratio: f64) -> (Self, Self) {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..self.features.len()).collect();
        let mut rng = rand::rng();
        indices.shuffle(&mut rng);

        let split_index = (self.features.len() as f64 * ratio) as usize;
        let (indices1, indices2) = indices.split_at(split_index);

        let mut f1 = Vec::with_capacity(indices1.len());
        let mut t1 = Vec::with_capacity(indices1.len());
        for &i in indices1 {
            f1.push(self.features[i].clone());
            t1.push(self.targets[i].clone());
        }

        let mut f2 = Vec::with_capacity(indices2.len());
        let mut t2 = Vec::with_capacity(indices2.len());
        for &i in indices2 {
            f2.push(self.features[i].clone());
            t2.push(self.targets[i].clone());
        }

        (
            Self {
                features: f1,
                feature_names: self.feature_names.clone(),
                targets: t1,
                target_name: self.target_name.clone(),
            },
            Self {
                features: f2,
                feature_names: self.feature_names.clone(),
                targets: t2,
                target_name: self.target_name.clone(),
            },
        )
    }

    /// Split dataframe into batches of size `batch_size`.
    pub fn batch(&self, batch_size: usize) -> Vec<Self> {
        let mut batches: Vec<Self> = Vec::new();

        let features: Vec<Vec<Vec<DataType>>> = self
            .features
            .chunks(batch_size)
            .map(|s| s.to_vec())
            .collect();

        let targets: Vec<Vec<DataType>> = self
            .targets
            .chunks(batch_size)
            .map(|s| s.to_vec())
            .collect();

        for (f, t) in features.iter().zip(targets.iter()) {
            batches.push(Self {
                features: f.clone(),
                feature_names: self.feature_names.clone(),
                targets: t.clone(),
                target_name: self.target_name.clone(),
            });
        }

        batches
    }

    /// This will return a 2 Matrices, inputs and targets.
    /// Matrix of shape (number of examples, number of features)
    pub fn convert_to_matrix(&self) -> Result<(Matrix, Matrix), Box<dyn Error>> {
        // Inputs
        let mut input_data: Vec<f64> = Vec::new();
        for row in &self.features {
            input_data.extend(row.iter().map(|f| f.into_vec().unwrap()).flatten());
        }

        // This will account for the one-hot encoding increasing the length.
        let input_cols = if self.features.is_empty() {
            0
        } else {
            self.features[0]
                .iter()
                .map(|f| f.into_vec().unwrap().len())
                .sum()
        };

        let inputs = Matrix::from(self.features.len(), input_cols, input_data);

        // Targets
        let mut target_data: Vec<f64> = Vec::new();

        for target in &self.targets {
            target_data.extend(target.into_vec().unwrap());
        }
        let target_cols = if self.targets.is_empty() {
            0
        } else {
            self.targets[0].into_vec()?.len()
        };

        let targets = Matrix::from(self.targets.len(), target_cols, target_data);

        Ok((inputs, targets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_to_feature() {
        // Binary
        let ft_binary = FeatureTypes::Binary("no", "yes");
        assert_eq!(
            Dataframe::convert_to_feature(&ft_binary, "no", "test").unwrap(),
            DataType::Binary(0.0)
        );
        assert_eq!(
            Dataframe::convert_to_feature(&ft_binary, "yes", "test").unwrap(),
            DataType::Binary(1.0)
        );

        // OneHot
        let ft_oh = FeatureTypes::OneHot(3, vec!["red", "green", "blue"]);
        assert_eq!(
            Dataframe::convert_to_feature(&ft_oh, "green", "test").unwrap(),
            DataType::OneHot(vec![0.0, 1.0, 0.0])
        );

        // Continuous
        let ft_cont = FeatureTypes::Continuous;
        assert_eq!(
            Dataframe::convert_to_feature(&ft_cont, "123.45", "test").unwrap(),
            DataType::Continuous(123.45)
        );
    }

    #[test]
    fn test_convert_to_matrix() {
        let features = vec![
            vec![
                DataType::Continuous(1.0),
                DataType::Binary(0.0),
                DataType::OneHot(vec![1.0, 0.0, 0.0]),
            ],
            vec![
                DataType::Continuous(2.0),
                DataType::Binary(1.0),
                DataType::OneHot(vec![0.0, 0.0, 1.0]),
            ],
        ];

        let df = Dataframe {
            features,
            feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
            targets: vec![DataType::Binary(0.0), DataType::Binary(1.0)],
            target_name: "target".to_string(),
        };

        let (input, target) = df.convert_to_matrix().unwrap();
        assert_eq!(input.rows, 2);
        assert_eq!(input.cols, 5); // 1 (cont) + 1 (binary) + 3 (one-hot)
        assert_eq!(
            input.data,
            vec![
                1.0, 0.0, 1.0, 0.0, 0.0, // row 0
                2.0, 1.0, 0.0, 0.0, 1.0 // row 1
            ]
        );

        assert_eq!(target.rows, 2);
        assert_eq!(target.cols, 1);
        assert_eq!(target.data, vec![0.0, 1.0]);
    }

    #[test]
    fn test_split() {
        let features: Vec<Vec<DataType>> = (0..10)
            .map(|i| vec![DataType::Continuous(i as f64)])
            .collect();
        let targets: Vec<DataType> = (0..10).map(|i| DataType::Continuous(i as f64)).collect();

        let df = Dataframe {
            features: features.clone(),
            feature_names: vec!["f1".to_string()],
            targets: targets.clone(),
            target_name: "target".to_string(),
        };

        let (df1, df2) = df.split(0.7);

        assert_eq!(df1.features.len(), 7);
        assert_eq!(df2.features.len(), 3);

        // Check all original values are still present
        let mut combined_features = df1.features.clone();
        combined_features.extend(df2.features.clone());
        combined_features.sort_by(|a, b| {
            if let (DataType::Continuous(v1), DataType::Continuous(v2)) = (&a[0], &b[0]) {
                v1.partial_cmp(v2).unwrap()
            } else {
                std::cmp::Ordering::Equal
            }
        });
        assert_eq!(combined_features, features);
    }

    #[test]
    fn test_batch() {
        let features: Vec<Vec<DataType>> = (0..5)
            .map(|i| vec![DataType::Continuous(i as f64)])
            .collect();
        let targets: Vec<DataType> = (0..5).map(|i| DataType::Continuous(i as f64)).collect();

        let df = Dataframe {
            features,
            feature_names: vec!["f1".to_string()],
            targets,
            target_name: "target".to_string(),
        };

        let batches = df.batch(2);

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].features.len(), 2);
        assert_eq!(batches[1].features.len(), 2);
        assert_eq!(batches[2].features.len(), 1);

        assert_eq!(batches[0].features[0][0], DataType::Continuous(0.0));
        assert_eq!(batches[1].features[0][0], DataType::Continuous(2.0));
        assert_eq!(batches[2].features[0][0], DataType::Continuous(4.0));
    }

    #[test]
    fn test_from_file_unknown_values() -> Result<(), Box<dyn Error>> {
        use std::fs;
        let file_path = "test_unknown.csv";
        let content = "1.0,red,yes\nbad_val,blue,no\n2.0,green,yes\n3.0,bad_cat,no";
        fs::write(file_path, content)?;

        let names = vec!["f1", "f2", "target"];
        let feature_types = vec![
            FeatureTypes::Continuous,
            FeatureTypes::OneHot(3, vec!["red", "green", "blue"]),
            FeatureTypes::Binary("no", "yes"),
        ];

        // Case 1: drop_unknown = true
        let df_dropped = Dataframe::from_file(
            file_path,
            names.clone(),
            2,
            feature_types.clone(),
            ",",
            true,
        )?;
        // Should have only 2 rows (row 0 and row 2)
        assert_eq!(df_dropped.features.len(), 2);
        assert_eq!(df_dropped.features[0][0], DataType::Continuous(1.0));
        assert_eq!(df_dropped.features[1][0], DataType::Continuous(2.0));

        // Case 2: drop_unknown = false
        let df_error = Dataframe::from_file(file_path, names, 2, feature_types, ",", false);
        assert!(df_error.is_err());

        fs::remove_file(file_path)?;
        Ok(())
    }
}
