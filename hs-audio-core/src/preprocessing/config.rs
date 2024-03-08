use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct AudioPreprocessingConfig {
    size: usize
}