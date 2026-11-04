//! Champion model serialization.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::eval::Aggregate;
use crate::nn::Net;

pub const MODEL_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMeta {
    pub training_seed: u64,
    pub generation: u32,
    pub validation: Option<Aggregate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFile {
    pub version: u32,
    pub sizes: Vec<usize>,
    pub params: Vec<f32>,
    pub meta: ModelMeta,
}

impl ModelFile {
    pub fn from_net(net: &Net, meta: ModelMeta) -> Self {
        Self {
            version: MODEL_VERSION,
            sizes: net.sizes.clone(),
            params: net.params.clone(),
            meta,
        }
    }

    pub fn net(&self) -> Net {
        Net {
            sizes: self.sizes.clone(),
            params: self.params.clone(),
        }
    }
}

pub fn save(path: &Path, model: &ModelFile) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let json = serde_json::to_string_pretty(model).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))?;
    Ok(())
}

pub fn load(path: &Path) -> Result<ModelFile, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read model {}: {e}", path.display()))?;
    let model: ModelFile =
        serde_json::from_str(&text).map_err(|e| format!("parse model {}: {e}", path.display()))?;
    if model.version != MODEL_VERSION {
        return Err(format!(
            "model version {} not supported (expected {MODEL_VERSION})",
            model.version
        ));
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::ForwardPass;
    use crate::rng::Rng;

    #[test]
    fn save_load_roundtrip() {
        let mut rng = Rng::new(1);
        let net = Net::random(&[3, 4, 2], &mut rng);
        let file = ModelFile::from_net(
            &net,
            ModelMeta {
                training_seed: 7,
                generation: 3,
                validation: None,
            },
        );
        let dir = std::env::temp_dir().join("racer_model_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.json");
        save(&path, &file).unwrap();
        let back = load(&path).unwrap();
        let input = [0.2, -0.3, 0.9];
        let mut p1 = ForwardPass::new();
        let mut p2 = ForwardPass::new();
        assert_eq!(
            net.forward(&input, &mut p1),
            back.net().forward(&input, &mut p2)
        );
        assert_eq!(back.meta.generation, 3);
        let _ = std::fs::remove_file(path);
    }
}
