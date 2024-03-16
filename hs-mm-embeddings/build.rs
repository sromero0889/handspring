use std::path::Path;

struct RemoteModel {
    pub url: &'static str,
    pub name: &'static str,
    pub modality: &'static str,
    pub filename: &'static str,
}

const M_CLIP_VIT_B_32_IMAGE: RemoteModel = RemoteModel {
    url: "https://github.com/sromero0889/model_clip_vit_b_32/raw/main/image/model.safetensors",
    name: "clip_vit_b_32",
    modality: "image",
    filename: "model.safetensors",
};

fn download_model_file(model_info: &RemoteModel) {
    use std::{fs::File, fs::create_dir_all, io::copy};
    let dir = Path::new("models").join(model_info.name).join(model_info.modality);

    let dst_path = &dir.join(model_info.filename);
    if !dst_path.exists() {
        println!("download_model {}", model_info.name);
        let mut response = reqwest::blocking::get(model_info.url).unwrap();

        create_dir_all(&dir).unwrap();
        let mut file = File::create(dst_path).unwrap();
        copy(&mut response, &mut file).unwrap();
    }
}
// Important: unwrap use here for now: Build does not make any sense without these files, I want it to panic
// todo improve errors before expose lib

fn main() {
    if cfg!(feature = "clip_vit_b_32_image") {
        download_model_file(&M_CLIP_VIT_B_32_IMAGE);
    };
    // TODO add condition here to download model every time you add a new one

}