use tch::{nn, nn::OptimizerConfig, Device, Tensor};
use image::{DynamicImage, GenericImageView, RgbaImage};
use std::fs::{File};
use std::io::{Read, Write};
use std::path::Path;
use rand::Rng;
use std::ops::Div;

const ALPHA: f64 = 1.0;  // weight for pixel loss
const BETA: f64 = 10.0;  // weight for bit loss

const TRAINING_THRESHOLD: f64 = 0.01; // 1% error
const EPOCHS: i64 = 1_000_000;
const BATCH_SIZE: i64 = 1024;

#[derive(Debug)]
struct SteganographyNN {
    encode_fc1: nn::Linear,
    encode_fc2: nn::Linear,
    encode_fc3: nn::Linear,
    encode_fc4: nn::Linear,
    decode_fc: nn::Linear,
}

impl SteganographyNN {
    fn new(vs: &nn::Path) -> Self {
        let encode_fc1 = nn::linear(vs, 4, 64, Default::default());
        let encode_fc2 = nn::linear(vs, 64, 32, Default::default());
        let encode_fc3 = nn::linear(vs, 32, 16, Default::default());
        let encode_fc4 = nn::linear(vs, 16, 3, Default::default()); // output RGB modified
        let decode_fc = nn::linear(vs, 3, 1, Default::default()); // decode bit
        Self { encode_fc1, encode_fc2, encode_fc3, encode_fc4, decode_fc }
    }

    // Encode one pixel (r,g,b) + bit => modified (r,g,b)
    fn encode(&self, input: &Tensor) -> Tensor {
        input
            .apply(&self.encode_fc1)
            .relu()
            .apply(&self.encode_fc2)
            .relu()
            .apply(&self.encode_fc3)
            .relu()
            .apply(&self.encode_fc4)
            .sigmoid()
            //.mul(255.0)
    }

    // Decode bit from pixel RGB
    fn decode(&self, rgb: &Tensor) -> Tensor {
        rgb.div(255.0)
            .apply(&self.decode_fc)
    }
}

fn train_model(vs: &nn::VarStore, model: &SteganographyNN, epochs: i64, batch_size: i64) {
    let mut opt = nn::Adam::default().build(vs, 1e-3).unwrap();

    for epoch in 0..epochs {
        let mut rng = rand::thread_rng();

        // Batch input generation
        let mut inputs = Vec::with_capacity(batch_size as usize * 4);
        let mut rgb_targets = Vec::with_capacity(batch_size as usize * 3);
        let mut bit_targets = Vec::with_capacity(batch_size as usize);

        for _ in 0..batch_size {
            let r = rng.gen_range(0.0..255.0);
            let g = rng.gen_range(0.0..255.0);
            let b = rng.gen_range(0.0..255.0);
            let bit = rng.gen_range(0..=1) as f32;

            // Normalized input (r, g, b) in [0, 1]
            inputs.extend_from_slice(&[r / 255.0, g / 255.0, b / 255.0, bit]);
            rgb_targets.extend_from_slice(&[r / 255.0, g / 255.0, b / 255.0]);
            bit_targets.push(bit);
        }

        // Convert to tensors
        let input_tensor = Tensor::of_slice(&inputs).view([batch_size, 4]);
        let target_rgb = Tensor::of_slice(&rgb_targets).view([batch_size, 3]);
        let target_bit = Tensor::of_slice(&bit_targets).view([batch_size, 1]);

        // Forward pass
        let modified_rgb = model.encode(&input_tensor);
        let decoded_bits = model.decode(&(&modified_rgb * 255.0)); // multiply without moving

        // Loss calculations
        let pixel_loss = modified_rgb.mse_loss(&target_rgb, tch::Reduction::Mean);
        let bit_loss = decoded_bits.binary_cross_entropy_with_logits::<Tensor>(&target_bit, None, None, tch::Reduction::Mean);
        let total_loss = pixel_loss.shallow_clone() * ALPHA + bit_loss.shallow_clone() * BETA;

        // Backward + optimize
        opt.zero_grad();
        total_loss.backward();
        opt.step();

        if epoch % 100 == 0 {
            println!(
                "Epoch {}: pixel_loss = {:.6}, bit_loss = {:.6}, total_loss = {:.6}",
                epoch,
                f64::from(&pixel_loss),
                f64::from(&bit_loss),
                f64::from(&total_loss)
            );
        }

        if f64::from(&total_loss) < TRAINING_THRESHOLD {
            println!("Early stopping at epoch {}: loss = {:.6}", epoch, f64::from(&total_loss));
            break;
        }
    }
}


/// Embed bits into image pixels via NN, return new image data buffer
fn embed_bits_in_image(
    model: &SteganographyNN,
    img: &DynamicImage,
    bits: &[u8]
) -> RgbaImage {
    let (width, height) = img.dimensions();
    let mut rgba = img.to_rgba8();

    // For each bit, modify pixel using NN
    for (i, &bit) in bits.iter().enumerate() {
        if i as u32 >= width * height {
            break; // no more pixels
        }
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        let mut pixel = *rgba.get_pixel(x, y);
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let bit_f32 = bit as f32;

        let input_tensor = Tensor::of_slice(&[r / 255.0, g / 255.0, b / 255.0, bit_f32])
            .view([1, 4]);

        let modified = model.encode(&input_tensor) * 255.0;
        let modified_cpu = modified.view(-1).detach().to(Device::Cpu);
        let modified_vec: Vec<f32> = Vec::<f32>::from(modified_cpu);
        let modified_rgb: Vec<u8> = modified_vec
            .iter()
            .map(|v| v.clamp(0.0, 255.0) as u8)
            .collect();

        pixel[0] = modified_rgb[0];
        pixel[1] = modified_rgb[1];
        pixel[2] = modified_rgb[2];
        rgba.put_pixel(x, y, pixel);
    }

    rgba
}

/// Decode bits from image pixels via NN
fn decode_bits_from_image(
    model: &SteganographyNN,
    img: &DynamicImage,
    num_bits: usize
) -> Vec<u8> {
    let (width, height) = img.dimensions();
    let rgba = img.to_rgba8();
    let mut bits = Vec::new();

    for i in 0..num_bits {
        if i as u32 >= width * height {
            break;
        }
        let x = (i as u32) % width;
        let y = (i as u32) / width;
        let pixel = rgba.get_pixel(x, y);

        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;

        let rgb_tensor = Tensor::of_slice(&[r, g, b]).view([1, 3]);
        let decoded_bit_tensor = model.decode(&rgb_tensor).sigmoid();
        let decoded_bit = f32::from(decoded_bit_tensor.squeeze());
        bits.push(if decoded_bit > 0.5 { 1u8 } else { 0u8 });
    }
    bits
}

fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8)
        .map(|byte_bits| {
            byte_bits.iter().fold(0u8, |acc, &b| (acc << 1) | b)
        })
        .collect()
}

fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    bytes.iter()
        .flat_map(|b| (0..8).rev().map(move |i| (b >> i) & 1))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vs = nn::VarStore::new(Device::Cpu);
    let model = SteganographyNN::new(&vs.root());

    // Train or load your model
    let model_path = "stego_model.pt";
    if Path::new(model_path).exists() {
        vs.load(model_path)?;
        println!("Loaded trained model");
    } else {
        println!("Training model...");
        train_model(&vs, &model, EPOCHS, BATCH_SIZE);
        vs.save(model_path)?;
        println!("Model trained and saved.");
    }

    // Load cover image
    let cover_img_path = "bird.jpeg";  // your input image file path
    let cover_img = image::open(cover_img_path)?;

    // Read secret file to embed
    let secret_path = "secret.txt";  // your secret file
    let mut secret_file = File::open(secret_path)?;
    let mut secret_data = Vec::new();
    secret_file.read_to_end(&mut secret_data)?;

    // Convert secret data to bits
    let bits = bytes_to_bits(&secret_data);

    // Check capacity
    let (width, height) = cover_img.dimensions();
    if bits.len() > (width * height) as usize {
        panic!("Secret file too large for the image.");
    }

    // Embed bits into image
    let stego_img = embed_bits_in_image(&model, &cover_img, &bits);

    // Save stego image
    let stego_path = "stego_output.png";
    stego_img.save(stego_path)?;
    println!("Stego image saved to {}", stego_path);

    // --- Decode phase ---

    let stego_img_loaded = image::open(stego_path)?;

    // Decode bits
    let decoded_bits = decode_bits_from_image(&model, &stego_img_loaded, bits.len());

    // Convert bits back to bytes
    let decoded_bytes = bits_to_bytes(&decoded_bits);

    // Save recovered secret
    let recovered_path = "recovered_secret.txt";
    let mut recovered_file = File::create(recovered_path)?;
    recovered_file.write_all(&decoded_bytes)?;

    println!("Recovered secret saved to {}", recovered_path);

    Ok(())
}
