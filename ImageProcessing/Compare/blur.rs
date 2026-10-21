// Same blur as BenchBlur.strict on Landscape.ppm, rustc -C opt-level=3 -C target-cpu=native blur.rs
use std::time::Instant;
fn main() {
	let file = std::fs::read("Landscape.ppm").unwrap();
	let mut fields = Vec::new();
	let mut start = 0;
	while fields.len() < 4 {
		while file[start].is_ascii_whitespace() { start += 1; }
		let end = start + file[start..].iter().position(|byte| byte.is_ascii_whitespace()).unwrap();
		fields.push(String::from_utf8_lossy(&file[start..end]).to_string());
		start = end;
	}
	let (width, height): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
	let (length, stride, radius) = (width * height * 3, width * 3, 4);
	let pixels = &file[start + 1..start + 1 + length];
	let mut blurred = vec![0u8; length];
	let area = ((2 * radius + 1) * (2 * radius + 1)) as u32;
	let started = Instant::now();
	for y in 0..height {
		for x in 0..width {
			for channel in 0..3 {
				let index = y * stride + x * 3 + channel;
				if x < radius || x >= width - radius || y < radius || y >= height - radius {
					blurred[index] = pixels[index];
				} else {
					let mut sum = 0u32;
					for dy in 0..2 * radius + 1 {
						for dx in 0..2 * radius + 1 {
							sum += pixels[index + dy * stride + dx * 3 - radius * stride - radius * 3] as u32;
						}
					}
					blurred[index] = (sum / area) as u8;
				}
			}
		}
	}
	println!("Rust: blurred {}x{} in {:.1} ms", width, height, started.elapsed().as_secs_f64() * 1000.0);
	let mut output = format!("P6\n{} {}\n255\n", width, height).into_bytes();
	output.extend_from_slice(&blurred);
	std::fs::write("Landscape.rust.ppm", output).unwrap();
}
