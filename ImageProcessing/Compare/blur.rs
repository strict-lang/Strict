// Same blur as BenchBlur.strict and blur.c, usage: blur width height radius
// rustc -C opt-level=3 -C target-cpu=native blur.rs
use std::time::Instant;
fn main() {
	let arguments: Vec<usize> = std::env::args().skip(1).map(|argument| argument.parse().unwrap()).collect();
	let (width, height, radius) = (*arguments.get(0).unwrap_or(&3840), *arguments.get(1).unwrap_or(&2160), *arguments.get(2).unwrap_or(&4));
	let (length, stride) = (width * height * 4, width * 4);
	let pixels: Vec<u8> = (0..length).map(|index| (index * 7919 % 256) as u8).collect();
	let mut blurred = vec![0u8; length];
	let area = ((2 * radius + 1) * (2 * radius + 1)) as u32;
	let started = Instant::now();
	for y in 0..height {
		for x in 0..width {
			for channel in 0..4 {
				let index = y * stride + x * 4 + channel;
				if x < radius || x >= width - radius || y < radius || y >= height - radius {
					blurred[index] = pixels[index];
				} else {
					let mut sum = 0u32;
					for dy in 0..2 * radius + 1 {
						for dx in 0..2 * radius + 1 {
							sum += pixels[index + dy * stride + dx * 4 - radius * stride - radius * 4] as u32;
						}
					}
					blurred[index] = (sum / area) as u8;
				}
			}
		}
	}
	println!("Rust: blurred {}x{} with {}x{} in {:.1} ms, pixel {}", width, height, 2 * radius + 1, 2 * radius + 1,
		started.elapsed().as_secs_f64() * 1000.0, blurred[length / 2 + stride / 2]);
}
