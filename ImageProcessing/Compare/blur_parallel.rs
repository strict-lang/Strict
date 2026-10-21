// Same 9x9 blur as blur.rs, the rows split over all cores with scoped threads, rustc -C opt-level=3 -C target-cpu=native blur_parallel.rs
use std::time::Instant;
fn main() {
	let file = std::fs::read("Landscape.ppm").unwrap();
	let header = String::from_utf8_lossy(&file[..40]).to_string();
	let fields: Vec<&str> = header.split_whitespace().collect();
	let (width, height): (i32, i32) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
	let rgb = &file[file.len() - (width * height * 3) as usize..];
	let data: Vec<u8> = rgb.chunks(3).flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255]).collect();
	let mut blurred = vec![0u8; (width * height * 4) as usize];
	let threads = std::thread::available_parallelism().unwrap().get();
	let rows_per_thread = (height as usize + threads - 1) / threads;
	let started = Instant::now();
	std::thread::scope(|scope| {
		for (part, rows) in blurred.chunks_mut(rows_per_thread * width as usize * 4).enumerate() {
			let data = &data;
			scope.spawn(move || {
				for (offset, row) in rows.chunks_mut(width as usize * 4).enumerate() {
					let y = (part * rows_per_thread + offset) as i32;
					for x in 0..width {
						let (mut red, mut green, mut blue, mut alpha, mut count) = (0u32, 0u32, 0u32, 0u32, 0u32);
						for dy in -4..=4 {
							for dx in -4..=4 {
								if x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height {
									let source_index = (((y + dy) * width + x + dx) * 4) as usize;
									red += data[source_index] as u32;
									green += data[source_index + 1] as u32;
									blue += data[source_index + 2] as u32;
									alpha += data[source_index + 3] as u32;
									count += 1;
								}
							}
						}
						let target_index = (x * 4) as usize;
						row[target_index] = (red / count) as u8;
						row[target_index + 1] = (green / count) as u8;
						row[target_index + 2] = (blue / count) as u8;
						row[target_index + 3] = (alpha / count) as u8;
					}
				}
			});
		}
	});
	println!("Rust parallel: blurred {}x{} in {:.1} ms", width, height, started.elapsed().as_secs_f64() * 1000.0);
	let mut output = format!("P6\n{} {}\n255\n", width, height).into_bytes();
	output.extend(blurred.chunks(4).flat_map(|pixel| [pixel[0], pixel[1], pixel[2]]));
	std::fs::write("Landscape.rust_parallel.ppm", output).unwrap();
}
