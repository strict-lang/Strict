// Same 9x9 blur as BenchBlur.strict on Landscape.ppm, goose -O2 -o blur_goose.c blur.goose, then clang -O3 -march=native
import std;
import os;

fn number(text: u8[>..]&, start: i64) -> i64 {
    var value = 0;
    var index = start;
    while text[index] >= '0' && text[index] <= '9' {
        value = value * 10 + (text[index] - '0');
        index++;
    }
    value
}

fn skip(text: u8[>..]&, start: i64) -> i64 {
    var index = start;
    while text[index] >= '0' && text[index] <= '9' { index++; }
    index + 1
}

fn main() {
    var file: u8[>..] = [];
    guard read_file("Landscape.ppm", file) else { print("Missing Landscape.ppm"); exit(1); }
    let width = number(file, 3);
    let height = number(file, skip(file, 3));
    let rgb = file.len - width * height * 3;
    var data: u8[>..] = [];
    var blurred: u8[>..] = [];
    for pixel in width * height {
        for channel in 3 { data.push(file[rgb + pixel * 3 + channel]); }
        data.push(255);
        for channel in 4 { blurred.push(0); }
    }
    let started = clock_ns();
    for y in height {
        for x in width {
            var red = 0;
            var green = 0;
            var blue = 0;
            var alpha = 0;
            var count = 0;
            for dy in -4..5 {
                for dx in -4..5 {
                    if x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height {
                        let sourceIndex = ((y + dy) * width + x + dx) * 4;
                        red += data[sourceIndex];
                        green += data[sourceIndex + 1];
                        blue += data[sourceIndex + 2];
                        alpha += data[sourceIndex + 3];
                        count++;
                    }
                }
            }
            let targetIndex = (y * width + x) * 4;
            blurred[targetIndex] = (red / count) as! u8;
            blurred[targetIndex + 1] = (green / count) as! u8;
            blurred[targetIndex + 2] = (blue / count) as! u8;
            blurred[targetIndex + 3] = (alpha / count) as! u8;
        }
    }
    let milliseconds = (clock_ns() - started) / 100000;
    print("Goose: blurred ", width, "x", height, " in ", milliseconds / 10, ".", milliseconds % 10, " ms");
    var output: u8[>..] = [];
    output.append("P6\n");
    output.append(file[3..rgb]);
    for pixel in width * height {
        for channel in 3 { output.push(blurred[pixel * 4 + channel]); }
    }
    guard write_file("Landscape.goose.ppm", output) else { exit(1); }
}
