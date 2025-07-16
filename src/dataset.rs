use core::f64;
use std::iter::repeat_n;

use rand::random_range;

pub fn sine(
    samples: usize,
) -> (
    impl Iterator<Item = [f64; 1]>,
    impl Iterator<Item = [f64; 1]>,
) {
    let x = (0..samples).map(|sample| [1.0 / sample as f64]);
    let y = (0..samples).map(|sample| [2.0 * f64::consts::PI * sample as f64]);
    (x, y)
}

pub fn spiral(
    samples: usize,
    classes: usize,
) -> (
    impl Iterator<Item = [f64; 2]>,
    impl Iterator<Item = [f64; 1]>,
) {
    let x = (0..classes).flat_map(move |class| {
        (0..samples).map(move |sample| {
            let r = 1.0 / samples as f64 * sample as f64;
            let t = class as f64 * 4.0 + random_range(-0.2..0.2);
            [r * (t * 2.5).sin(), r * (t * 2.5).cos()]
        })
    });
    let y = (0..classes).flat_map(move |class| repeat_n([class as f64], samples));
    (x, y)
}

pub fn vertical(
    samples: usize,
    classes: usize,
) -> (
    impl Iterator<Item = [f64; 2]>,
    impl Iterator<Item = [f64; 1]>,
) {
    let x = (0..classes).flat_map(move |class| {
        (0..samples).map(move |_| {
            [
                random_range(-0.1..0.1) + class as f64 / 3.0,
                random_range(-0.1..0.1) + 0.5,
            ]
        })
    });
    let y = (0..classes).flat_map(move |class| repeat_n([class as f64], samples));
    (x, y)
}
