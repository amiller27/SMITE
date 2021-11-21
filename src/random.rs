#![feature(rand)]

extern crate rand;

pub enum Mode {
    Identity,
    Uninitialized,
}

pub fn permutation<RNG>(n: usize, n_shuffles: usize, mode: Mode, rng: &mut RNG) -> Vec<usize>
where
    RNG: rand::Rng,
{
    let mut p = vec![0; n];

    if let Identity = mode {
        for i in 0..n {
            p[i] = i;
        }
    }

    if n < 10 {
        for i in 0..n {
            let v = rng.gen_range(0..n);
            let u = rng.gen_range(0..n);
            p.swap(v, u);
        }
    } else {
        for i in 0..n_shuffles {
            let v = rng.gen_range(0..n - 3);
            let u = rng.gen_range(0..n - 3);
            p.swap(v + 0, u + 2);
            p.swap(v + 1, u + 3);
            p.swap(v + 2, u + 0);
            p.swap(v + 3, u + 1);
        }
    }

    p
}
