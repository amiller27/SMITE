extern crate rand;

pub enum Mode {
    Identity,
    #[allow(dead_code)]
    Uninitialized,
}

pub fn permutation<RNG>(n: usize, n_shuffles: usize, mode: Mode, rng: &mut RNG) -> Vec<usize>
where
    RNG: rand::Rng,
{
    let mut p = vec![0; n];

    if let Mode::Identity = mode {
        for i in 0..n {
            p[i] = i;
        }
    }

    if n < 10 {
        for _i in 0..n {
            let v = rng.gen_range(0..n);
            let u = rng.gen_range(0..n);
            p.swap(v, u);
        }
    } else {
        for _i in 0..n_shuffles {
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
