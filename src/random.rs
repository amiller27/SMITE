use crate::config::DEBUG_RANDOM;
use std::convert::TryInto;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
extern crate core;
extern crate rand;
extern crate rand_core;

macro_rules! debug {
    ($($x: expr),*) => {
        if DEBUG_RANDOM {
            println!($($x,)*);
        }
    };
}

pub trait RangeRng {
    fn gen_range(&mut self, r: core::ops::Range<usize>) -> usize;
}

pub struct MockRng {
    results: Vec<u64>,
    i: usize,
}

impl MockRng {
    pub fn from_trace<P>(filename: P) -> Result<MockRng, Box<dyn std::error::Error>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(filename)?;
        let lines = io::BufReader::new(file).lines();

        Ok(MockRng {
            results: lines
                .filter_map(|maybe_line| maybe_line.ok()?.parse().ok())
                .collect(),
            i: 0,
        })
    }

    fn next_u64(&mut self) -> u64 {
        let curr_i = self.i;
        self.i += 1;
        debug!("next_u64: {}", self.results[curr_i]);
        self.results[curr_i]
    }
}

impl RangeRng for MockRng {
    fn gen_range(&mut self, r: core::ops::Range<usize>) -> usize {
        if r.start != 0 {
            panic!("Not implemented");
        }

        let maybe: usize = self.next_u64().try_into().unwrap();
        maybe % r.end
    }
}

impl<T: rand::Rng> RangeRng for T {
    fn gen_range(&mut self, r: core::ops::Range<usize>) -> usize {
        // Hmm
        self.gen_range(r)
    }
}

pub enum Mode {
    Identity,
    #[allow(dead_code)]
    Uninitialized,
}

pub fn permutation<RNG>(n: usize, n_shuffles: usize, mode: Mode, rng: &mut RNG) -> Vec<usize>
where
    RNG: RangeRng,
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
            let v = rng.gen_range(0..(n - 3));
            let u = rng.gen_range(0..(n - 3));
            p.swap(v + 0, u + 2);
            p.swap(v + 1, u + 3);
            p.swap(v + 2, u + 0);
            p.swap(v + 3, u + 1);
        }
    }

    p
}
