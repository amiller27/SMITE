use std::convert::TryInto;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
extern crate core;
extern crate rand;
extern crate rand_core;

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
        println!("next_u64: {}", self.results[curr_i]);
        self.results[curr_i]
    }
}

impl RangeRng for MockRng {
    fn gen_range(&mut self, r: core::ops::Range<usize>) -> usize {
        if r.start != 0 {
            panic!("Not implemented");
        }

        let maybe: usize = self.next_u64().try_into().unwrap();
        // println!(
        //     "GENERATING RANGE start {}, end {}, result {}, final {}",
        //     r.start, r.end, maybe, maybe % r.end
        // );
        maybe % r.end
    }
}

// impl rand::RngCore for MockRng {
//     fn next_u32(&mut self) -> u32 {
//         self.next_u64() as u32
//     }
//
//     fn next_u64(&mut self) -> u64 {
//         let curr_i = self.i;
//         self.i += 1;
//         println!("next_u64: {}", self.results[curr_i]);
//         self.results[curr_i]
//     }
//
//     fn fill_bytes(&mut self, dest: &mut [u8]) {
//         rand_core::impls::fill_bytes_via_next(self, dest)
//     }
//
//     fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
//         Ok(self.fill_bytes(dest))
//     }
// }

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

    // println!("p: {:?}", p);
    // println!("n: {}", n);

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
            // println!("v: {}, u: {}, p: {:?}", v, u, p);
        }
    }

    p
}
