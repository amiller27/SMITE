use crate::graph::{Graph, WeightedGraph};
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

#[allow(dead_code)]
pub fn read_graph<P>(filename: P) -> Result<WeightedGraph, Box<dyn std::error::Error>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let mut lines = io::BufReader::new(file)
        .lines()
        .filter(|s| !s.as_ref().unwrap().starts_with("%"));

    let line = match lines.next() {
        Some(Ok(line)) => line,
        _ => panic!(),
    };

    let v = line
        .split_whitespace()
        .take(4)
        .map(|s| s.parse().ok().unwrap())
        .collect::<Vec<usize>>();
    let [_n_vertices, _n_edges, _ignored, n_constraints] = <[usize; 4]>::try_from(v).ok().unwrap();

    let mut graph = WeightedGraph {
        graph: Graph {
            x_adjacency: vec![],
            adjacency_lists: vec![],
        },
        edge_weights: None,
        vertex_weights: Some(vec![]),
    };

    lines.enumerate().for_each(|(_i, maybe_line)| {
        if let Ok(line) = maybe_line {
            let numbers = line
                .split_whitespace()
                .map(|s| s.parse().ok().unwrap())
                .collect::<Vec<usize>>();

            graph
                .graph
                .x_adjacency
                .push(graph.graph.adjacency_lists.len());
            for weight in numbers[..n_constraints as usize].iter() {
                graph.vertex_weights.as_mut().unwrap().push(*weight as i32);
            }
            graph
                .graph
                .adjacency_lists
                .extend_from_slice(&numbers[n_constraints as usize..]);
        }
    });

    Ok(graph)
}
