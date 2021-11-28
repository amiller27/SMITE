use crate::graph::{Graph, WeightedGraph};
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

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
        .take(3)
        .map(|s| s.parse().ok().unwrap())
        .collect::<Vec<usize>>();
    let [rows, cols, _nonzeros] = <[usize; 3]>::try_from(v).ok().unwrap();

    if rows != cols {
        panic!("Matrix isn't square");
    }

    let mut graph = WeightedGraph {
        graph: Graph {
            x_adjacency: vec![0],
            adjacency_lists: vec![],
        },
        edge_weights: None,
        vertex_weights: None,
    };

    let mut transpose_graph = vec![vec![]; rows];

    let mut last_col = lines.enumerate().fold(0, |mut last_col, (_i, maybe_line)| {
        if let Ok(line) = maybe_line {
            let numbers = line
                .split_whitespace()
                .map(|s| s.parse().ok().unwrap())
                .collect::<Vec<f64>>();
            let [row_plus_one, col_plus_one, _weight] = <[f64; 3]>::try_from(numbers).ok().unwrap();
            let row = row_plus_one as usize - 1;
            let col = col_plus_one as usize - 1;

            if row == col {
                return last_col;
            }

            while last_col != col {
                graph
                    .graph
                    .adjacency_lists
                    .extend(&transpose_graph[last_col]);
                graph
                    .graph
                    .x_adjacency
                    .push(graph.graph.adjacency_lists.len());
                last_col += 1;
            }

            graph.graph.adjacency_lists.push(row);
            transpose_graph[row].push(col);
        }

        last_col
    });

    while last_col != cols {
        graph
            .graph
            .adjacency_lists
            .extend(&transpose_graph[last_col]);
        graph
            .graph
            .x_adjacency
            .push(graph.graph.adjacency_lists.len());
        last_col += 1;
    }

    Ok(graph)
}
