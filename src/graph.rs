use crate::config::{Index, DEBUG_COMPRESS};

macro_rules! debug {
    ($($x: expr),*) => {
        if DEBUG_COMPRESS {
            println!($($x,)*);
        }
    };
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub x_adjacency: Vec<usize>,
    pub adjacency_lists: Vec<usize>,
}

impl Graph {
    //fn init(n_vertices: usize, n_edges: usize) -> Graph {
    //    Graph {
    //        x_adjacency: vec![0; n_vertices + 1],
    //        adjacency_lists: vec![0; n_edges],
    //    }
    //}

    pub fn degree(&self, vertex: usize) -> usize {
        self.x_adjacency[vertex + 1] - self.x_adjacency[vertex]
    }

    pub fn neighbors(&self, vertex: usize) -> &[usize] {
        &self.adjacency_lists[self.x_adjacency[vertex]..self.x_adjacency[vertex + 1]]
    }

    pub fn n_vertices(&self) -> usize {
        self.x_adjacency.len() - 1
    }

    pub fn n_edges(&self) -> usize {
        self.adjacency_lists.len()
    }
}

#[derive(Clone, Debug)]
pub struct WeightedGraph {
    pub graph: Graph,
    pub vertex_weights: Vec<Index>,
    pub edge_weights: Vec<Index>,
}

impl WeightedGraph {
    #[allow(dead_code)]
    pub fn from_unweighted(graph: Graph) -> WeightedGraph {
        let n_vertices = graph.n_vertices();
        WeightedGraph {
            graph: graph,
            vertex_weights: vec![1; n_vertices],
            edge_weights: vec![],
        }
    }

    pub fn weighted_neighbors(
        &self,
        vertex: usize,
    ) -> std::iter::Zip<std::slice::Iter<usize>, std::slice::Iter<i32>> {
        let start = self.graph.x_adjacency[vertex];
        let end = self.graph.x_adjacency[vertex + 1];
        self.graph.adjacency_lists[start..end]
            .iter()
            .zip(self.edge_weights[start..end].iter())
    }
}

#[derive(Debug)]
pub struct MutableGraph {
    pub adjacency: Vec<Vec<usize>>,
}

impl MutableGraph {
    pub fn from_graph(graph: Graph) -> MutableGraph {
        MutableGraph {
            adjacency: (0..graph.n_vertices())
                .map(|v| graph.neighbors(v).to_vec())
                .collect(),
        }
    }

    pub fn degree(&self, vertex: usize) -> usize {
        self.adjacency[vertex].len()
    }

    pub fn neighbors(&self, vertex: usize) -> &[usize] {
        &self.adjacency[vertex]
    }

    pub fn neighbors_copy(&self, vertex: usize) -> Vec<usize> {
        self.adjacency[vertex].clone()
    }

    pub fn n_vertices(&self) -> usize {
        self.adjacency.len()
    }

    // pub fn add_neighbor(&mut self, vertex: usize, neighbor: usize) {
    //     self.adjacency[vertex].push(neighbor);
    // }
}

#[derive(Debug, Clone)]
pub struct KeyAndValue {
    pub key: usize,
    pub value: usize,
}

impl Ord for KeyAndValue {
    fn cmp(&self, other: &KeyAndValue) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for KeyAndValue {
    fn partial_cmp(&self, other: &KeyAndValue) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for KeyAndValue {
    fn eq(&self, other: &KeyAndValue) -> bool {
        self.key == other.key
    }
}

impl Eq for KeyAndValue {}

pub fn sort(mut v: Vec<KeyAndValue>) -> Vec<KeyAndValue> {
    // v.sort_unstable();
    crate::qsort::quicksort(&mut v);
    v
}

pub fn compress_graph(
    graph: &Graph,
    vertex_weights: &Vec<Index>,
) -> Option<(WeightedGraph, Vec<usize>, Vec<usize>, Vec<usize>)> {
    debug!("CALLED compress_graph");

    // Each key is the sum of the vertex indices that the vertex is adjacent to
    let keys_unsorted = (0..graph.n_vertices())
        .map(|vertex| KeyAndValue {
            key: graph.neighbors(vertex).iter().sum::<usize>() + vertex,
            value: vertex,
        })
        .collect::<Vec<KeyAndValue>>();

    debug!("Keys unsorted: {:?}", keys_unsorted);
    let keys = sort(keys_unsorted);
    debug!("Keys: {:?}", keys);

    let mut mark = vec![Option::<usize>::None; graph.n_vertices()];

    let mut uncompressed_to_compressed = vec![Option::<usize>::None; graph.n_vertices()];
    let mut compressed_to_uncompressed_ptr = vec![0; graph.n_vertices() + 1];
    let mut compressed_to_uncompressed = vec![0; graph.n_vertices()];

    let mut compressed_vertex = 0;
    compressed_to_uncompressed_ptr[0] = 0;

    let mut compressed_n_vertices = 0;

    for i in 0..graph.n_vertices() {
        let vertex = keys[i].value;

        if uncompressed_to_compressed[vertex].is_some() {
            continue;
        }

        debug!("Looking at vertex {}", vertex);

        mark[vertex] = Some(i);

        for &neighbor in graph.neighbors(vertex) {
            mark[neighbor] = Some(i);
        }

        uncompressed_to_compressed[vertex] = Some(compressed_n_vertices);
        compressed_to_uncompressed[compressed_vertex] = vertex;
        compressed_vertex += 1;

        for j in i + 1..graph.n_vertices() {
            let vertex2 = keys[j].value as usize;
            debug!("Looking at vertex2 {}", vertex2);

            // Break if keys or degrees are different
            if keys[i].key != keys[j].key || graph.degree(vertex) != graph.degree(vertex2) {
                break;
            }

            debug!("SELECTED");

            if uncompressed_to_compressed[vertex2].is_some() {
                continue;
            }

            debug!("SELECTED2");

            let mut found_mismatch = false;
            for &neighbor in graph.neighbors(vertex2) {
                if mark[neighbor] != Some(i) {
                    found_mismatch = true;
                    break;
                }
            }

            if !found_mismatch {
                debug!("SELECTED3");
                uncompressed_to_compressed[vertex2] = Some(compressed_n_vertices);
                compressed_to_uncompressed[compressed_vertex] = vertex2;
                compressed_vertex += 1;
            }
        }

        compressed_n_vertices += 1;
        compressed_to_uncompressed_ptr[compressed_n_vertices] = compressed_vertex;
    }

    const COMPRESSION_FRACTION: f64 = 0.85;
    if (compressed_n_vertices as f64) >= COMPRESSION_FRACTION * (graph.n_vertices() as f64) {
        debug!("EXITED compress_graph");
        return None;
    }

    debug!("COMPRESSING");

    let compressed_n_edges_bound: usize = (0..compressed_n_vertices)
        .map(|i| graph.degree(compressed_to_uncompressed[compressed_to_uncompressed_ptr[i]]))
        .sum();

    debug!("compressed_n_edges_bound: {}", compressed_n_edges_bound);

    mark = vec![None; graph.n_vertices()];
    let mut compressed_vertex_weights = vec![0; compressed_n_vertices];
    let mut compressed_x_adjacency = vec![0; compressed_n_vertices + 1];
    let mut compressed_adjacency = vec![0; compressed_n_edges_bound];
    let mut compressed_n_edges = 0;
    for i in 0..compressed_n_vertices {
        mark[i] = Some(i);
        for j in compressed_to_uncompressed_ptr[i]..compressed_to_uncompressed_ptr[i + 1] {
            let vertex = compressed_to_uncompressed[j];

            compressed_vertex_weights[i] += vertex_weights[i];

            for &neighbor in graph.neighbors(vertex) {
                let compressed_vertex = uncompressed_to_compressed[neighbor].unwrap();
                if mark[compressed_vertex] != Some(i) {
                    mark[compressed_vertex] = Some(i);
                    compressed_adjacency[compressed_n_edges] = compressed_vertex;
                    compressed_n_edges += 1;
                }
            }
        }
        compressed_x_adjacency[i + 1] = compressed_n_edges;
    }

    debug!("EXITED compress_graph");

    compressed_adjacency.truncate(compressed_n_edges);

    return Some((
        WeightedGraph {
            graph: Graph {
                x_adjacency: compressed_x_adjacency,
                adjacency_lists: compressed_adjacency,
            },
            vertex_weights: compressed_vertex_weights,
            edge_weights: vec![1; compressed_n_edges],
        },
        compressed_to_uncompressed_ptr,
        compressed_to_uncompressed,
        (0..graph.n_vertices()).collect(),
    ));
}
