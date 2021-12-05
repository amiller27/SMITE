use crate::config::{Index, Real};

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
    pub vertex_weights: Option<Vec<Index>>,
    pub edge_weights: Option<Vec<Index>>,
}

impl WeightedGraph {
    pub fn from_unweighted(graph: Graph) -> WeightedGraph {
        let n_vertices = graph.n_vertices();
        WeightedGraph {
            graph: graph,
            vertex_weights: Some(vec![1; n_vertices]),
            edge_weights: None,
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
            .zip(self.edge_weights.as_ref().unwrap()[start..end].iter())
    }
}

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

    pub fn add_neighbor(&mut self, vertex: usize, neighbor: usize) {
        self.adjacency[vertex].push(neighbor);
    }
}

#[derive(PartialOrd, Eq, Debug, Clone)]
pub struct KeyAndValue {
    pub key: usize,
    pub value: usize,
}

impl Ord for KeyAndValue {
    fn cmp(&self, other: &KeyAndValue) -> std::cmp::Ordering {
        return self.key.cmp(&other.key);
    }
}

impl PartialEq for KeyAndValue {
    fn eq(&self, other: &KeyAndValue)  -> bool {
        return self.key == other.key;
    }
}

pub fn sort(mut v: Vec<KeyAndValue>) -> Vec<KeyAndValue> {
    v.sort_unstable();
    v
}

pub fn compress_graph(
    graph: &Graph,
    vertex_weights: &Option<Vec<Index>>,
) -> Option<(WeightedGraph, Vec<usize>, Vec<usize>, Vec<usize>)> {
    // Each key is the sum of the vertex indices that the vertex is adjacent to
    let keys_unsorted = (0..graph.n_vertices())
        .map(|vertex| KeyAndValue {
            key: graph.neighbors(vertex).iter().sum::<usize>() + vertex,
            value: vertex,
        })
        .collect::<Vec<KeyAndValue>>();

    println!("Keys unsorted: {:?}", keys_unsorted);
    let keys = sort(keys_unsorted);
    println!("Keys: {:?}", keys);

    let mut mark = vec![Option::<usize>::None; graph.n_vertices()];

    let mut uncompressed_to_compressed = vec![Option::<usize>::None; graph.n_vertices()];
    let mut compressed_to_uncompressed_ptr = vec![0; graph.n_vertices() + 1];
    let mut compressed_to_uncompressed = vec![0; graph.n_vertices()];

    let mut compressed_vertex = 0;
    compressed_to_uncompressed_ptr[0] = 0;

    let mut compressed_n_vertices = 0;

    for i in 0..graph.n_vertices() {
        let vertex = keys[i].value as usize;

        if uncompressed_to_compressed[vertex].is_some() {
            continue;
        }

        mark[vertex] = Some(i);

        for &neighbor in graph.neighbors(vertex) {
            mark[neighbor] = Some(vertex);
        }

        uncompressed_to_compressed[vertex] = Some(compressed_n_vertices);
        compressed_to_uncompressed[compressed_vertex] = vertex;
        compressed_vertex += 1;

        for j in i + 1..graph.n_vertices() {
            let vertex2 = keys[j].value as usize;

            // Break if keys or degrees are different
            if keys[i].key != keys[j].key || graph.degree(vertex) != graph.degree(vertex2) {
                break;
            }

            if uncompressed_to_compressed[vertex2].is_some() {
                continue;
            }

            let mut found_mismatch = false;
            for &neighbor in graph.neighbors(vertex2) {
                if mark[neighbor].is_none() || mark[neighbor].unwrap() != i {
                    found_mismatch = true;
                    break;
                }
            }

            if !found_mismatch {
                uncompressed_to_compressed[vertex2] = Some(compressed_n_vertices);
                compressed_to_uncompressed[compressed_vertex] = vertex2;
            }
        }

        compressed_n_vertices += 1;
        compressed_to_uncompressed_ptr[compressed_n_vertices] = compressed_vertex;
    }

    const COMPRESSION_FRACTION: Real = 0.85;
    if (compressed_n_vertices as Real) >= COMPRESSION_FRACTION * (graph.n_vertices() as Real) {
        return None;
    }

    let compressed_n_edges: usize = (0..compressed_n_vertices)
        .map(|i| graph.degree(compressed_to_uncompressed[compressed_to_uncompressed_ptr[i]]))
        .sum();

    mark = vec![None; graph.n_vertices()];
    let mut compressed_vertex_weights = vec![0; compressed_n_vertices];
    let mut compressed_x_adjacency = vec![0; compressed_n_vertices];
    let mut compressed_adjacency = vec![0; compressed_n_edges];
    let mut l = 0;
    for i in 0..compressed_n_vertices {
        mark[i] = Some(i);
        for j in compressed_to_uncompressed_ptr[i]..compressed_to_uncompressed_ptr[i + 1] {
            let vertex = compressed_to_uncompressed[j];

            compressed_vertex_weights[i] += match vertex_weights {
                None => 1,
                Some(weights) => weights[i],
            };

            for neighbor in graph.neighbors(vertex) {
                let compressed_vertex = uncompressed_to_compressed[*neighbor].unwrap();
                if mark[compressed_vertex].is_none() || mark[compressed_vertex].unwrap() != i {
                    mark[compressed_vertex] = Some(i);
                    compressed_adjacency[l] = compressed_vertex;
                    l += 1;
                }
            }
        }
        compressed_x_adjacency[i + 1] = l;
    }

    let n_edges = compressed_adjacency.len();
    return Some((
        WeightedGraph {
            graph: Graph {
                x_adjacency: compressed_x_adjacency,
                adjacency_lists: compressed_adjacency,
            },
            vertex_weights: Some(compressed_vertex_weights),
            edge_weights: Some(vec![1; n_edges]),
        },
        compressed_to_uncompressed_ptr,
        compressed_to_uncompressed,
        (0..graph.n_vertices()).collect(),
    ));
}
