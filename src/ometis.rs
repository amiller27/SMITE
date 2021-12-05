use crate::config::{Config, Index};
use crate::graph::{compress_graph, Graph, WeightedGraph};
use crate::random::RangeRng;
use std::error::Error;
use std::fmt;

pub struct NodeNDResult {
    pub permutation: Vec<usize>,
    pub inverse_permutation: Vec<usize>,
}

#[derive(Debug)]
pub enum MetisError {}

impl Error for MetisError {}

impl fmt::Display for MetisError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Oh no, something bad went down")
    }
}

enum GraphData {
    Uncompressed,
    Compressed(Vec<usize>, Vec<usize>),
}

pub fn node_nd<RNG>(
    graph: Graph,
    vertex_weights: Option<Vec<Index>>,
    rng: &mut RNG,
) -> Result<NodeNDResult, MetisError>
where
    RNG: RangeRng,
{
    let config = crate::config::default_config();

    /* prune the dense columns */
    if config.pruning_factor > 0.0 {
        // Do stuff
        panic!("Not implemented");
    }

    let maybe_compressed_graph = match config.compress_before_ordering {
        true => compress_graph(&graph, &vertex_weights),
        false => None,
    };

    let n_vertices = graph.n_vertices();
    let (compressed_graph, info, labels) = match maybe_compressed_graph {
        Some((
            compressed_graph,
            compressed_to_uncompressed_ptr,
            compressed_to_uncompressed,
            labels,
        )) => (
            compressed_graph,
            GraphData::Compressed(compressed_to_uncompressed_ptr, compressed_to_uncompressed),
            labels,
        ),
        None => (
            match vertex_weights {
                Some(weights) => {
                    let n_edges = graph.n_edges();
                    WeightedGraph {
                        graph: graph,
                        vertex_weights: Some(weights),
                        edge_weights: Some(vec![1; n_edges]),
                    }
                }
                None => WeightedGraph::from_unweighted(graph),
            },
            GraphData::Uncompressed,
            (0..n_vertices).collect(),
        ),
    };

    println!(
        "Compressed: {}, {}: {:?}",
        compressed_graph.graph.n_vertices(),
        compressed_graph.graph.n_edges(),
        compressed_graph.graph.adjacency_lists
    );

    let compressed_n_vertices = compressed_graph.graph.n_vertices();
    let mut inverse_permutation = vec![0; compressed_graph.graph.n_vertices()];
    if config.connected_components_order {
        m_level_nested_dissection_connected_components(
            &config,
            compressed_graph,
            labels,
            0,
            &mut inverse_permutation,
        )
    } else {
        m_level_nested_dissection(
            &config,
            compressed_graph,
            labels,
            0,
            &mut inverse_permutation,
            rng,
        )
    }

    let mut permutation = vec![0; n_vertices];
    if let GraphData::Compressed(compressed_to_uncompressed_ptr, compressed_to_uncompressed) = info
    {
        // uncompress the ordering
        // construct perm from iperm
        for i in 0..compressed_n_vertices {
            permutation[inverse_permutation[i]] = i;
        }

        let mut l = 0;
        for ii in 0..compressed_n_vertices {
            let i = permutation[ii];
            for j in compressed_to_uncompressed_ptr[i]..compressed_to_uncompressed_ptr[i + 1] {
                inverse_permutation[compressed_to_uncompressed[j]] = l;
                l += 1;
            }
        }
    }

    for i in 0..n_vertices {
        permutation[inverse_permutation[i]] = i;
    }

    Ok(NodeNDResult {
        permutation: permutation,
        inverse_permutation: inverse_permutation,
    })
}

fn m_level_nested_dissection_connected_components(
    _config: &Config,
    _graph: WeightedGraph,
    _labels: Vec<usize>,
    _first_vertex: usize,
    _order: &mut Vec<usize>,
) {
    panic!("Not implemented");
}

fn m_level_nested_dissection<RNG>(
    config: &Config,
    graph: WeightedGraph,
    labels: Vec<usize>,
    first_vertex: usize,
    order: &mut Vec<usize>,
    rng: &mut RNG,
) where
    RNG: RangeRng,
{
    let n_vertices = graph.graph.n_vertices();
    let boundarized_pyramid = m_level_node_bisection_multiple(config, graph, rng);
    let curr_level = boundarized_pyramid.first().unwrap();

    for (i, &vertex) in curr_level.boundary_info.boundary_ind.iter().enumerate() {
        order[labels[vertex]] = first_vertex + n_vertices - i;
    }

    let (left_graph, left_labels, right_graph, right_labels) = split_graph_order(
        config,
        &curr_level.graph,
        &curr_level.boundary_info,
        &labels,
    );

    const MMD_SWITCH: usize = 120;
    let left_n_vertices = left_graph.graph.n_vertices();
    if left_graph.graph.n_vertices() > MMD_SWITCH && left_graph.graph.n_edges() > 0 {
        m_level_nested_dissection(config, left_graph, left_labels, first_vertex, order, rng)
    } else {
        mmd_order(config, left_graph.graph, &left_labels, first_vertex, order);
    }

    if right_graph.graph.n_vertices() > MMD_SWITCH && right_graph.graph.n_edges() > 0 {
        m_level_nested_dissection(
            config,
            right_graph,
            right_labels,
            first_vertex + left_n_vertices,
            order,
            rng,
        )
    } else {
        mmd_order(
            config,
            right_graph.graph,
            &right_labels,
            first_vertex + left_n_vertices,
            order,
        )
    }
}

fn m_level_node_bisection_multiple<RNG>(
    config: &Config,
    graph: WeightedGraph,
    rng: &mut RNG,
) -> Vec<crate::separator_refinement::BoundarizedGraphPyramidLevel>
where
    RNG: RangeRng,
{
    if config.n_separators == 1
        || graph.graph.n_vertices() < config.single_separator_threshold_node_bisection_multiple()
    {
        return m_level_node_bisection_l2(config, graph, rng);
    } else {
        panic!();
    }
}

fn m_level_node_bisection_l2<RNG>(
    config: &Config,
    graph: WeightedGraph,
    rng: &mut RNG,
) -> Vec<crate::separator_refinement::BoundarizedGraphPyramidLevel>
where
    RNG: RangeRng,
{
    if graph.graph.n_vertices() < config.single_separator_threshold_node_bisection_l2() {
        return m_level_node_bisection_l1(config, graph, rng);
    } else {
        panic!();
    }
}

fn m_level_node_bisection_l1<RNG>(
    config: &Config,
    graph: WeightedGraph,
    rng: &mut RNG,
) -> Vec<crate::separator_refinement::BoundarizedGraphPyramidLevel>
where
    RNG: RangeRng,
{
    let n_vertices = graph.graph.n_vertices();
    let coarsen_to = (graph.graph.n_vertices() / 8).clamp(40, 100);
    let total_weights = graph.vertex_weights.as_ref().unwrap().iter().sum();
    let graph_pyramid =
        crate::coarsen::coarsen_graph(config, graph, coarsen_to, total_weights, rng);
    let n_i_parts = std::cmp::max(
        1,
        if n_vertices <= coarsen_to {
            config.init_n_i_parts() / 2
        } else {
            config.init_n_i_parts()
        },
    );
    let separated_graph_pyramid =
        crate::initialize_partition::initialize_separator(config, graph_pyramid, n_i_parts, rng);
    let which_graph = separated_graph_pyramid.len() - 1;
    let boundarized_pyramid = crate::separator_refinement::refine_two_way_node(
        config,
        separated_graph_pyramid,
        0,
        which_graph,
        rng,
    );

    boundarized_pyramid
}

fn split_graph_order(
    _config: &Config,
    graph: &WeightedGraph,
    boundary_info: &crate::separator_refinement::BoundaryInfo,
    labels: &Vec<usize>,
) -> (WeightedGraph, Vec<usize>, WeightedGraph, Vec<usize>) {
    let mut split_n_vertices = [0, 0, 0];
    let mut split_n_edges = [0, 0, 0];
    let mut rename = vec![0; graph.graph.n_vertices()];
    for vertex in 0..graph.graph.n_vertices() {
        let k = boundary_info._where[vertex];
        rename[vertex] = split_n_vertices[k];
        split_n_vertices[k] += 1;
        split_n_edges[k] += graph.graph.degree(vertex);
    }

    /* Go and use bnd_ptr to also mark the boundary nodes in the two partitions */
    let is_boundary = {
        let mut is_boundary = vec![false; graph.graph.n_vertices()];
        for &vertex in &boundary_info.boundary_ind {
            for &neighbor in graph.graph.neighbors(vertex) {
                is_boundary[neighbor] = true;
            }
        }
        is_boundary
    };

    let mut left_graph = WeightedGraph {
        graph: Graph {
            x_adjacency: vec![],
            adjacency_lists: vec![],
        },
        vertex_weights: Some(vec![]),
        edge_weights: Some(vec![]),
    };
    let mut right_graph = WeightedGraph {
        graph: Graph {
            x_adjacency: vec![],
            adjacency_lists: vec![],
        },
        vertex_weights: Some(vec![]),
        edge_weights: Some(vec![]),
    };
    let mut graphs = [&mut left_graph, &mut right_graph];

    let mut left_labels = vec![];
    let mut right_labels = vec![];
    let split_labels = [&mut left_labels, &mut right_labels];

    for i in 0..graph.graph.n_vertices() {
        let my_part = boundary_info._where[i];
        if my_part == 2 {
            continue;
        }

        let my_graph = &mut graphs[my_part];

        if !is_boundary[i] {
            /* This is an interior vertex */
            for &neighbor in graph.graph.neighbors(i) {
                my_graph.graph.adjacency_lists.push(neighbor);
            }
        } else {
            for &neighbor in graph.graph.neighbors(i).iter() {
                if boundary_info._where[neighbor] == my_part {
                    my_graph.graph.adjacency_lists.push(neighbor);
                }
            }
        }

        my_graph
            .vertex_weights
            .as_mut()
            .unwrap()
            .push(graph.vertex_weights.as_ref().unwrap()[i]);
        split_labels[my_part].push(labels[i]);
        my_graph
            .graph
            .x_adjacency
            .push(my_graph.graph.adjacency_lists.len()); // off by one?
    }

    for my_part in 0..2 {
        graphs[my_part].edge_weights = Some(vec![1; graphs[my_part].graph.n_edges()]);

        for i in 0..graphs[my_part].graph.n_edges() {
            graphs[my_part].graph.adjacency_lists[i] =
                rename[graphs[my_part].graph.adjacency_lists[i]];
        }
    }

    // SetupGraph_tvwgt(lgraph)
    // SetupGraph_tvwgt(rgraph)

    (left_graph, left_labels, right_graph, right_labels)
}

fn mmd_order(
    _config: &Config,
    graph: Graph,
    labels: &Vec<usize>,
    first_vertex: usize,
    order: &mut Vec<usize>,
) {
    let n_vertices = graph.n_vertices();
    let mmd_result = crate::mmd::gen_mmd(graph);

    // This puts iperm back into the _global_ ordering, meaning order should be an in-out...
    for i in 0..n_vertices {
        order[labels[i]] = first_vertex + mmd_result.iperm[i];
    }
}
