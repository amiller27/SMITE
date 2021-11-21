use crate::config::{Config, Index};
use crate::graph::{compress_graph, Graph, WeightedGraph};

struct NodeNDResult {
    permutation: Vec<usize>,
    inverse_permutation: Vec<usize>,
}

enum MetisError {}

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
    RNG: rand::Rng,
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
                Some(weights) => WeightedGraph {
                    graph: graph,
                    vertex_weights: Some(weights),
                    edge_weights: None,
                },
                None => WeightedGraph::from_unweighted(graph),
            },
            GraphData::Uncompressed,
            (0..graph.n_vertices()).collect(),
        ),
    };

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

    let mut permutation = vec![0; graph.n_vertices()];
    if let GraphData::Compressed(compressed_to_uncompressed_ptr, compressed_to_uncompressed) = info
    {
        // uncompress the ordering
        // construct perm from iperm
        for i in 0..compressed_graph.graph.n_vertices() {
            permutation[inverse_permutation[i]] = i;
        }

        let mut l = 0;
        for ii in 0..compressed_graph.graph.n_vertices() {
            let i = permutation[ii];
            for j in compressed_to_uncompressed_ptr[i]..compressed_to_uncompressed_ptr[i + 1] {
                inverse_permutation[compressed_to_uncompressed[j]] = l;
                l += 1;
            }
        }
    }

    for i in 0..graph.n_vertices() {
        permutation[inverse_permutation[i]] = i;
    }

    Ok(NodeNDResult {
        permutation: permutation,
        inverse_permutation: inverse_permutation,
    })
}

fn m_level_nested_dissection_connected_components(
    config: &Config,
    graph: WeightedGraph,
    labels: Vec<usize>,
    first_vertex: usize,
    order: &mut Vec<usize>,
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
    RNG: rand::Rng,
{
    let boundarized_pyramid = m_level_node_bisection_multiple(config, graph, rng);
    let coarsest_level = boundarized_pyramid.last().unwrap();

    for (i, &vertex) in coarsest_level.boundary_info.boundary_ind.iter().enumerate() {
        order[labels[vertex]] = first_vertex + graph.graph.n_vertices() - i;
    }

    let (left_graph, left_labels, right_graph, right_labels) =
        split_graph_order(config, graph, coarsest_level.boundary_info, &labels);

    const MMD_SWITCH: usize = 120;
    if left_graph.graph.n_vertices() > MMD_SWITCH && left_graph.graph.n_edges() > 0 {
        m_level_nested_dissection(
            config,
            left_graph,
            left_labels,
            first_vertex,
            &mut order,
            rng,
        )
    } else {
        mmd_order(
            config,
            left_graph.graph,
            &left_labels,
            first_vertex,
            &mut order,
        );
    }

    if right_graph.graph.n_vertices() > MMD_SWITCH && right_graph.graph.n_edges() > 0 {
        m_level_nested_dissection(
            config,
            right_graph,
            right_labels,
            first_vertex + left_graph.graph.n_vertices(),
            &mut order,
            rng,
        )
    } else {
        mmd_order(
            config,
            right_graph.graph,
            &right_labels,
            first_vertex + left_graph.graph.n_vertices(),
            &mut order,
        )
    }
}

fn m_level_node_bisection_multiple<RNG>(
    config: &Config,
    graph: WeightedGraph,
    rng: &mut RNG,
) -> Vec<crate::separator_refinement::BoundarizedGraphPyramidLevel>
where
    RNG: rand::Rng,
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
    RNG: rand::Rng,
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
    RNG: rand::Rng,
{
    let coarsen_to = (graph.graph.n_vertices() / 8).clamp(40, 100);
    let graph_pyramid = crate::coarsen::coarsen_graph(
        config,
        graph,
        coarsen_to,
        graph.vertex_weights.unwrap().iter().sum(),
    );
    let n_i_parts = std::cmp::max(
        1,
        if graph.graph.n_vertices() <= coarsen_to {
            config.init_n_i_parts() / 2
        } else {
            config.init_n_i_parts()
        },
    );
    let separated_graph_pyramid =
        crate::initialize_partition::initialize_separator(config, graph_pyramid, n_i_parts, rng);
    let boundarized_pyramid = crate::separator_refinement::refine_two_way_node(
        config,
        separated_graph_pyramid,
        0,
        separated_graph_pyramid.len() - 1,
    );

    boundarized_pyramid
}

fn split_graph_order(
    config: &Config,
    graph: WeightedGraph,
    boundary_info: crate::separator_refinement::BoundaryInfo,
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
        for vertex in boundary_info.boundary_ind {
            for &neighbor in graph.graph.neighbors(vertex) {
                is_boundary[neighbor] = true;
            }
        }
        is_boundary
    };

    let mut split_vertex_weights = [vec![], vec![]];
    let mut split_labels = [vec![], vec![]];
    let mut split_x_adjacency = [vec![], vec![]];
    let mut split_adjacency: [Vec<usize>; 2] = [vec![], vec![]];

    for i in 0..graph.graph.n_vertices() {
        let my_part = boundary_info._where[i];
        if my_part == 2 {
            continue;
        }

        if !is_boundary[i] {
            /* This is an interior vertex */
            for (i, &neighbor) in graph.graph.neighbors(i).iter().enumerate() {
                split_adjacency[my_part].push(neighbor);
            }
        } else {
            for &neighbor in graph.graph.neighbors(i).iter() {
                if boundary_info._where[neighbor] == my_part {
                    split_adjacency[my_part].push(neighbor);
                }
            }
        }

        split_vertex_weights[my_part].push(graph.vertex_weights.unwrap()[i]);
        split_labels[my_part].push(labels[i]);
        split_x_adjacency[my_part].push(split_adjacency[my_part].len()); // off by one?
    }

    let split_adjacency_weights = [
        vec![1; split_adjacency[0].len()],
        vec![1; split_adjacency[1].len()],
    ];

    for my_part in 0..2 {
        for i in 0..split_adjacency[my_part].len() {
            split_adjacency[my_part][i] = rename[split_adjacency[my_part][i]];
        }
    }

    // SetupGraph_tvwgt(lgraph)
    // SetupGraph_tvwgt(rgraph)

    let left_graph = WeightedGraph {
        graph: Graph {
            x_adjacency: split_x_adjacency[0],
            adjacency_lists: split_adjacency[0],
        },
        vertex_weights: Some(split_vertex_weights[0]),
        edge_weights: Some(split_adjacency_weights[0]),
    };
    let right_graph = WeightedGraph {
        graph: Graph {
            x_adjacency: split_x_adjacency[1],
            adjacency_lists: split_adjacency[1],
        },
        vertex_weights: Some(split_vertex_weights[1]),
        edge_weights: Some(split_adjacency_weights[1]),
    };

    (left_graph, split_labels[0], right_graph, split_labels[1])
}

fn mmd_order(
    config: &Config,
    graph: Graph,
    labels: &Vec<usize>,
    first_vertex: usize,
    order: &mut Vec<usize>,
) {
    let mmd_result = crate::mmd::gen_mmd(graph);

    // This puts iperm back into the _global_ ordering, meaning order should be an in-out...
    for i in 0..graph.n_vertices() {
        order[labels[i]] = first_vertex + mmd_result.iperm[i];
    }
}
