use crate::config::{Config, Index, DEBUG_OMETIS};
use crate::graph::{compress_graph, Graph, WeightedGraph};
use crate::random::RangeRng;
use std::error::Error;
use std::fmt;

macro_rules! debug {
    ($($x: expr),*) => {
        if DEBUG_OMETIS {
            println!($($x,)*);
        }
    };
}

#[derive(Debug)]
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
    vertex_weights: Vec<Index>,
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
    let (config, compressed_graph, info, labels) = match maybe_compressed_graph {
        Some((
            compressed_graph,
            compressed_to_uncompressed_ptr,
            compressed_to_uncompressed,
            labels,
        )) => (
            Config {
                n_separators: (if (graph.n_vertices() as f64
                    / compressed_graph.graph.n_vertices() as f64)
                    > 1.5
                    && config.n_separators == 1
                {
                    2
                } else {
                    config.n_separators
                }),
                ..config
            },
            compressed_graph,
            GraphData::Compressed(compressed_to_uncompressed_ptr, compressed_to_uncompressed),
            labels,
        ),
        None => (
            config,
            match vertex_weights {
                weights => {
                    let n_edges = graph.n_edges();
                    WeightedGraph {
                        graph: graph,
                        vertex_weights: weights,
                        edge_weights: vec![1; n_edges],
                    }
                } //None => WeightedGraph::from_unweighted(graph),
            },
            GraphData::Uncompressed,
            (0..n_vertices).collect(),
        ),
    };

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
            matches!(info, GraphData::Compressed(_, _)),
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

        // Resize to full size from compressed size
        inverse_permutation = vec![0; n_vertices];

        let mut l = 0;
        for ii in 0..compressed_n_vertices {
            let i = permutation[ii];
            debug!("uncompressing i: {}, ii: {}", i, ii);
            for j in compressed_to_uncompressed_ptr[i]..compressed_to_uncompressed_ptr[i + 1] {
                debug!(
                    "uncompressing j: {}, cind: {}",
                    j, compressed_to_uncompressed[j]
                );
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
    graph_is_compressed: bool,
    rng: &mut RNG,
) where
    RNG: RangeRng,
{
    debug!("CALLED m_level_nested_dissection");
    debug!("{:?}", graph);
    let n_vertices = graph.graph.n_vertices();
    let boundarized_pyramid =
        m_level_node_bisection_multiple(config, graph, graph_is_compressed, rng);
    let curr_level = boundarized_pyramid.last().unwrap();

    debug!("nbnd: {}", curr_level.boundary_info.boundary_ind.len());
    // for (i, level) in boundarized_pyramid.iter().enumerate() {
    //     debug!("i: {}, {:?}", i, level.boundary_info.boundary_ind);
    // }
    for (i, &vertex) in curr_level.boundary_info.boundary_ind.iter().enumerate() {
        debug!(
            "v: {}, label: {}, result: {}",
            vertex,
            labels[vertex],
            first_vertex + n_vertices - i - 1
        );
        order[labels[vertex]] = first_vertex + n_vertices - i - 1;
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
        m_level_nested_dissection(
            config,
            left_graph,
            left_labels,
            first_vertex,
            order,
            graph_is_compressed,
            rng,
        )
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
            graph_is_compressed,
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
    graph_is_compressed: bool,
    rng: &mut RNG,
) -> Vec<crate::separator_refinement::BoundarizedGraphPyramidLevel>
where
    RNG: RangeRng,
{
    debug!("CALLED m_level_node_bisection_multiple");

    debug!("nseps: {}", config.n_separators);
    if config.n_separators == 1
        || graph.graph.n_vertices()
            < config.single_separator_threshold_node_bisection_multiple(graph_is_compressed)
    {
        let (result, _min_cut) = m_level_node_bisection_l2(config, graph, graph_is_compressed, rng);
        debug!("EARLY EXITED m_level_node_bisection_multiple");
        return result;
    }

    // The METIS implementation of this does a little less allocation and
    // copying, but I'm pretty sure the intent is just to do this, i.e. compute
    // this multiple times and pick the best based on min_cut

    let mut best_result = None;
    let mut best_min_cut = 0;
    for i in 0..config.n_separators {
        let (result, min_cut) =
            m_level_node_bisection_l2(config, graph.clone(), graph_is_compressed, rng);

        debug!("min_cut: {}", min_cut);
        if i == 0
            || min_cut < best_min_cut
            // doesn't change the min_cut of the best result, just for equivalence with METIS
            || (i == config.n_separators - 1 && min_cut == best_min_cut)
        {
            best_result = Some(result);
            best_min_cut = min_cut;
        }

        if min_cut == 0 {
            break;
        }
    }

    debug!("best_min_cut: {}", best_min_cut);

    debug!("EXITED m_level_node_bisection_multiple");

    best_result.unwrap()
}

fn m_level_node_bisection_l2<RNG>(
    config: &Config,
    graph: WeightedGraph,
    graph_is_compressed: bool,
    rng: &mut RNG,
) -> (
    Vec<crate::separator_refinement::BoundarizedGraphPyramidLevel>,
    Index,
)
where
    RNG: RangeRng,
{
    debug!("CALLED m_level_node_bisection_l2");

    if graph.graph.n_vertices() < config.single_separator_threshold_node_bisection_l2() {
        let result = m_level_node_bisection_l1(
            config,
            graph,
            graph_is_compressed,
            config.init_n_i_parts(),
            rng,
        );
        debug!("EARLY EXITED m_level_node_bisection_l2");
        return result;
    }

    panic!();
    // let coarsen_to = std::cmp::max(100, graph.graph.n_vertices() / 30);
    // let graph_pyramid = crate::coarsen::coarsen_graph_n_levels(config, graph, coarsen_to, 4);

    // let mut best_where;
    // let min_cut = graph.vertex_weights.unwrap().iter().sum();
    // const N_RUNS: usize = 5;
    // for i in 0..N_RUNS {
    //     // This is most definitely wrong
    //     let (separated_pyramid, new_min_cut, _where) = m_level_node_bisection_l1(
    //         config,
    //         graph_pyramid.last().unwrap(),
    //         graph_is_compressed,
    //         (0.7 * config.init_n_i_parts() as f64) as usize,
    //         rng,
    //     );

    //     if i == 0 || new_min_cut < min_cut {
    //         min_cut = new_min_cut;
    //         if i < N_RUNS - 1 {
    //             best_where = _where.clone();
    //         }
    //     }

    //     if min_cut == 0 {
    //         break;
    //     }
    // }

    // let _where = best_where;

    // let which_graph = separated_graph_pyramid.len();
    // crate::separator_refinement::refine_two_way_node(
    //     config,
    //     separated_graph_pyramid,
    //     0,
    //     which_graph,
    //     graph_is_compressed,
    //     rng,
    // )

    // debug!("EXITED m_level_node_bisection_l2");
}

fn m_level_node_bisection_l1<RNG>(
    config: &Config,
    graph: WeightedGraph,
    graph_is_compressed: bool,
    n_i_parts: usize,
    rng: &mut RNG,
) -> (
    Vec<crate::separator_refinement::BoundarizedGraphPyramidLevel>,
    Index,
)
where
    RNG: RangeRng,
{
    debug!("CALLED m_level_node_bisection_l1");
    debug!("{:?}", graph);

    let coarsen_to = (graph.graph.n_vertices() / 8).clamp(40, 100);
    let total_weights = graph.vertex_weights.iter().sum();
    let graph_pyramid =
        crate::coarsen::coarsen_graph(config, graph, coarsen_to, total_weights, rng);
    debug!("pyramid: {:?}", graph_pyramid);

    let n_i_parts = std::cmp::max(
        1,
        if graph_pyramid.last().unwrap().graph.graph.n_vertices() <= coarsen_to {
            n_i_parts / 2
        } else {
            n_i_parts
        },
    );

    debug!("n_i_parts: {}", n_i_parts);

    let separated_graph_pyramid = crate::initialize_partition::initialize_separator(
        config,
        graph_pyramid,
        n_i_parts,
        graph_is_compressed,
        rng,
    );

    debug!("separated pyramid: {:?}", separated_graph_pyramid);
    let which_graph = separated_graph_pyramid.len() - 1;
    let (boundarized_pyramid, min_cut) = crate::separator_refinement::refine_two_way_node(
        config,
        separated_graph_pyramid,
        0,
        which_graph,
        graph_is_compressed,
        rng,
    );

    debug!("boundarized pyramid: {:?}", boundarized_pyramid);

    debug!("EXITED m_level_node_bisection_l1");

    (boundarized_pyramid, min_cut)
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
            x_adjacency: vec![0],
            adjacency_lists: vec![],
        },
        vertex_weights: vec![],
        edge_weights: vec![],
    };
    let mut right_graph = WeightedGraph {
        graph: Graph {
            x_adjacency: vec![0],
            adjacency_lists: vec![],
        },
        vertex_weights: vec![],
        edge_weights: vec![],
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

        my_graph.vertex_weights.push(graph.vertex_weights[i]);
        split_labels[my_part].push(labels[i]);
        my_graph
            .graph
            .x_adjacency
            .push(my_graph.graph.adjacency_lists.len()); // off by one?
    }

    for my_part in 0..2 {
        graphs[my_part].edge_weights = vec![1; graphs[my_part].graph.n_edges()];

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
