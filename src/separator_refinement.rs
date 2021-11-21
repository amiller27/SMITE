use crate::config::{Config, RefinementType};
use crate::graph::WeightedGraph;

struct NrInfo {
    pub e_degrees: [usize; 2],
}

pub struct BoundaryInfo {
    pub _where: Vec<usize>,
    pub partition_weights: [i32; 3],
    pub boundary_ind: Vec<usize>,
    pub boundary_ptr: Vec<Option<usize>>,
    pub nr_info: Vec<NrInfo>,
}

pub struct GraphPyramidLevel {
    graph: WeightedGraph,
    coarsening_map: Vec<usize>,
    coarser_graph_where: Vec<usize>,
    total_vertex_weights: i32,
}

pub struct BoundarizedGraphPyramidLevel {
    pub graph: WeightedGraph,
    // Do we need this whole thing?
    pub boundary_info: BoundaryInfo,
}

pub fn refine_two_way_node(
    config: &Config,
    graph_pyramid: Vec<GraphPyramidLevel>,
    org_graph: usize,
    graph: usize,
) -> Vec<BoundarizedGraphPyramidLevel> {
    let mut boundarized_pyramid = Vec::new();

    if graph == org_graph {
        panic!();
    } else {
        loop {
            graph -= 1;

            let boundary_info = project_two_way_node_partition(
                config,
                graph_pyramid[graph].graph,
                graph_pyramid[graph].coarsening_map,
                graph_pyramid[graph].coarser_graph_where,
            );

            // delete graph->coarser

            let boundary_info = crate::fm_separator_refinement::two_way_node_balance(
                config,
                graph_pyramid[graph].graph,
                boundary_info,
                graph_pyramid[graph].total_vertex_weights,
            );

            let (min_cut, boundary_info) = match config.refinement_type {
                RefinementType::SEP1SIDED => {
                    crate::fm_separator_refinement::two_way_node_refine_one_sided(
                        config,
                        graph_pyramid[graph].graph,
                        boundary_info,
                        config.n_iterations,
                    )
                }
                RefinementType::SEP2SIDED => {
                    crate::fm_separator_refinement::two_way_node_refine_two_sided(
                        config,
                        graph_pyramid[graph].graph,
                        boundary_info,
                        config.n_iterations,
                    )
                }
                _ => panic!("What's this"),
            };

            boundarized_pyramid.push(BoundarizedGraphPyramidLevel {
                graph: graph_pyramid[graph].graph,
                boundary_info: boundary_info,
            });

            if graph == org_graph {
                break;
            }
        }
    }

    boundarized_pyramid
}

fn project_two_way_node_partition(
    config: &Config,
    graph: WeightedGraph,
    coarsening_map: Vec<usize>,
    coarser_graph_where: Vec<usize>,
) -> BoundaryInfo {
    let graph_where = coarsening_map
        .iter()
        .map(|v| coarser_graph_where[*v])
        .collect();

    let (partition_weights, graph_boundary_ind, graph_boundary_ptr, graph_nr_info) =
        compute_two_way_node_partitioning_params(config, graph, graph_where);

    BoundaryInfo {
        _where: graph_where,
        partition_weights: partition_weights,
        boundary_ind: graph_boundary_ind,
        boundary_ptr: graph_boundary_ptr,
        nr_info: graph_nr_info,
    }
}

pub fn compute_two_way_node_partitioning_params(
    config: &Config,
    graph: WeightedGraph,
    graph_where: Vec<usize>,
) -> ([i32; 3], Vec<usize>, Vec<Option<usize>>, Vec<NrInfo>) {
    let mut partition_weights = [0, 0, 0];
    let mut graph_boundary_ind = Vec::new();
    let mut graph_boundary_ptr = vec![None; graph.graph.n_vertices()];
    let mut graph_nr_info = Vec::new();

    for i in 0..graph.graph.n_vertices() {
        let me = graph_where[i];
        partition_weights[me] += graph.vertex_weights.unwrap()[i];

        if me == 2 {
            // if it is on the separator do some computations
            graph_boundary_ind.push(i);
            graph_boundary_ptr[i] = Some(graph_boundary_ind.len() - 1);

            let mut nr_info = NrInfo { e_degrees: [0, 0] };
            for neighbor in graph.graph.neighbors(i) {
                let other = graph_where[*neighbor];
                if other != 2 {
                    nr_info.e_degrees[other] += graph.vertex_weights.unwrap()[*neighbor] as usize;
                }
            }

            graph_nr_info.push(nr_info);
        }
    }

    let min_cut = partition_weights[2];

    (
        partition_weights,
        graph_boundary_ind,
        graph_boundary_ptr,
        graph_nr_info,
    )
}