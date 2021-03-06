use crate::config::{Config, Index, RefinementType, DEBUG_SEPARATOR_REFINEMENT};
use crate::graph::WeightedGraph;
use crate::random::RangeRng;
use std::fmt;

macro_rules! debug {
    ($($x: expr),*) => {
        if DEBUG_SEPARATOR_REFINEMENT {
            println!($($x,)*);
        }
    };
}

#[derive(Debug)]
pub struct NrInfo {
    pub e_degrees: [usize; 2],
}

pub struct BoundaryInfo {
    pub _where: Vec<usize>,
    pub partition_weights: [i32; 3],
    pub boundary_ind: Vec<usize>,
    pub boundary_ptr: Vec<Option<usize>>,
    pub nr_info: Vec<Option<NrInfo>>,
}

impl fmt::Debug for BoundaryInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // skip nr_info since it's super annoying to print from metis
        f.debug_struct("BoundaryInfo")
            .field("_where", &self._where)
            .field("partition_weights", &self.partition_weights)
            .field("boundary_ind", &self.boundary_ind)
            .field("boundary_ptr", &self.boundary_ptr)
            .finish()
    }
}

impl BoundaryInfo {
    pub fn delete(&mut self, i: usize) {
        let index_to_update = self.boundary_ptr[i].unwrap();
        if index_to_update + 1 == self.boundary_ind.len() {
            self.boundary_ind.pop();
            self.boundary_ptr[i] = None;
        } else {
            let value_to_move = self.boundary_ind.pop();
            self.boundary_ind[index_to_update] = value_to_move.unwrap();
            self.boundary_ptr[value_to_move.unwrap()] = Some(index_to_update);
            self.boundary_ptr[i] = None;
        }
    }

    pub fn insert(&mut self, i: usize) {
        self.boundary_ind.push(i);
        self.boundary_ptr[i] = Some(self.boundary_ind.len() - 1);
    }
}

#[derive(Debug)]
pub struct GraphPyramidLevel {
    pub graph: WeightedGraph,
    pub coarsening_map: Vec<usize>,
    pub coarser_graph_where: Vec<usize>,
    pub total_vertex_weights: i32,
}

#[derive(Debug)]
pub struct BoundarizedGraphPyramidLevel {
    pub graph: WeightedGraph,
    // Do we need this whole thing?
    pub boundary_info: BoundaryInfo,
}

pub fn refine_two_way_node<RNG>(
    config: &Config,
    graph_pyramid: Vec<GraphPyramidLevel>,
    org_graph: usize,
    mut graph: usize,
    graph_is_compressed: bool,
    rng: &mut RNG,
) -> (Vec<BoundarizedGraphPyramidLevel>, Index)
where
    RNG: RangeRng,
{
    debug!("CALLED refine_two_way_node");

    let mut boundarized_pyramid = Vec::new();
    let mut min_cut;

    if graph == org_graph {
        panic!();
    } else {
        // get rid of this copy?
        let mut _where = graph_pyramid[graph].coarser_graph_where.clone();

        loop {
            graph -= 1;

            debug!(
                "graph: {}\nwhere: {:?}",
                graph_pyramid.len() - graph - 1,
                _where
            );
            let boundary_info = project_two_way_node_partition(
                config,
                &graph_pyramid[graph].graph,
                &graph_pyramid[graph + 1].coarsening_map,
                &_where,
            );

            // delete graph->coarser

            let boundary_info = crate::fm_separator_refinement::two_way_node_balance(
                config,
                &graph_pyramid[graph].graph,
                boundary_info,
                graph_pyramid[graph].total_vertex_weights,
                rng,
            );

            let (new_min_cut, boundary_info) = match config.refinement_type {
                RefinementType::SEP1SIDED => {
                    crate::fm_separator_refinement::two_way_node_refine_one_sided(
                        config,
                        &graph_pyramid[graph].graph,
                        boundary_info,
                        config.n_iterations,
                        graph_is_compressed,
                        rng,
                    )
                }
                RefinementType::SEP2SIDED => {
                    crate::fm_separator_refinement::two_way_node_refine_two_sided(
                        config,
                        &graph_pyramid[graph].graph,
                        boundary_info,
                        config.n_iterations,
                        rng,
                    )
                }
                _ => panic!("What's this"),
            };

            min_cut = new_min_cut;
            _where = boundary_info._where.clone();

            boundarized_pyramid.push(BoundarizedGraphPyramidLevel {
                graph: graph_pyramid[graph].graph.clone(), // Eek, this is unnecessary
                boundary_info: boundary_info,
            });

            if graph == org_graph {
                break;
            }
        }
    }

    debug!("EXITED refine_two_way_node");

    (boundarized_pyramid, min_cut)
}

fn project_two_way_node_partition(
    config: &Config,
    graph: &WeightedGraph,
    coarsening_map: &Vec<usize>,
    coarser_graph_where: &Vec<usize>,
) -> BoundaryInfo {
    debug!("CALLED project_two_way_node_partition");

    debug!("graph: {:?}", graph);
    debug!("coarsening_map: {:?}", coarsening_map);
    debug!("coarser_graph_where: {:?}", coarser_graph_where);

    let graph_where = coarsening_map
        .iter()
        .map(|&v| coarser_graph_where[v])
        .collect();

    let (partition_weights, graph_boundary_ind, graph_boundary_ptr, graph_nr_info) =
        compute_two_way_node_partitioning_params(config, graph, &graph_where);

    debug!("EXITED project_two_way_node_partition");

    BoundaryInfo {
        _where: graph_where,
        partition_weights: partition_weights,
        boundary_ind: graph_boundary_ind,
        boundary_ptr: graph_boundary_ptr,
        nr_info: graph_nr_info,
    }
}

pub fn compute_two_way_node_partitioning_params(
    _config: &Config,
    graph: &WeightedGraph,
    graph_where: &Vec<usize>,
) -> (
    [i32; 3],
    Vec<usize>,
    Vec<Option<usize>>,
    Vec<Option<NrInfo>>,
) {
    let mut partition_weights = [0, 0, 0];
    let mut graph_boundary_ind = Vec::new();
    let mut graph_boundary_ptr = vec![None; graph.graph.n_vertices()];
    let mut graph_nr_info = Vec::new();

    for i in 0..graph.graph.n_vertices() {
        let me = graph_where[i];
        partition_weights[me] += graph.vertex_weights[i];

        if me == 2 {
            // if it is on the separator do some computations
            graph_boundary_ind.push(i);
            graph_boundary_ptr[i] = Some(graph_boundary_ind.len() - 1);

            let mut nr_info = NrInfo { e_degrees: [0, 0] };
            for &neighbor in graph.graph.neighbors(i) {
                let other = graph_where[neighbor];
                if other != 2 {
                    nr_info.e_degrees[other] += graph.vertex_weights[neighbor] as usize;
                }
            }

            graph_nr_info.push(Some(nr_info));
        } else {
            graph_nr_info.push(None)
        }
    }

    let _min_cut = partition_weights[2];

    (
        partition_weights,
        graph_boundary_ind,
        graph_boundary_ptr,
        graph_nr_info,
    )
}
