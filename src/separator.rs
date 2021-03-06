use crate::config::{Config, DEBUG_SEPARATOR};
use crate::graph::WeightedGraph;
use crate::random::RangeRng;
use crate::refinement::{BoundaryInfo, WhereIdEd};

macro_rules! debug {
    ($($x: expr),*) => {
        if DEBUG_SEPARATOR {
            println!($($x,)*);
        }
    };
}

pub fn construct_separator<RNG>(
    config: &Config,
    graph: &WeightedGraph,
    boundary_info: BoundaryInfo,
    where_id_ed: &WhereIdEd,
    graph_is_compressed: bool,
    rng: &mut RNG,
) -> (i32, BoundaryInfo, Vec<usize>)
where
    RNG: RangeRng,
{
    debug!("ENTERED CONSTRUCT_SEPARATOR");
    debug!("graph: {:?}", graph);
    debug!("{:?}", boundary_info);
    debug!("{:?}", where_id_ed);
    let mut _where = where_id_ed._where.clone();

    // put the nodes in the boundary into the separator
    for j in boundary_info.boundary_ind {
        if graph.graph.degree(j) > 0 {
            _where[j] = 2;
        }
    }

    // free boundary_info, where_id_ed, nr_info
    // allocate sr::BoundaryInfo

    let (partition_weights, graph_boundary_ind, graph_boundary_ptr, graph_nr_info) =
        crate::separator_refinement::compute_two_way_node_partitioning_params(
            config, graph, &_where,
        );

    let (_min_cut, boundary_info) = crate::fm_separator_refinement::two_way_node_refine_two_sided(
        config,
        graph,
        crate::separator_refinement::BoundaryInfo {
            _where: _where,
            boundary_ind: graph_boundary_ind,
            boundary_ptr: graph_boundary_ptr,
            partition_weights: partition_weights,
            nr_info: graph_nr_info,
        },
        1,
        rng,
    );

    let (min_cut, boundary_info) = crate::fm_separator_refinement::two_way_node_refine_one_sided(
        config,
        graph,
        boundary_info,
        4,
        graph_is_compressed,
        rng,
    );

    (
        min_cut,
        BoundaryInfo {
            partition_weights: boundary_info.partition_weights,
            boundary_ind: boundary_info.boundary_ind,
            boundary_ptr: boundary_info.boundary_ptr,
        },
        boundary_info._where, // Previously had _where here?
    )
}
