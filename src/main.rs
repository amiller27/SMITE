mod balance;
mod bucketsort;
mod coarsen;
mod config;
mod fm;
mod fm_separator_refinement;
mod graph;
mod initialize_partition;
mod io;
mod io_mtx;
mod mc_util;
mod metis_ffi;
mod mmd;
mod ometis;
mod priority_queue;
mod random;
mod refinement;
mod separator;
mod separator_refinement;
mod tests;
mod qsort;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let mat_name = "494_bus";

    // let graph = crate::io_mtx::read_graph(format!(
    //     "/home/aaron/SMITE/test/matrices/{}/{}.mtx",
    //     mat_name, mat_name
    // ))?;

    // if graph.graph.n_vertices() >= 5000 {
    //     // not implemented yet
    //     return Ok(());
    // }

    // let mut rng = crate::random::MockRng::from_trace(format!(
    //     "/home/aaron/SMITE/test/rng_traces/{}.txt",
    //     mat_name
    // ))?;

    // let result = crate::ometis::node_nd(graph.graph, graph.vertex_weights, &mut rng)?;
    // println!("SMITE RESULT: {:?}", result);

    // Ok(())

    tests::test_metis_equivalence_all()
}
