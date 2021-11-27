mod ometis;
mod config;
mod graph;
mod mmd;
mod io;
mod coarsen;
mod random;
mod bucketsort;
mod separator_refinement;
mod fm_separator_refinement;
mod priority_queue;
mod initialize_partition;
mod refinement;
mod balance;
mod mc_util;
mod fm;
mod separator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = crate::io::read_graph("/home/aaron/metis-cpp/METIS/graphs/test.mgraph")?;
    crate::ometis::node_nd(graph.graph, graph.vertex_weights, &mut rand::thread_rng())?;
    Ok(())
}
