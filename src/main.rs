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
mod mmd;
mod ometis;
mod priority_queue;
mod random;
mod refinement;
mod separator;
mod separator_refinement;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    //let graph = crate::io::read_graph("/home/aaron/metis-cpp/METIS/graphs/test.mgraph")?;

    let mat_name = "bcsstk01";
    let graph = crate::io_mtx::read_graph(format!(
        "/home/aaron/matrices/{}/{}.mtx",
        mat_name, mat_name
    ))?;
    println!("{:?}", graph.graph.x_adjacency);
    println!("{:?}", graph.graph.adjacency_lists);
    for node in 0..graph.graph.n_vertices() {
        println!("{}: {:?}", node, graph.graph.neighbors(node));
    }

    let mut rng =
        crate::random::MockRng::from_trace(format!("/home/aaron/rng_traces/{}.txt", mat_name))?;

    crate::ometis::node_nd(graph.graph, graph.vertex_weights, &mut rng)?;
    Ok(())
}
