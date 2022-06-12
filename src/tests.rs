use crate::metis_ffi::run_metis;
use std::collections::HashSet;

#[allow(dead_code)]
fn test_metis_equivalence_mat(mat_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("TESTING {}", mat_name);
    let graph = crate::io_mtx::read_graph(format!(
        "/home/aaron/SMITE/test/matrices/{}/{}.mtx",
        mat_name, mat_name
    ))?;

    if graph.graph.n_vertices() >= 5000 {
        // not implemented yet
        println!("SKIPPING, too big: {}", graph.graph.n_vertices());
        return Ok(());
    }

    let metis_result = run_metis(mat_name, graph.graph.n_vertices());
    println!("METIS RESULT: {:?}", metis_result);

    let mut rng = crate::random::MockRng::from_trace(format!(
        "/home/aaron/SMITE/test/rng_traces/{}.txt",
        mat_name
    ))?;

    let result = crate::ometis::node_nd(graph.graph, graph.vertex_weights, &mut rng)?;
    println!("SMITE RESULT: {:?}", result);

    assert_eq!(metis_result, result.permutation);

    println!("{} PASSED", mat_name);
    Ok(())
}

pub fn test_metis_equivalence_all() -> Result<(), Box<dyn std::error::Error>> {
    let paths = std::fs::read_dir("/home/aaron/SMITE/test/matrices")?;

    //let skip = HashSet::from(["494_bus", "nos5", "bcsstk07"]);
    let skip = HashSet::<String>::new();

    for path in paths {
        if !path.as_ref().unwrap().file_type()?.is_dir() {
            continue;
        }
        let mat_name = path?.file_name().into_string().unwrap();

        if skip.contains(mat_name.as_str()) {
            println!("Skipping test {}", mat_name);
            continue;
        }

        test_metis_equivalence_mat(&mat_name)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metis_equivalence() -> Result<(), Box<dyn std::error::Error>> {
        test_metis_equivalence_all()
    }
}
