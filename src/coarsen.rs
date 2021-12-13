use crate::config::{CoarseningType, Config};
use crate::graph::{Graph, WeightedGraph};
use crate::random::RangeRng;
use std::convert::TryFrom;

#[derive(Debug)]
pub struct CoarseGraphResult {
    pub graph: WeightedGraph,
    pub total_vertex_weights: i32,
    pub coarsening_map: Vec<usize>,
}

const DEBUG_COARSEN: bool = false;

macro_rules! debug {
    ($($x: expr),*) => {
        if DEBUG_COARSEN {
            println!($($x,)*);
        }
    };
}

/**
 * Returns the coarsening pyramid, with the original graph first, followed by coarser levels in order
 */
pub fn coarsen_graph<RNG>(
    config: &Config,
    graph: WeightedGraph,
    coarsen_to: usize,
    total_vertex_weights: i32,
    rng: &mut RNG,
) -> Vec<CoarseGraphResult>
where
    RNG: RangeRng,
{
    let equal_edge_weights = graph
        .edge_weights
        .iter()
        .skip(1)
        .all(|w| *w == graph.edge_weights[0]);

    let max_coarsest_vertex_weight = (1.5 * total_vertex_weights as f32 / coarsen_to as f32) as i32;

    let mut pyramid = vec![CoarseGraphResult {
        graph: graph,
        total_vertex_weights: total_vertex_weights,
        coarsening_map: vec![],
    }];

    loop {
        let graph = &pyramid.last().as_ref().unwrap().graph;

        let last_n_vertices = graph.graph.n_vertices();

        let coarse_graph_result = match config.coarsening_type {
            CoarseningType::RM => match_random(config, graph, max_coarsest_vertex_weight, rng),
            CoarseningType::SHEM => {
                if equal_edge_weights || graph.graph.n_edges() == 0 {
                    match_random(config, graph, max_coarsest_vertex_weight, rng)
                } else {
                    match_shem(config, graph)
                }
            }
        };
        let graph = &coarse_graph_result.graph;

        let keep_going = graph.graph.n_vertices() > coarsen_to
            && graph.graph.n_vertices()
                < (config.coarsen_fraction() * (last_n_vertices as f32)) as usize
            && graph.graph.n_edges() > graph.graph.n_vertices() / 2;

        pyramid.push(coarse_graph_result);

        if !keep_going {
            break;
        }
    }

    pyramid
}

#[derive(Clone, Debug)]
enum Match {
    Unmatched,
    Matched(usize),
}

fn match_random<RNG>(
    config: &Config,
    graph: &WeightedGraph,
    max_coarsest_vertex_weight: i32,
    rng: &mut RNG,
) -> CoarseGraphResult
where
    RNG: RangeRng,
{
    let tperm = crate::random::permutation(
        graph.graph.n_vertices(),
        graph.graph.n_vertices() / 8,
        crate::random::Mode::Identity,
        rng,
    );

    // WTFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    let average_degree = (4.0
        * (graph.graph.x_adjacency[graph.graph.n_vertices()] / graph.graph.n_vertices()) as f32)
        as usize;
    let degrees = (0..graph.graph.n_vertices())
        .map(|i| {
            std::cmp::min(
                average_degree,
                ((1 + graph.graph.degree(i)) as f32).sqrt() as usize,
            )
        })
        .collect();

    // Again, why is tperm unused?
    let perm = crate::bucketsort::bucket_sort_keys_increasing(average_degree, &degrees, tperm);

    let mut matches = vec![Match::Unmatched; graph.graph.n_vertices()];

    let mut last_unmatched = 0;
    let mut n_unmatched = 0;
    let mut coarsening_map = vec![0; graph.graph.n_vertices()];
    let mut coarse_n_vertices = 0;

    for pi in 0..graph.graph.n_vertices() {
        let i = perm[pi];

        if let Match::Unmatched = matches[i] {
            let mut max_idx = Match::Matched(i);

            // ncon should always be 1
            if graph.vertex_weights[i] < max_coarsest_vertex_weight {
                if graph.graph.degree(i) == 0 {
                    last_unmatched = std::cmp::max(pi, last_unmatched) + 1;

                    loop {
                        let j = perm[last_unmatched];
                        if let Match::Unmatched = matches[j] {
                            max_idx = Match::Matched(j);
                            break;
                        }

                        last_unmatched += 1;
                        if last_unmatched == graph.graph.n_vertices() {
                            break;
                        }
                    }
                } else {
                    for &k in graph.graph.neighbors(i) {
                        if matches!(matches[k], Match::Unmatched)
                            && graph.vertex_weights[i] + graph.vertex_weights[k]
                                <= max_coarsest_vertex_weight
                        {
                            max_idx = Match::Matched(k);
                            break;
                        }
                    }

                    if matches!(max_idx, Match::Matched(idx) if idx == i)
                        && 2 * graph.vertex_weights[i] < max_coarsest_vertex_weight
                    {
                        n_unmatched += 1;
                        max_idx = Match::Unmatched;
                    }
                }
            }

            if let Match::Matched(max_idx) = max_idx {
                coarsening_map[i] = coarse_n_vertices;
                coarsening_map[max_idx] = coarse_n_vertices;
                coarse_n_vertices += 1;
                matches[i] = Match::Matched(max_idx);
                matches[max_idx] = Match::Matched(i);
            }
        }
    }

    let n_vertices = graph.graph.n_vertices();
    if config.two_hop_matching
        && n_unmatched > (config.unmatched_for_two_hop() * graph.graph.n_vertices() as f32) as usize
    {
        // Once again, this value is never read
        // coarse_n_vertices = match_two_hop(
        match_two_hop(
            config,
            graph,
            perm,
            &mut matches,
            coarse_n_vertices,
            n_unmatched,
            &mut coarsening_map,
        );
    }

    coarse_n_vertices = 0;
    for i in 0..n_vertices {
        match matches[i] {
            Match::Unmatched => {
                matches[i] = Match::Matched(i);
                coarsening_map[i] = coarse_n_vertices;
                coarse_n_vertices += 1;
            }
            Match::Matched(m) => {
                if i <= m {
                    coarsening_map[i] = coarse_n_vertices;
                    coarsening_map[m] = coarse_n_vertices;
                    coarse_n_vertices += 1;
                }
            }
        }
    }

    create_coarse_graph(config, graph, coarse_n_vertices, matches, coarsening_map)
}

fn match_shem(_config: &Config, _graph: &WeightedGraph) -> CoarseGraphResult {
    panic!();
}

fn match_two_hop(
    config: &Config,
    graph: &WeightedGraph,
    perm: Vec<usize>,
    matches: &mut Vec<Match>,
    mut coarse_n_vertices: usize,
    mut n_unmatched: usize,
    coarsening_map: &mut Vec<usize>,
) -> usize {
    let mut cnv_and_nunmatched = match_two_hop_any(
        config,
        graph,
        &perm,
        matches,
        coarse_n_vertices,
        n_unmatched,
        2,
        coarsening_map,
    );
    coarse_n_vertices = cnv_and_nunmatched.0;
    n_unmatched = cnv_and_nunmatched.1;

    cnv_and_nunmatched = match_two_hop_all(
        config,
        graph,
        &perm,
        matches,
        coarse_n_vertices,
        n_unmatched,
        64,
        coarsening_map,
    );
    coarse_n_vertices = cnv_and_nunmatched.0;
    n_unmatched = cnv_and_nunmatched.1;

    if n_unmatched
        > (1.5 * config.unmatched_for_two_hop() * graph.graph.n_vertices() as f32) as usize
    {
        cnv_and_nunmatched = match_two_hop_any(
            config,
            graph,
            &perm,
            matches,
            coarse_n_vertices,
            n_unmatched,
            3,
            coarsening_map,
        );
        coarse_n_vertices = cnv_and_nunmatched.0;
        n_unmatched = cnv_and_nunmatched.1;
    }

    if n_unmatched
        > (2.0 * config.unmatched_for_two_hop() * graph.graph.n_vertices() as f32) as usize
    {
        cnv_and_nunmatched = match_two_hop_any(
            config,
            graph,
            &perm,
            matches,
            coarse_n_vertices,
            n_unmatched,
            graph.graph.n_vertices(),
            coarsening_map,
        );
        coarse_n_vertices = cnv_and_nunmatched.0;
        // Unused:
        // n_unmatched = cnv_and_nunmatched.1;
    }

    coarse_n_vertices
}

fn match_two_hop_any(
    _config: &Config,
    graph: &WeightedGraph,
    perm: &Vec<usize>,
    matches: &mut Vec<Match>,
    mut coarse_n_vertices: usize,
    mut n_unmatched: usize,
    max_degree: usize,
    coarsening_map: &mut Vec<usize>,
) -> (usize, usize) {
    let mut col_ptr = vec![0; graph.graph.n_vertices() + 1];
    for i in 0..graph.graph.n_vertices() {
        if matches!(matches[i], Match::Unmatched) && graph.graph.degree(i) < max_degree {
            for neighbor in graph.graph.neighbors(i) {
                col_ptr[*neighbor] += 1;
            }
        }
    }

    crate::bucketsort::make_csr(graph.graph.n_vertices(), &mut col_ptr);

    let mut row_ind = vec![0; col_ptr[graph.graph.n_vertices()]];
    for pi in 0..graph.graph.n_vertices() {
        let i = perm[pi];
        if matches!(matches[i], Match::Unmatched) && graph.graph.degree(i) < max_degree {
            for neighbor in graph.graph.neighbors(i) {
                row_ind[col_ptr[*neighbor]] = i;
                col_ptr[*neighbor] += 1;
            }
        }
    }
    crate::bucketsort::shift_csr(graph.graph.n_vertices(), &mut col_ptr);

    for pi in 0..graph.graph.n_vertices() {
        let i = perm[pi];
        if col_ptr[i + 1] - col_ptr[i] < 2 {
            continue;
        }

        let mut j = col_ptr[i];
        let mut j_end = col_ptr[i + 1];
        while j < j_end {
            if matches!(matches[row_ind[j]], Match::Unmatched) {
                j_end -= 1;
                while j < j_end {
                    if matches!(matches[row_ind[j_end]], Match::Unmatched) {
                        coarsening_map[row_ind[j]] = coarse_n_vertices;
                        coarsening_map[row_ind[j_end]] = coarse_n_vertices;
                        matches[row_ind[j]] = Match::Matched(row_ind[j_end]);
                        matches[row_ind[j_end]] = Match::Matched(row_ind[j]);
                        coarse_n_vertices += 1;
                        n_unmatched -= 2;
                        break;
                    }

                    j_end -= 1;
                }
            }

            j += 1;
        }
    }

    (coarse_n_vertices, n_unmatched)
}

fn match_two_hop_all(
    _config: &Config,
    graph: &WeightedGraph,
    perm: &Vec<usize>,
    matches: &mut Vec<Match>,
    mut coarse_n_vertices: usize,
    mut n_unmatched: usize,
    max_degree: usize,
    coarsening_map: &mut Vec<usize>,
) -> (usize, usize) {
    let mask = usize::MAX / max_degree;

    let mut keys_unsorted = Vec::<crate::graph::KeyAndValue>::new();
    for pi in 0..graph.graph.n_vertices() {
        let i = perm[pi];
        let degree = graph.graph.degree(i);
        if matches!(matches[i], Match::Unmatched) && degree > 1 && degree < max_degree {
            let mut k = 0;
            for neighbor in graph.graph.neighbors(i) {
                k += *neighbor % mask;
            }

            keys_unsorted.push(crate::graph::KeyAndValue {
                key: (k % mask) * max_degree + degree,
                value: i,
            });
        }
    }
    let keys = crate::graph::sort(keys_unsorted);

    let mut mark = vec![0; graph.graph.n_vertices()];
    for pi in 0..keys.len() {
        let i = keys[pi].value;

        if matches!(matches[i], Match::Unmatched) {
            continue;
        }

        for neighbor in graph.graph.neighbors(i) {
            mark[*neighbor] = i;
        }

        for pk in pi + 1..keys.len() {
            let k = keys[pk].value;

            if matches!(matches[k], Match::Unmatched) {
                continue;
            }

            if keys[pi].key != keys[pk].key {
                break;
            }

            if graph.graph.degree(i) != graph.graph.degree(k) {
                break;
            }

            let mut neighbors_match = true;
            for neighbor in graph.graph.neighbors(k) {
                if mark[*neighbor] != i {
                    neighbors_match = false;
                    break;
                }
            }

            if neighbors_match {
                coarsening_map[i] = coarse_n_vertices;
                coarsening_map[k] = coarse_n_vertices;
                coarse_n_vertices += 1;
                matches[i] = Match::Matched(k);
                matches[k] = Match::Matched(i);
                n_unmatched -= 2;
                break;
            }
        }
    }

    (coarse_n_vertices, n_unmatched)
}

fn create_coarse_graph(
    config: &Config,
    graph: &WeightedGraph,
    coarse_n_vertices: usize,
    matches: Vec<Match>,
    coarsening_map: Vec<usize>,
) -> CoarseGraphResult {
    debug!("CreateCoarseGraph!!!");
    debug!(
        "graph: {}, {}: {:?}",
        graph.graph.n_vertices(),
        graph.graph.n_edges(),
        graph.graph.adjacency_lists
    );
    debug!("coarse_n: {}", coarse_n_vertices);
    debug!("matches: {:?}", matches);
    debug!("coarsening: {:?}", coarsening_map);
    let do_vsize = matches!(config.objective_type, crate::config::ObjectiveType::VOLUME);
    if do_vsize {
        panic!();
    }

    let mask = (1 << 13) - 1;

    // drop_edges is false

    let mut hash_table: Vec<i32> = vec![-1; mask + 1];
    // let mut direct_table = vec![-1; coarse_n_vertices];

    let mut coarse_vertex_weights = Vec::<i32>::new();
    let mut coarse_x_adjacency = vec![0];
    let mut coarse_adjacency = Vec::<usize>::new();
    let mut coarse_adjacency_weights = Vec::<i32>::new();
    for v in 0..graph.graph.n_vertices() {
        let u = match matches[v] {
            Match::Matched(m) => m,
            Match::Unmatched => panic!(),
        };

        if u < v {
            continue;
        }

        // ncon is 1
        coarse_vertex_weights.push(graph.vertex_weights[v]);

        if v != u {
            *coarse_vertex_weights.last_mut().unwrap() += graph.vertex_weights[u];
        }

        // take care of the edges
        if graph.graph.degree(v) + graph.graph.degree(u) < mask >> 2 {
            // use mask
            let coarse_n_vertices = coarse_x_adjacency.len() - 1;
            hash_table[coarse_n_vertices & mask] = 0;
            coarse_adjacency.push(coarse_n_vertices);
            coarse_adjacency_weights.push(-1);

            for j in graph.graph.x_adjacency[v]..graph.graph.x_adjacency[v + 1] {
                let neighbor = graph.graph.adjacency_lists[j];
                let k = coarsening_map[neighbor];

                let mut kk = k & mask;
                while hash_table[kk] != -1
                    && coarse_adjacency[usize::try_from(hash_table[kk]).unwrap()
                        + coarse_x_adjacency.last().unwrap()]
                        != k
                {
                    kk = (kk + 1) & mask;
                }

                debug!(
                    "hash_table: {:?}",
                    hash_table
                        .iter()
                        .enumerate()
                        .filter(|(_i, &e)| e >= 0)
                        .collect::<Vec<(usize, &i32)>>()
                );
                debug!("{:?}", coarse_adjacency);

                let m = hash_table[kk];
                if m == -1 {
                    coarse_adjacency.push(k);
                    coarse_adjacency_weights.push(graph.edge_weights[j]);
                    hash_table[kk] =
                        i32::try_from(coarse_adjacency.len() - coarse_x_adjacency.last().unwrap())
                            .unwrap()
                            - 1;
                } else {
                    coarse_adjacency_weights
                        [usize::try_from(m).unwrap() + coarse_x_adjacency.last().unwrap()] +=
                        graph.edge_weights[j];
                }
            }

            debug!("ca: {:?}", coarse_adjacency);
            debug!("ca_w: {:?}", coarse_adjacency_weights);
            debug!(
                "hash_table: {:?}",
                hash_table
                    .iter()
                    .enumerate()
                    .filter(|(_i, &e)| e >= 0)
                    .collect::<Vec<(usize, &i32)>>()
            );

            if v != u {
                for j in graph.graph.x_adjacency[u]..graph.graph.x_adjacency[u + 1] {
                    let neighbor = graph.graph.adjacency_lists[j];
                    let k = coarsening_map[neighbor];

                    let mut kk = k & mask;
                    while hash_table[kk] != -1
                        && coarse_adjacency[usize::try_from(hash_table[kk]).unwrap()
                            + coarse_x_adjacency.last().unwrap()]
                            != k
                    {
                        kk = (kk + 1) & mask;
                    }

                    let m = hash_table[kk];
                    if m == -1 {
                        coarse_adjacency.push(k);
                        coarse_adjacency_weights.push(graph.edge_weights[j]);
                        hash_table[kk] = i32::try_from(
                            coarse_adjacency.len() - coarse_x_adjacency.last().unwrap(),
                        )
                        .unwrap()
                            - 1;
                    } else {
                        coarse_adjacency_weights
                            [usize::try_from(m).unwrap() + coarse_x_adjacency.last().unwrap()] +=
                            graph.edge_weights[j];
                    }
                }
            }

            // zero out the hash table
            debug!("zeroing");
            for &k in &coarse_adjacency[*coarse_x_adjacency.last().unwrap()..] {
                let mut kk = k & mask;
                debug!("k: {}", k);
                while coarse_adjacency[usize::try_from(
                    hash_table[kk] + i32::try_from(*coarse_x_adjacency.last().unwrap()).unwrap(),
                )
                .unwrap()]
                    != k
                {
                    debug!(
                        "kk: {}, htable: {}, cadj: {}",
                        kk,
                        hash_table[kk],
                        coarse_adjacency[usize::try_from(
                            hash_table[kk]
                                + i32::try_from(*coarse_x_adjacency.last().unwrap()).unwrap()
                        )
                        .unwrap()]
                    );
                    kk = (kk + 1) & mask;
                }
                hash_table[kk] = -1;
            }
            debug!("Zeroed");

            // remove the contracted vertex from the list
            debug!(
                "ca len: {}, caw len: {}, cxa last: {}",
                coarse_adjacency.len(),
                coarse_adjacency_weights.len(),
                coarse_x_adjacency.last().unwrap()
            );
            if coarse_adjacency.len() > *coarse_x_adjacency.last().unwrap() + 1 {
                coarse_adjacency[*coarse_x_adjacency.last().unwrap()] =
                    coarse_adjacency.pop().unwrap();
                coarse_adjacency_weights[*coarse_x_adjacency.last().unwrap()] =
                    coarse_adjacency_weights.pop().unwrap();
            } else {
                coarse_adjacency.pop();
                coarse_adjacency_weights.pop();
            }
        } else {
            // don't use mask
            panic!();

            // for j in graph.graph.x_adjacency[v]..graph.graph.x_adjacency[v + 1] {
            //     let neighbor = graph.graph.adjacency_lists[j];
            //     let k = coarsening_map[neighbor];

            //     let m = direct_table[k];
            //     if m == -1 {
            //         coarse_adjacency.push(k);
            //         coarse_adjacency_weights.push(graph.edge_weights.as_ref().unwrap()[j]);
            //         direct_table[k] = coarse_adjacency.len() as i32 - 1;
            //     } else {
            //         coarse_adjacency_weights[m as usize] += graph.edge_weights.as_ref().unwrap()[j];
            //     }
            // }

            // if v != u {
            //     for j in graph.graph.x_adjacency[u]..graph.graph.x_adjacency[u + 1] {
            //         let neighbor = graph.graph.adjacency_lists[j];
            //         let k = coarsening_map[neighbor];
            //         let m = direct_table[k];
            //         if m == -1 {
            //             coarse_adjacency.push(k);
            //             coarse_adjacency_weights.push(graph.edge_weights.as_ref().unwrap()[j]);
            //             direct_table[k] = coarse_adjacency.len() as i32 - 1;
            //         } else {
            //             coarse_adjacency_weights[m as usize] +=
            //                 graph.edge_weights.as_ref().unwrap()[j];
            //         }
            //     }

            //     // remove the contracted self-loop, when present
            //     let j = direct_table[coarse_vertex_weights.len() - 1];
            //     if j != -1 {
            //         coarse_adjacency[j as usize] = coarse_adjacency.pop().unwrap();
            //         coarse_adjacency_weights[j as usize] = coarse_adjacency_weights.pop().unwrap();
            //         direct_table[coarse_vertex_weights.len() - 1] = -1;
            //     }
            // }

            // // zero out the direct table
            // for &k in coarse_adjacency.iter() {
            //     direct_table[k] = -1;
            // }
        }

        // dropedges is false

        coarse_x_adjacency.push(coarse_adjacency.len());
    }

    // compact the adjacency structure of the coarser graph to keep only +ve edges
    // dropedges is false

    let total_vertex_weights = coarse_vertex_weights.iter().sum();

    CoarseGraphResult {
        graph: WeightedGraph {
            graph: Graph {
                x_adjacency: coarse_x_adjacency,
                adjacency_lists: coarse_adjacency,
            },
            vertex_weights: coarse_vertex_weights,
            edge_weights: coarse_adjacency_weights,
        },
        total_vertex_weights: total_vertex_weights,
        coarsening_map: coarsening_map,
    }
}
