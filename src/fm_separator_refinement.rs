use crate::config::Config;
use crate::graph::WeightedGraph;
use crate::priority_queue::PriorityQueue;
use crate::random::RangeRng;
use crate::separator_refinement::BoundaryInfo;

#[derive(Clone, Debug)]
enum Moved {
    None,
    Some(usize),
    Neg(usize),
}

const DEBUG_TWO_SIDED: bool = false;

macro_rules! debug_two {
    ($($x: expr),*) => {
        if DEBUG_TWO_SIDED {
            println!($($x,)*);
        }
    };
}

const DEBUG_ONE_SIDED: bool = false;

macro_rules! debug_one {
    ($($x: expr),*) => {
        if DEBUG_ONE_SIDED {
            println!($($x,)*);
        }
    };
}

const DEBUG_BALANCE: bool = false;

macro_rules! debug_bal {
    ($($x: expr),*) => {
        if DEBUG_BALANCE {
            println!($($x,)*);
        }
    };
}

pub fn two_way_node_refine_two_sided<RNG>(
    config: &Config,
    graph: &WeightedGraph,
    mut boundary_info: BoundaryInfo,
    n_iterations: i32,
    rng: &mut RNG,
) -> (i32, BoundaryInfo)
where
    RNG: RangeRng,
{
    debug_two!("ENTERING TWO_WAY_NODE_REFINE_TWO_SIDED");
    debug_two!("{:?}", graph);
    debug_two!("{:?}", boundary_info);
    debug_two!("{}", n_iterations);

    let mult = 0.5 * config.ub_factors[0];
    let bad_max_partition_weight =
        (mult * (boundary_info.partition_weights.iter().sum::<i32>() as f32)) as usize;

    let mut min_cut_result = None;

    for pass in 0..n_iterations {
        let mut moved = vec![Moved::None; graph.graph.n_vertices()];

        let mut queues = [
            PriorityQueue::create(graph.graph.n_vertices()),
            PriorityQueue::create(graph.graph.n_vertices()),
        ];

        let mut min_cut_order: i32 = -1;
        let init_cut = boundary_info.partition_weights[2];
        let mut min_cut = boundary_info.partition_weights[2];
        let n_boundary = boundary_info.boundary_ind.len();

        // use the swaps array in place of the traditional perm array to save memory
        // NOTE(aaron): well fuck that
        let perm =
            crate::random::permutation(n_boundary, n_boundary, crate::random::Mode::Identity, rng);

        for ii in 0..n_boundary {
            let i = boundary_info.boundary_ind[perm[ii]];
            queues[0].insert(
                i,
                (graph.vertex_weights[i]
                    - boundary_info.nr_info[i].as_ref().unwrap().e_degrees[1] as i32)
                    as f32,
            );
            queues[1].insert(
                i,
                (graph.vertex_weights[i]
                    - boundary_info.nr_info[i].as_ref().unwrap().e_degrees[0] as i32)
                    as f32,
            );
        }

        let limit = if config.compress_before_ordering {
            std::cmp::min(5 * n_boundary, 400)
        } else {
            std::cmp::min(2 * n_boundary, 300)
        };

        // get into the FM loop
        let mut m_ptr = Vec::new();
        let mut min_diff =
            (boundary_info.partition_weights[0] - boundary_info.partition_weights[1]).abs();
        // Holy shit, rust statically proves that this value is never read
        //let mut to = if boundary_info.partition_weights[0] < boundary_info.partition_weights[1] {
        //    0
        //} else {
        //    1
        //};
        let mut to;
        let mut swaps = Vec::new();
        for n_swaps in 0..graph.graph.n_vertices() {
            for _i in 0..5 {
                debug_two!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            }
            debug_two!("N_SWAPS: {}", n_swaps);

            debug_two!("{:?}", boundary_info);
            debug_two!("swaps: {:?}", swaps);
            debug_two!("min_cut_result: {:?}", min_cut_result);
            debug_two!("moved: {:?}", moved);
            debug_two!("queues: {:?}", queues);
            debug_two!("min_cut_order: {}", min_cut_order);
            debug_two!("min_cut: {}", min_cut);
            debug_two!("m_ptr: {:?}", m_ptr);
            debug_two!("min_diff: {}", min_diff);

            let mut m_ind = Vec::new();

            let maybe_u = [queues[0].peek(), queues[1].peek()];

            if maybe_u[0].is_some() && maybe_u[1].is_some() {
                let u = [maybe_u[0].unwrap(), maybe_u[1].unwrap()];

                let g = [
                    graph.vertex_weights[u[0]]
                        - boundary_info.nr_info[u[0]].as_ref().unwrap().e_degrees[1] as i32,
                    graph.vertex_weights[u[1]]
                        - boundary_info.nr_info[u[1]].as_ref().unwrap().e_degrees[0] as i32,
                ];

                to = if g[0] > g[1] {
                    0
                } else if g[0] < g[1] {
                    1
                } else {
                    (pass % 2) as usize
                };

                if boundary_info.partition_weights[to] + graph.vertex_weights[u[to]]
                    > bad_max_partition_weight as i32
                {
                    to = if to == 0 { 1 } else { 0 };
                }
            } else if maybe_u[0].is_none() && maybe_u[1].is_none() {
                break;
            } else if maybe_u[0].is_some()
                && boundary_info.partition_weights[0] + graph.vertex_weights[maybe_u[0].unwrap()]
                    <= bad_max_partition_weight as i32
            {
                to = 0;
            } else if maybe_u[1].is_some()
                && boundary_info.partition_weights[1] + graph.vertex_weights[maybe_u[1].unwrap()]
                    <= bad_max_partition_weight as i32
            {
                to = 1;
            } else {
                break;
            }

            let other = if to == 0 { 1 } else { 0 };

            let high_gain = queues[to].pop().unwrap();

            if matches!(moved[high_gain], Moved::None) {
                // delete if it was in the separator originally
                queues[other].delete(high_gain);
            }

            // the following check is to ensure we break out if there is a possibility of
            // over-running the m_ind array
            // NOTE(aaron): Probably not necessary with Vec?
            if m_ptr.iter().map(|l: &Vec<usize>| l.len()).sum::<usize>()
                + m_ind.len()
                + graph.graph.degree(high_gain)
                >= 2 * graph.graph.n_vertices() - 1
            {
                panic!();
            }

            boundary_info.partition_weights[2] -= graph.vertex_weights[high_gain]
                - boundary_info.nr_info[high_gain].as_ref().unwrap().e_degrees[other] as i32;

            let new_diff = (boundary_info.partition_weights[to] + graph.vertex_weights[high_gain]
                - (boundary_info.partition_weights[other]
                    - boundary_info.nr_info[high_gain].as_ref().unwrap().e_degrees[other] as i32))
                .abs();

            if boundary_info.partition_weights[2] < min_cut
                || (boundary_info.partition_weights[2] == min_cut && new_diff < min_diff)
            {
                min_cut = boundary_info.partition_weights[2];
                min_cut_order = n_swaps as i32;
                min_diff = new_diff;
            } else if n_swaps as i32 - min_cut_order > 2 * limit as i32
                || (n_swaps as i32 - min_cut_order > limit as i32
                    && boundary_info.partition_weights[2] as f32 > 1.10 * min_cut as f32)
            {
                boundary_info.partition_weights[2] += graph.vertex_weights[high_gain]
                    - boundary_info.nr_info[high_gain].as_ref().unwrap().e_degrees[other] as i32;
                break; // no further improvement, break out
            }

            boundary_info.delete(high_gain);

            boundary_info.partition_weights[to] += graph.vertex_weights[high_gain];
            boundary_info._where[high_gain] = to;
            moved[high_gain] = Moved::Some(n_swaps);
            debug_two!("SETTING MOVED {} = {:?}", high_gain, Moved::Some(n_swaps));
            swaps.push(high_gain);

            // update the degrees of the affected nodes
            for &k in graph.graph.neighbors(high_gain) {
                if boundary_info._where[k] == 2 {
                    // for the in-separator vertices modify their edegree[to]
                    let old_gain = graph.vertex_weights[k]
                        - boundary_info.nr_info[k].as_ref().unwrap().e_degrees[to] as i32;
                    boundary_info.nr_info[k].as_mut().unwrap().e_degrees[to] +=
                        graph.vertex_weights[high_gain] as usize; // eek

                    debug_two!(
                        "----------- Branch 1, Moved {:?}, other {}, k {}, new {}",
                        moved[k],
                        other,
                        k,
                        (old_gain - graph.vertex_weights[high_gain]) as f32
                    );
                    match moved[k] {
                        Moved::None => queues[other]
                            .update(k, (old_gain - graph.vertex_weights[high_gain]) as f32),
                        Moved::Neg(i) if i == other => queues[other]
                            .update(k, (old_gain - graph.vertex_weights[high_gain]) as f32),
                        _ => {}
                    }
                } else if boundary_info._where[k] == other {
                    // the vertex is pulled into the separator
                    boundary_info.boundary_ind.push(k);
                    boundary_info.boundary_ptr[k] = Some(boundary_info.boundary_ind.len() - 1);

                    m_ind.push(k); // keep track for rollback
                    boundary_info._where[k] = 2;
                    boundary_info.partition_weights[other] -= graph.vertex_weights[k];

                    debug_two!("--------- Branch 2");
                    debug_two!("{:?}", boundary_info._where);
                    let mut e_degrees = [0, 0];
                    for &kk in graph.graph.neighbors(k) {
                        if boundary_info._where[kk] != 2 {
                            e_degrees[boundary_info._where[kk]] +=
                                graph.vertex_weights[kk] as usize;
                        } else {
                            let old_gain = graph.vertex_weights[kk]
                                - boundary_info.nr_info[kk].as_ref().unwrap().e_degrees[other]
                                    as i32;
                            boundary_info.nr_info[kk].as_mut().unwrap().e_degrees[other] -=
                                graph.vertex_weights[k] as usize;
                            debug_two!(
                                "Moved {:?}, to {}, k {}, kk {}, new {}",
                                moved[kk],
                                to,
                                k,
                                kk,
                                (old_gain + graph.vertex_weights[k]) as f32
                            );
                            match moved[kk] {
                                Moved::None => queues[to]
                                    .update(kk, (old_gain + graph.vertex_weights[k]) as f32),
                                Moved::Neg(i) if i == to => queues[to]
                                    .update(kk, (old_gain + graph.vertex_weights[k]) as f32),
                                _ => {}
                            }
                        }
                    }
                    debug_two!("Set nrinfo {} to {:?}", k, e_degrees);
                    boundary_info.nr_info[k] = Some(crate::separator_refinement::NrInfo {
                        e_degrees: e_degrees,
                    });

                    // insert the new vertex into the priority queue.  Only one side!
                    if matches!(moved[k], Moved::None) {
                        queues[to].insert(
                            k,
                            (graph.vertex_weights[k] - e_degrees[other] as i32) as f32,
                        );
                        moved[k] = Moved::Neg(to);
                    }
                }
            }

            m_ptr.push(m_ind);
        }

        // roll back computation
        for (m_ind, &high_gain) in m_ptr
            .iter()
            .zip(&swaps)
            .rev()
            .take((swaps.len() as i32 - min_cut_order - 1) as usize)
        {
            debug_two!("Rolling {}, {:?}", high_gain, m_ind);
            debug_two!("where: {:?}", boundary_info._where);
            let to = boundary_info._where[high_gain];
            let other = if to == 0 { 1 } else { 0 };

            boundary_info.partition_weights[2] += graph.vertex_weights[high_gain];
            boundary_info.partition_weights[to] -= graph.vertex_weights[high_gain];

            boundary_info._where[high_gain] = 2;

            boundary_info.boundary_ind.push(high_gain);
            boundary_info.boundary_ptr[high_gain] = Some(boundary_info.boundary_ind.len() - 1);

            let mut e_degrees = [0, 0];
            for &k in graph.graph.neighbors(high_gain) {
                if boundary_info._where[k] == 2 {
                    boundary_info.nr_info[k].as_mut().unwrap().e_degrees[to] -=
                        graph.vertex_weights[high_gain] as usize;
                } else {
                    e_degrees[boundary_info._where[k]] += graph.vertex_weights[k] as usize;
                }
            }
            boundary_info.nr_info[high_gain] = Some(crate::separator_refinement::NrInfo {
                e_degrees: e_degrees,
            });

            // push nodes out of the separator
            for &k in m_ind {
                boundary_info._where[k] = other;
                boundary_info.partition_weights[other] += graph.vertex_weights[k];
                boundary_info.partition_weights[2] -= graph.vertex_weights[k];

                boundary_info.delete(k);

                for &kk in graph.graph.neighbors(k) {
                    if boundary_info._where[kk] == 2 {
                        boundary_info.nr_info[kk].as_mut().unwrap().e_degrees[other] +=
                            graph.vertex_weights[k] as usize;
                    }
                }
            }
        }

        min_cut_result = Some(min_cut);

        if min_cut_order == -1 || min_cut >= init_cut {
            break;
        }
    }

    debug_two!("Returning {:?}", boundary_info);
    (min_cut_result.unwrap(), boundary_info)
}

pub fn two_way_node_refine_one_sided<RNG>(
    config: &Config,
    graph: &WeightedGraph,
    mut boundary_info: BoundaryInfo,
    n_iterations: i32,
    graph_is_compressed: bool,
    rng: &mut RNG,
) -> (i32, BoundaryInfo)
where
    RNG: RangeRng,
{
    debug_one!("CALLED two_way_node_refine_one_sided");
    debug_one!("{:?}", graph);
    debug_one!("where: {:?}", boundary_info._where);
    debug_one!("partition_weights: {:?}", boundary_info.partition_weights);
    debug_one!("boundary_ind: {:?}", boundary_info.boundary_ind);
    debug_one!("boundary_ptr: {:?}", boundary_info.boundary_ptr);
    debug_one!("n_iterations: {}", n_iterations);
    let mut queue = PriorityQueue::create(graph.graph.n_vertices());

    let mult = 0.5 * config.ub_factors[0];
    let bad_max_partition_weight =
        (mult * (boundary_info.partition_weights.iter().sum::<i32>() as f32)) as usize;

    let (mut to, mut other) =
        if boundary_info.partition_weights[0] < boundary_info.partition_weights[1] {
            (1, 0)
        } else {
            (0, 1)
        };

    let mut min_cut_result = None;

    for pass in 0..2 * n_iterations {
        // the 2 * niter is for the two sides
        std::mem::swap(&mut to, &mut other);
        debug_one!("NEW PASS {} ({}, {})", pass, to, other);

        debug_one!("pre reset: {:?}", queue);

        queue.reset();

        debug_one!("after reset: {:?}", queue);

        let mut min_cut_order = None;
        let init_cut = boundary_info.partition_weights[2];
        let mut min_cut = boundary_info.partition_weights[2];
        let n_boundary = boundary_info.boundary_ind.len();

        // use the swaps array in place of the traditional perm array to save memory
        // NOTE(aaron): well fuck that
        let perm =
            crate::random::permutation(n_boundary, n_boundary, crate::random::Mode::Identity, rng);

        debug_one!("perm: {:?}", perm);

        for ii in 0..n_boundary {
            let i = boundary_info.boundary_ind[perm[ii]];

            debug_one!("pre insert: {:?}", queue);
            debug_one!(
                "i: {}, vwgt: {}, other: {}, deg: {}",
                i,
                graph.vertex_weights[i],
                other,
                boundary_info.nr_info[i].as_ref().unwrap().e_degrees[other]
            );

            queue.insert(
                i,
                (graph.vertex_weights[i]
                    - boundary_info.nr_info[i].as_ref().unwrap().e_degrees[other] as i32)
                    as f32,
            );

            debug_one!("post insert: {:?}", queue);
        }

        let limit = if graph_is_compressed {
            std::cmp::min(5 * n_boundary, 500)
        } else {
            std::cmp::min(3 * n_boundary, 300)
        };

        // get into the FM loop
        let mut m_ptr = Vec::new();
        let mut min_diff =
            (boundary_info.partition_weights[0] - boundary_info.partition_weights[1]).abs();
        let mut swaps = Vec::new();

        for n_swaps in 0..graph.graph.n_vertices() {
            debug_one!("SWAP {}", n_swaps);
            debug_one!("where: {:?}", boundary_info._where);
            debug_one!("partition_weights: {:?}", boundary_info.partition_weights);
            debug_one!("boundary_ind: {:?}", boundary_info.boundary_ind);
            debug_one!("boundary_ptr: {:?}", boundary_info.boundary_ptr);
            debug_one!("{:?}", queue);
            debug_one!("to: {}, other: {}", to, other);
            debug_one!("swaps: {:?}", swaps);
            debug_one!("min_cut_order: {:?}", min_cut_order);
            debug_one!("min_cut: {}", min_cut);
            debug_one!("m_ptr: {:?}", m_ptr);
            debug_one!("min_diff: {}", min_diff);

            let mut m_ind = Vec::new();

            let maybe_high_gain = queue.pop();
            if maybe_high_gain.is_none() {
                break;
            }

            let high_gain = match maybe_high_gain {
                Some(high_gain) => high_gain,
                None => panic!(),
            };

            // the following check is to ensure we break out if there is a possibility of
            // over-running the m_ind array
            // NOTE(aaron): Probably not necessary with Vec?
            if m_ptr.iter().map(|l: &Vec<usize>| l.len()).sum::<usize>()
                + m_ind.len()
                + graph.graph.degree(high_gain)
                >= 2 * graph.graph.n_vertices() - 1
            {
                panic!();
            }

            debug_one!(
                "pwgt: {}, vwgt: {}, bmpw: {}",
                boundary_info.partition_weights[to],
                graph.vertex_weights[high_gain],
                bad_max_partition_weight
            );
            if boundary_info.partition_weights[to] + graph.vertex_weights[high_gain]
                > bad_max_partition_weight as i32
            {
                debug_one!("Break 469");
                break; // no point going any further. balance will be bad
            }

            boundary_info.partition_weights[2] -= graph.vertex_weights[high_gain]
                - boundary_info.nr_info[high_gain].as_ref().unwrap().e_degrees[other] as i32;

            let new_diff = (boundary_info.partition_weights[to] + graph.vertex_weights[high_gain]
                - (boundary_info.partition_weights[other]
                    - boundary_info.nr_info[high_gain].as_ref().unwrap().e_degrees[other] as i32))
                .abs();

            if boundary_info.partition_weights[2] < min_cut
                || (boundary_info.partition_weights[2] == min_cut && new_diff < min_diff)
            {
                min_cut = boundary_info.partition_weights[2];
                min_cut_order = Some(n_swaps);
                min_diff = new_diff;
            } else if n_swaps as i32
                - match min_cut_order {
                    Some(o) => o as i32,
                    None => -1,
                }
                > 3 * limit as i32
                || (n_swaps as i32
                    - match min_cut_order {
                        Some(o) => o as i32,
                        None => -1,
                    }
                    > limit as i32
                    && boundary_info.partition_weights[2] as f32 > 1.10 * min_cut as f32)
            {
                boundary_info.partition_weights[2] += graph.vertex_weights[high_gain]
                    - boundary_info.nr_info[high_gain].as_ref().unwrap().e_degrees[other] as i32;
                break; // no further improvement, break out
            }

            debug_one!("high_gain: {}", high_gain);
            boundary_info.delete(high_gain);

            boundary_info.partition_weights[to] += graph.vertex_weights[high_gain];
            boundary_info._where[high_gain] = to;
            swaps.push(high_gain);

            // update the degrees of the affected nodes
            for &k in graph.graph.neighbors(high_gain) {
                debug_one!("Neighbor {}", k);
                if boundary_info._where[k] == 2 {
                    debug_one!("Branch 1");
                    // for the in-separator vertices modify their edegree[to]
                    boundary_info.nr_info[k].as_mut().unwrap().e_degrees[to] +=
                        graph.vertex_weights[high_gain] as usize;
                } else if boundary_info._where[k] == other {
                    debug_one!("Branch 2");
                    // this vertex is pulled into the separator
                    debug_one!("Inserting k {}", k);
                    boundary_info.insert(k);

                    m_ind.push(k); // keep track for rollback
                    boundary_info._where[k] = 2;
                    boundary_info.partition_weights[other] -= graph.vertex_weights[k];

                    let mut e_degrees = [0, 0];
                    for &kk in graph.graph.neighbors(k) {
                        if boundary_info._where[kk] != 2 {
                            e_degrees[boundary_info._where[kk]] +=
                                graph.vertex_weights[kk] as usize;
                        } else {
                            boundary_info.nr_info[kk].as_mut().unwrap().e_degrees[other] -=
                                graph.vertex_weights[k] as usize;

                            // since the moves are one-sided this vertex has not been moved yet
                            debug_one!("pre update: {:?}", queue);
                            queue.update(
                                kk,
                                (graph.vertex_weights[kk]
                                    - boundary_info.nr_info[kk].as_ref().unwrap().e_degrees[other]
                                        as i32) as f32,
                            );
                            debug_one!("post update: {:?}", queue);
                        }
                    }
                    boundary_info.nr_info[k] = Some(crate::separator_refinement::NrInfo {
                        e_degrees: e_degrees,
                    });

                    // insert the new vertex into the priority queue.  Safe due to one-sided moves
                    debug_one!("pre insert 2: {:?}", queue);
                    queue.insert(
                        k,
                        (graph.vertex_weights[k] - e_degrees[other] as i32) as f32,
                    );
                    debug_one!("post insert 2: {:?}", queue);
                }
            }

            m_ptr.push(m_ind);
        }

        // roll back computation
        for (m_ind, &high_gain) in
            m_ptr
                .iter()
                .zip(&swaps)
                .rev()
                .take(if let Some(n) = min_cut_order {
                    swaps.len() - n - 1
                } else {
                    swaps.len()
                })
        {
            boundary_info.partition_weights[2] += graph.vertex_weights[high_gain];
            boundary_info.partition_weights[to] -= graph.vertex_weights[high_gain];

            boundary_info._where[high_gain] = 2;

            boundary_info.boundary_ind.push(high_gain);
            boundary_info.boundary_ptr[high_gain] = Some(boundary_info.boundary_ind.len() - 1);

            let mut e_degrees = [0, 0];
            for &k in graph.graph.neighbors(high_gain) {
                if boundary_info._where[k] == 2 {
                    boundary_info.nr_info[k].as_mut().unwrap().e_degrees[to] -=
                        graph.vertex_weights[high_gain] as usize;
                } else {
                    e_degrees[boundary_info._where[k]] += graph.vertex_weights[k] as usize;
                }
            }
            boundary_info.nr_info[high_gain] = Some(crate::separator_refinement::NrInfo {
                e_degrees: e_degrees,
            });

            // push nodes out of the separator
            for &k in m_ind {
                boundary_info._where[k] = other;
                boundary_info.partition_weights[other] += graph.vertex_weights[k];
                boundary_info.partition_weights[2] -= graph.vertex_weights[k];

                boundary_info.delete(k);

                for &kk in graph.graph.neighbors(k) {
                    if boundary_info._where[kk] == 2 {
                        boundary_info.nr_info[kk].as_mut().unwrap().e_degrees[other] +=
                            graph.vertex_weights[k] as usize;
                    }
                }
            }
        }

        min_cut_result = Some(min_cut);

        if pass % 2 == 1 && (min_cut_order == None || min_cut >= init_cut) {
            break;
        }
    }

    debug_one!("EXITED two_way_node_refine_one_sided");

    (min_cut_result.unwrap(), boundary_info)
}

pub fn two_way_node_balance<RNG>(
    config: &Config,
    graph: &WeightedGraph,
    mut boundary_info: BoundaryInfo,
    total_vertex_weights: i32,
    rng: &mut RNG,
) -> BoundaryInfo
where
    RNG: RangeRng,
{
    debug_bal!("CALLED two_way_node_balance");

    debug_one!("{:?}", graph);
    debug_one!("where: {:?}", boundary_info._where);
    debug_one!("partition_weights: {:?}", boundary_info.partition_weights);
    debug_one!("boundary_ind: {:?}", boundary_info.boundary_ind);
    debug_one!("boundary_ptr: {:?}", boundary_info.boundary_ptr);

    let mult = 0.5 * config.ub_factors[0];

    let bad_max_partition_weight = (mult
        * (boundary_info.partition_weights[0] + boundary_info.partition_weights[1]) as f32)
        as i32;

    // wtf
    if std::cmp::max(
        boundary_info.partition_weights[0],
        boundary_info.partition_weights[1],
    ) < bad_max_partition_weight
    {
        debug_bal!("EARLY EXITED two_way_node_balance");
        return boundary_info;
    }

    if (boundary_info.partition_weights[0] - boundary_info.partition_weights[1]).abs()
        < 3 * total_vertex_weights / graph.graph.n_vertices() as i32
    {
        debug_bal!("EARLY EXITED 2 two_way_node_balance");
        return boundary_info;
    }

    let (to, other) = if boundary_info.partition_weights[0] < boundary_info.partition_weights[1] {
        (0, 1)
    } else {
        (1, 0)
    };

    let mut queue = PriorityQueue::create(graph.graph.n_vertices());
    let mut moved = vec![false; graph.graph.n_vertices()];

    let perm = crate::random::permutation(
        boundary_info.boundary_ind.len(),
        boundary_info.boundary_ind.len(),
        crate::random::Mode::Identity,
        rng,
    );

    for i in perm
        .iter()
        .map(|perm_ii| boundary_info.boundary_ind[*perm_ii])
    {
        queue.insert(
            i,
            (graph.vertex_weights[i]
                - boundary_info.nr_info[i].as_ref().unwrap().e_degrees[other] as i32)
                as f32,
        );
        debug_bal!(
            "Inserting {} {}",
            i,
            graph.vertex_weights[i]
                - boundary_info.nr_info[i].as_ref().unwrap().e_degrees[other] as i32
        );
        debug_bal!("{:?}", queue);
    }

    // Get into the FM loop
    for n_swaps in 0..graph.graph.n_vertices() {
        debug_bal!("SWAP {}", n_swaps);
        debug_bal!("where: {:?}", boundary_info._where);
        debug_bal!("partition_weights: {:?}", boundary_info.partition_weights);
        debug_bal!("boundary_ind: {:?}", boundary_info.boundary_ind);
        debug_bal!("boundary_ptr: {:?}", boundary_info.boundary_ptr);
        debug_bal!("{:?}", queue);
        debug_bal!("to: {}, other: {}", to, other);

        let maybe_high_gain = queue.pop();
        if maybe_high_gain.is_none() {
            break;
        }

        let high_gain = match maybe_high_gain {
            Some(high_gain) => high_gain,
            None => panic!(),
        };

        moved[high_gain] = true;

        let gain = graph.vertex_weights[high_gain]
            - boundary_info.nr_info[high_gain].as_ref().unwrap().e_degrees[other] as i32;

        let bad_max_partition_weight = (mult
            * (boundary_info.partition_weights[0] + boundary_info.partition_weights[1]) as f32)
            as i32;

        // break if other is now underweight
        if boundary_info.partition_weights[to] > boundary_info.partition_weights[other] {
            break;
        }

        // break if balance is achieved and no +ve or zero gain
        if gain < 0 && boundary_info.partition_weights[other] < bad_max_partition_weight {
            break;
        }

        // skip this vertex if it will violate balance on the other side
        if boundary_info.partition_weights[to] + graph.vertex_weights[high_gain]
            > bad_max_partition_weight
        {
            continue;
        }

        boundary_info.partition_weights[2] -= gain;

        boundary_info.delete(high_gain);

        boundary_info.partition_weights[to] += graph.vertex_weights[high_gain];
        boundary_info._where[high_gain] = to;

        // update the degrees of the affected nodes
        for &k in graph.graph.neighbors(high_gain) {
            debug_bal!("k: {}", k);
            if boundary_info._where[k] == 2 {
                // for the in-separator vertices modify their edegree[to]
                boundary_info.nr_info[k].as_mut().unwrap().e_degrees[to] +=
                    graph.vertex_weights[high_gain] as usize;
            } else if boundary_info._where[k] == other {
                // this vertex is pulled into the separator
                boundary_info.boundary_ind.push(k);
                boundary_info.boundary_ptr[k] = Some(boundary_info.boundary_ind.len() - 1);

                boundary_info._where[k] = 2;
                boundary_info.partition_weights[other] -= graph.vertex_weights[k];

                let mut e_degrees: [usize; 2] = [0, 0];
                for &kk in graph.graph.neighbors(k) {
                    debug_bal!("kk: {}", kk);
                    if boundary_info._where[kk] != 2 {
                        e_degrees[boundary_info._where[kk]] += graph.vertex_weights[kk] as usize;
                    } else {
                        let old_gain = graph.vertex_weights[kk]
                            - boundary_info.nr_info[kk].as_ref().unwrap().e_degrees[other] as i32;

                        boundary_info.nr_info[kk].as_mut().unwrap().e_degrees[other] -=
                            graph.vertex_weights[k] as usize;

                        if !moved[kk] {
                            queue.update(kk, (old_gain + graph.vertex_weights[k]) as f32);
                        }
                    }
                }
                debug_bal!("nr_info: {}", boundary_info.nr_info.len());
                boundary_info.nr_info[k] = Some(crate::separator_refinement::NrInfo {
                    e_degrees: e_degrees,
                });

                // insert the new vertex into the priority queue
                queue.insert(
                    k,
                    (graph.vertex_weights[k] - e_degrees[other] as i32) as f32,
                );
            }
        }
    }

    let _min_cut = boundary_info.partition_weights[2];

    debug_bal!("EXITED two_way_node_balance");

    boundary_info
}
