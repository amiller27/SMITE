use crate::config::Config;
use crate::graph::WeightedGraph;
use crate::priority_queue::PriorityQueue;
use crate::separator_refinement::BoundaryInfo;

#[derive(Clone)]
enum Moved {
    None,
    Some(usize),
    Neg(usize),
}

pub fn two_way_node_refine_two_sided(
    config: &Config,
    graph: WeightedGraph,
    mut boundary_info: BoundaryInfo,
    n_iterations: i32,
) -> (i32, BoundaryInfo) {
    let mut swaps = Vec::new();

    let mult = 0.5 * config.ub_factors[0];
    let bad_max_partition_weight =
        (mult * (boundary_info.partition_weights.iter().sum::<i32>() as f32)) as usize;

    let mut min_cut_result;

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
        let perm = crate::random::permutation(
            n_boundary,
            n_boundary,
            crate::random::Mode::Identity,
            &mut rand::thread_rng(),
        );

        for ii in 0..n_boundary {
            let i = boundary_info.boundary_ind[perm[ii]];
            queues[0].insert(
                i,
                (graph.vertex_weights.as_ref().unwrap()[i]
                    - boundary_info.nr_info[i].e_degrees[1] as i32) as f32,
            );
            queues[1].insert(
                i,
                (graph.vertex_weights.as_ref().unwrap()[i]
                    - boundary_info.nr_info[i].e_degrees[0] as i32) as f32,
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
        for n_swaps in 0..graph.graph.n_vertices() {
            let mut m_ind = Vec::new();

            let maybe_u = [queues[0].peek(), queues[1].peek()];

            if maybe_u[0].is_some() && maybe_u[1].is_some() {
                let u = [maybe_u[0].unwrap(), maybe_u[1].unwrap()];

                let g = [
                    graph.vertex_weights.as_ref().unwrap()[u[0]]
                        - boundary_info.nr_info[u[0]].e_degrees[1] as i32,
                    graph.vertex_weights.as_ref().unwrap()[u[1]]
                        - boundary_info.nr_info[u[1]].e_degrees[0] as i32,
                ];

                to = if g[0] > g[1] {
                    0
                } else if g[0] < g[1] {
                    1
                } else {
                    (pass % 2) as usize
                };

                if boundary_info.partition_weights[to]
                    + graph.vertex_weights.as_ref().unwrap()[u[to]]
                    > bad_max_partition_weight as i32
                {
                    to = if to == 0 { 1 } else { 0 };
                }
            } else if maybe_u[0].is_none() && maybe_u[1].is_none() {
                break;
            } else if maybe_u[0].is_some()
                && boundary_info.partition_weights[0]
                    + graph.vertex_weights.as_ref().unwrap()[maybe_u[0].unwrap()]
                    <= bad_max_partition_weight as i32
            {
                to = 0;
            } else if maybe_u[1].is_some()
                && boundary_info.partition_weights[1]
                    + graph.vertex_weights.as_ref().unwrap()[maybe_u[1].unwrap()]
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

            boundary_info.partition_weights[2] -= graph.vertex_weights.as_ref().unwrap()[high_gain]
                - boundary_info.nr_info[high_gain].e_degrees[other] as i32;

            let new_diff = (boundary_info.partition_weights[to]
                + graph.vertex_weights.as_ref().unwrap()[high_gain]
                - (boundary_info.partition_weights[other]
                    - boundary_info.nr_info[high_gain].e_degrees[other] as i32))
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
                boundary_info.partition_weights[2] += graph.vertex_weights.as_ref().unwrap()
                    [high_gain]
                    - boundary_info.nr_info[high_gain].e_degrees[other] as i32;
                break; // no further improvement, break out
            }

            let index_to_update = boundary_info.boundary_ptr[high_gain];
            let value_to_move = boundary_info.boundary_ind.pop();
            boundary_info.boundary_ind[index_to_update.unwrap()] = value_to_move.unwrap();
            boundary_info.boundary_ptr[value_to_move.unwrap()] = index_to_update;
            boundary_info.boundary_ptr[high_gain] = None;

            boundary_info.partition_weights[to] +=
                graph.vertex_weights.as_ref().unwrap()[high_gain];
            boundary_info._where[high_gain] = to;
            moved[high_gain] = Moved::Some(n_swaps);
            swaps.push(high_gain);

            // update the degrees of the affected nodes
            for &k in graph.graph.neighbors(high_gain) {
                if boundary_info._where[k] == 2 {
                    // for the in-separator vertices modify their edegree[to]
                    let old_gain = graph.vertex_weights.as_ref().unwrap()[k]
                        - boundary_info.nr_info[k].e_degrees[to] as i32;
                    boundary_info.nr_info[k].e_degrees[to] +=
                        graph.vertex_weights.as_ref().unwrap()[high_gain] as usize; // eek

                    match moved[k] {
                        Moved::None => queues[other].update(
                            k,
                            (old_gain - graph.vertex_weights.as_ref().unwrap()[high_gain]) as f32,
                        ),
                        Moved::Neg(i) if i == other => queues[other].update(
                            k,
                            (old_gain - graph.vertex_weights.as_ref().unwrap()[high_gain]) as f32,
                        ),
                        _ => {}
                    }
                } else if boundary_info._where[k] == other {
                    // the vertex is pulled into the separator
                    boundary_info.boundary_ind.push(k);
                    boundary_info.boundary_ptr[k] = Some(boundary_info.boundary_ind.len() - 1);

                    m_ind.push(k); // keep track for rollback
                    boundary_info._where[k] = 2;
                    boundary_info.partition_weights[other] -=
                        graph.vertex_weights.as_ref().unwrap()[k];

                    let mut e_degrees = [0, 0];
                    for &kk in graph.graph.neighbors(k) {
                        if boundary_info._where[kk] != 2 {
                            e_degrees[boundary_info._where[kk]] +=
                                graph.vertex_weights.as_ref().unwrap()[kk] as usize;
                        } else {
                            boundary_info.nr_info[kk].e_degrees[other] -=
                                graph.vertex_weights.as_ref().unwrap()[k] as usize;
                            let old_gain = graph.vertex_weights.as_ref().unwrap()[kk]
                                - boundary_info.nr_info[kk].e_degrees[other] as i32;
                            match moved[k] {
                                Moved::None => queues[to].update(
                                    kk,
                                    (old_gain + graph.vertex_weights.as_ref().unwrap()[k]) as f32,
                                ),
                                Moved::Neg(i) if i == other => queues[to].update(
                                    kk,
                                    (old_gain + graph.vertex_weights.as_ref().unwrap()[k]) as f32,
                                ),
                                _ => {}
                            }
                        }
                    }
                    boundary_info.nr_info[k].e_degrees = e_degrees;

                    // insert the new vertex into the priority queue.  Only one side!
                    if matches!(moved[k], Moved::None) {
                        queues[to].insert(
                            k,
                            (graph.vertex_weights.as_ref().unwrap()[k] - e_degrees[other] as i32)
                                as f32,
                        );
                        moved[k] = Moved::Neg(to);
                    }
                }
            }

            m_ptr.push(m_ind);
        }

        // roll back computation
        for (&m_ind, &high_gain) in m_ptr
            .iter()
            .zip(swaps.iter())
            .rev()
            .take((swaps.len() as i32 - min_cut_order - 1) as usize)
        {
            let to = boundary_info._where[high_gain];
            let other = if to == 0 { 1 } else { 0 };

            boundary_info.partition_weights[2] += graph.vertex_weights.as_ref().unwrap()[high_gain];
            boundary_info.partition_weights[to] -=
                graph.vertex_weights.as_ref().unwrap()[high_gain];

            boundary_info._where[high_gain] = 2;

            boundary_info.boundary_ind.push(high_gain);
            boundary_info.boundary_ptr[high_gain] = Some(boundary_info.boundary_ind.len() - 1);

            let mut e_degrees = [0, 0];
            for &k in graph.graph.neighbors(high_gain) {
                if boundary_info._where[k] == 2 {
                    boundary_info.nr_info[k].e_degrees[to] -=
                        graph.vertex_weights.as_ref().unwrap()[high_gain] as usize;
                } else {
                    e_degrees[boundary_info._where[k]] +=
                        graph.vertex_weights.as_ref().unwrap()[k] as usize;
                }
            }
            boundary_info.nr_info[high_gain].e_degrees = e_degrees;

            // push nodes out of the separator
            for k in m_ind {
                boundary_info._where[k] = other;
                boundary_info.partition_weights[other] += graph.vertex_weights.as_ref().unwrap()[k];
                boundary_info.partition_weights[2] -= graph.vertex_weights.as_ref().unwrap()[k];

                let index_to_update = boundary_info.boundary_ptr[k];
                let value_to_move = boundary_info.boundary_ind.pop();
                boundary_info.boundary_ind[index_to_update.unwrap()] = value_to_move.unwrap();
                boundary_info.boundary_ptr[value_to_move.unwrap()] = index_to_update;
                boundary_info.boundary_ptr[k] = None;

                for &kk in graph.graph.neighbors(k) {
                    if boundary_info._where[kk] == 2 {
                        boundary_info.nr_info[kk].e_degrees[other] +=
                            graph.vertex_weights.as_ref().unwrap()[k] as usize;
                    }
                }
            }
        }

        min_cut_result = min_cut;

        if min_cut_order == -1 || min_cut >= init_cut {
            break;
        }
    }

    (min_cut_result, boundary_info)
}

pub fn two_way_node_refine_one_sided(
    config: &Config,
    graph: WeightedGraph,
    mut boundary_info: BoundaryInfo,
    n_iterations: i32,
) -> (i32, BoundaryInfo) {
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

    let mut min_cut_result;

    let mut swaps = Vec::new();

    for pass in 0..2 * n_iterations {
        // the 2 * niter is fo the two sides
        std::mem::swap(&mut to, &mut other);

        queue.reset();

        let mut min_cut_order = None;
        let init_cut = boundary_info.partition_weights[2];
        let mut min_cut = boundary_info.partition_weights[2];
        let n_boundary = boundary_info.boundary_ind.len();

        // use the swaps array in place of the traditional perm array to save memory
        // NOTE(aaron): well fuck that
        let perm = crate::random::permutation(
            n_boundary,
            n_boundary,
            crate::random::Mode::Identity,
            &mut rand::thread_rng(),
        );

        for ii in 0..n_boundary {
            let i = boundary_info.boundary_ind[perm[ii]];
            queue.insert(
                i,
                (graph.vertex_weights.as_ref().unwrap()[i]
                    - boundary_info.nr_info[i].e_degrees[other] as i32) as f32,
            );
        }

        let limit = if config.compress_before_ordering {
            std::cmp::min(5 * n_boundary, 500)
        } else {
            std::cmp::min(3 * n_boundary, 300)
        };

        // get into the FM loop
        let mut m_ptr = Vec::new();
        let mut min_diff =
            (boundary_info.partition_weights[0] - boundary_info.partition_weights[1]).abs();
        for n_swaps in 0..graph.graph.n_vertices() {
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

            if boundary_info.partition_weights[to]
                + graph.vertex_weights.as_ref().unwrap()[high_gain]
                > bad_max_partition_weight as i32
            {
                break; // no point going any further. balance will be bad
            }

            boundary_info.partition_weights[2] -= graph.vertex_weights.as_ref().unwrap()[high_gain]
                - boundary_info.nr_info[high_gain].e_degrees[other] as i32;

            let new_diff = (boundary_info.partition_weights[to]
                + graph.vertex_weights.as_ref().unwrap()[high_gain]
                - (boundary_info.partition_weights[other]
                    - boundary_info.nr_info[high_gain].e_degrees[other] as i32))
                .abs();

            if boundary_info.partition_weights[2] < min_cut
                || (boundary_info.partition_weights[2] == min_cut && new_diff < min_diff)
            {
                min_cut = boundary_info.partition_weights[2];
                min_cut_order = Some(n_swaps);
                min_diff = new_diff;
            } else if n_swaps - min_cut_order.unwrap() > 3 * limit
                || (n_swaps - min_cut_order.unwrap() > limit
                    && boundary_info.partition_weights[2] as f32 > 1.10 * min_cut as f32)
            {
                boundary_info.partition_weights[2] += graph.vertex_weights.as_ref().unwrap()
                    [high_gain]
                    - boundary_info.nr_info[high_gain].e_degrees[other] as i32;
                break; // no further improvement, break out
            }

            let index_to_update = boundary_info.boundary_ptr[high_gain];
            let value_to_move = boundary_info.boundary_ind.pop();
            boundary_info.boundary_ind[index_to_update.unwrap()] = value_to_move.unwrap();
            boundary_info.boundary_ptr[value_to_move.unwrap()] = index_to_update;
            boundary_info.boundary_ptr[high_gain] = None;

            boundary_info.partition_weights[to] +=
                graph.vertex_weights.as_ref().unwrap()[high_gain];
            boundary_info._where[high_gain] += to;
            swaps.push(high_gain);

            // update the degrees of the affected nodes
            for k_ptr in graph.graph.neighbors(high_gain) {
                let k = *k_ptr;

                if boundary_info._where[k] == 2 {
                    // for the in-separator vertices modify their edegree[to]
                    boundary_info.nr_info[k].e_degrees[to] +=
                        graph.vertex_weights.as_ref().unwrap()[high_gain] as usize;
                } else if boundary_info._where[k] == other {
                    // this vertex is pulled into the separator
                    boundary_info.boundary_ind.push(k);
                    boundary_info.boundary_ptr[k] =
                        Some(boundary_info.boundary_ind[boundary_info.boundary_ind.len() - 1]);

                    m_ind.push(k); // keep track for rollback
                    boundary_info._where[k] = 2;
                    boundary_info.partition_weights[other] -=
                        graph.vertex_weights.as_ref().unwrap()[k];

                    let mut e_degrees = [0, 0];
                    for kk_ptr in graph.graph.neighbors(k) {
                        let kk = *kk_ptr;
                        if boundary_info._where[kk] != 2 {
                            e_degrees[boundary_info._where[kk]] +=
                                graph.vertex_weights.as_ref().unwrap()[kk] as usize;
                        } else {
                            boundary_info.nr_info[kk].e_degrees[other] -=
                                graph.vertex_weights.as_ref().unwrap()[k] as usize;
                        }

                        // since the moves are one-sided this vertex has not been moved yet
                        queue.update(
                            kk,
                            (graph.vertex_weights.as_ref().unwrap()[kk]
                                - boundary_info.nr_info[kk].e_degrees[other] as i32)
                                as f32,
                        );
                    }
                    boundary_info.nr_info[k].e_degrees = e_degrees;

                    // insert the new vertex into the priority queue.  Safe due to one-sided moves
                    queue.insert(
                        k,
                        (graph.vertex_weights.as_ref().unwrap()[k] - e_degrees[other] as i32)
                            as f32,
                    );
                }
            }

            m_ptr.push(m_ind);
        }

        // roll back computation
        for (&m_ind, &high_gain) in
            m_ptr
                .iter()
                .zip(swaps.iter())
                .rev()
                .take(if let Some(n) = min_cut_order {
                    swaps.len() - n - 1
                } else {
                    swaps.len()
                })
        {
            boundary_info.partition_weights[2] += graph.vertex_weights.as_ref().unwrap()[high_gain];
            boundary_info.partition_weights[to] -=
                graph.vertex_weights.as_ref().unwrap()[high_gain];

            boundary_info._where[high_gain] = 2;

            boundary_info.boundary_ind.push(high_gain);
            boundary_info.boundary_ptr[high_gain] = Some(boundary_info.boundary_ind.len() - 1);

            let mut e_degrees = [0, 0];
            for &k in graph.graph.neighbors(high_gain) {
                if boundary_info._where[k] == 2 {
                    boundary_info.nr_info[k].e_degrees[to] -=
                        graph.vertex_weights.as_ref().unwrap()[high_gain] as usize;
                } else {
                    e_degrees[boundary_info._where[k]] +=
                        graph.vertex_weights.as_ref().unwrap()[k] as usize;
                }
            }
            boundary_info.nr_info[high_gain].e_degrees = e_degrees;

            // push nodes out of the separator
            for k in m_ind {
                boundary_info._where[k] = other;
                boundary_info.partition_weights[other] += graph.vertex_weights.as_ref().unwrap()[k];
                boundary_info.partition_weights[2] -= graph.vertex_weights.as_ref().unwrap()[k];

                let index_to_update = boundary_info.boundary_ptr[k];
                let value_to_move = boundary_info.boundary_ind.pop();
                boundary_info.boundary_ind[index_to_update.unwrap()] = value_to_move.unwrap();
                boundary_info.boundary_ptr[value_to_move.unwrap()] = index_to_update;
                boundary_info.boundary_ptr[k] = None;

                for &kk in graph.graph.neighbors(k) {
                    if boundary_info._where[kk] == 2 {
                        boundary_info.nr_info[kk].e_degrees[other] +=
                            graph.vertex_weights.as_ref().unwrap()[k] as usize;
                    }
                }
            }
        }

        min_cut_result = min_cut;

        if pass % 2 == 1 && (min_cut_order == None || min_cut >= init_cut) {
            break;
        }
    }

    (min_cut_result, boundary_info)
}

pub fn two_way_node_balance(
    config: &Config,
    graph: WeightedGraph,
    mut boundary_info: BoundaryInfo,
    total_vertex_weights: i32,
) -> BoundaryInfo {
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
        return boundary_info;
    }

    if (boundary_info.partition_weights[0] - boundary_info.partition_weights[1]).abs()
        < 3 * total_vertex_weights / graph.graph.n_vertices() as i32
    {
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
        &mut rand::thread_rng(),
    );

    for i in perm
        .iter()
        .map(|perm_ii| boundary_info.boundary_ind[*perm_ii])
    {
        queue.insert(
            i,
            (graph.vertex_weights.as_ref().unwrap()[i]
                - boundary_info.nr_info[i].e_degrees[other] as i32) as f32,
        );
    }

    // Get into the FM loop
    for _n_swaps in 0..graph.graph.n_vertices() {
        let maybe_high_gain = queue.pop();
        if maybe_high_gain.is_none() {
            break;
        }

        let high_gain = match maybe_high_gain {
            Some(high_gain) => high_gain,
            None => panic!(),
        };

        moved[high_gain] = true;

        let gain = graph.vertex_weights.as_ref().unwrap()[high_gain]
            - boundary_info.nr_info[high_gain].e_degrees[other] as i32;

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
        if boundary_info.partition_weights[to] + graph.vertex_weights.as_ref().unwrap()[high_gain]
            > bad_max_partition_weight
        {
            continue;
        }

        boundary_info.partition_weights[2] -= gain;

        let index_to_update = boundary_info.boundary_ptr[high_gain];
        let value_to_move = boundary_info.boundary_ind.pop();
        boundary_info.boundary_ind[index_to_update.unwrap()] = value_to_move.unwrap();
        boundary_info.boundary_ptr[value_to_move.unwrap()] = index_to_update;
        boundary_info.boundary_ptr[high_gain] = None;

        boundary_info.partition_weights[to] += graph.vertex_weights.as_ref().unwrap()[high_gain];
        boundary_info._where[high_gain] = to;

        // update the degrees of the affected nodes
        for k in graph.graph.neighbors(high_gain) {
            if boundary_info._where[*k] == 2 {
                // for the in-separator vertices modify their edegree[to]
                boundary_info.nr_info[*k].e_degrees[to] +=
                    graph.vertex_weights.as_ref().unwrap()[high_gain] as usize;
            } else if boundary_info._where[*k] == other {
                // this vertex is pulled into the separator
                boundary_info.boundary_ind.push(*k);
                boundary_info.boundary_ptr[*k] = Some(boundary_info.boundary_ind.len() - 1);

                boundary_info._where[*k] = 2;
                boundary_info.partition_weights[other] -=
                    graph.vertex_weights.as_ref().unwrap()[*k];

                let mut e_degrees: [usize; 2] = [0, 0];
                for kk in graph.graph.neighbors(*k) {
                    if boundary_info._where[*kk] != 2 {
                        e_degrees[boundary_info._where[*kk]] +=
                            graph.vertex_weights.as_ref().unwrap()[*kk] as usize;
                    } else {
                        let old_gain = graph.vertex_weights.as_ref().unwrap()[*kk]
                            - boundary_info.nr_info[*kk].e_degrees[other] as i32;

                        boundary_info.nr_info[*kk].e_degrees[other] -=
                            graph.vertex_weights.as_ref().unwrap()[*k] as usize;

                        if !moved[*kk] {
                            queue.update(
                                *kk,
                                (old_gain + graph.vertex_weights.as_ref().unwrap()[*k]) as f32,
                            );
                        }
                    }
                }
                boundary_info.nr_info[*k].e_degrees = e_degrees;

                // insert the new vertex into the priority queue
                queue.insert(
                    *k,
                    (graph.vertex_weights.as_ref().unwrap()[*k] - e_degrees[other] as i32) as f32,
                );
            }
        }
    }

    let _min_cut = boundary_info.partition_weights[2];

    boundary_info
}
