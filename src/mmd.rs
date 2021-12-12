use crate::config::Index;
use crate::graph::{Graph, MutableGraph};

pub struct MMDResult {
    pub iperm: Vec<usize>,
    pub perm: Vec<usize>,
}

const DEBUG_MMD: bool = false;

macro_rules! debug {
    ($($x: expr),*) => {
        if DEBUG_MMD{
            println!($($x,)*);
        }
    };
}

pub fn gen_mmd(full_graph: Graph) -> MMDResult {
    const DELTA: usize = 1;

    debug!("HERE BE DRAGONS");
    debug!("THIS IS NOT A PLACE OF HONOR");
    debug!("{:?}", full_graph);

    if full_graph.n_vertices() == 0 {
        panic!();
    }

    let mut graph = MutableGraph::from_graph(full_graph);

    // Never read, REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    // let mut nnz_upper_bound = 0;

    let (mut head, mut inverse_perm, mut perm, mut qsize, mut _list, mut marker) =
        initialization(&graph);

    let mut n_ordered_nodes = 0;

    // eliminate all isolated nodes
    let mut next_minimum_degree_node = head[0];
    while let ForwardPtr::Next(minimum_degree_node) = next_minimum_degree_node {
        debug!("mini: {}", minimum_degree_node);
        next_minimum_degree_node = inverse_perm[minimum_degree_node];
        marker[minimum_degree_node] = Marker::Zero;
        inverse_perm[minimum_degree_node] = ForwardPtr::NextNeg(n_ordered_nodes);
        n_ordered_nodes += 1;
    }

    if n_ordered_nodes >= graph.n_vertices() {
        return numbering(graph.n_vertices(), inverse_perm, qsize);
    }

    let mut tag = 1;
    head[0] = ForwardPtr::None;
    let mut minimum_degree = 2;

    let mut i = 0;
    loop {
        debug!(
            "LOOP {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
            i
        );
        i += 1;
        debug!("{:?}", graph);
        debug!("head: {:?}", head);
        debug!("inverse_perm: {:?}", inverse_perm);
        debug!("perm: {:?}", perm);
        debug!("qsize: {:?}", qsize);
        debug!("marker: {:?}", marker);
        debug!("n_ordered_nodes: {:?}", n_ordered_nodes);
        debug!("next_minimum_degree_node: {:?}", next_minimum_degree_node);
        debug!("tag: {}", tag);
        debug!("minimum_degree: {}", minimum_degree);

        while matches!(head[minimum_degree], ForwardPtr::None) {
            minimum_degree += 1;
        }

        let minimum_degree_limit = minimum_degree + DELTA;
        let mut ehead = Vec::new();

        // n500
        loop {
            let mut maybe_minimum_degree_node = head[minimum_degree];

            // This can be simplified if DELTA == 1, which is the only way this is used in METIS
            let mut hit_limit = false;
            debug!("minimum degree start: {}", minimum_degree);
            while matches!(maybe_minimum_degree_node, ForwardPtr::None) {
                minimum_degree += 1;

                if minimum_degree > minimum_degree_limit {
                    hit_limit = true;
                    break;
                }

                maybe_minimum_degree_node = head[minimum_degree];
            }
            debug!("post loop: {}", minimum_degree);

            if hit_limit {
                break;
            }

            let minimum_degree_node = match maybe_minimum_degree_node {
                ForwardPtr::Next(i) => i,
                _ => panic!(),
            };

            // remove minimum_degree_node from the degree structure
            let next_minimum_degree_node = inverse_perm[minimum_degree_node];
            head[minimum_degree] = next_minimum_degree_node;
            if let ForwardPtr::Next(i) = next_minimum_degree_node {
                perm[i] = BackPtr::Degree(minimum_degree);
            }
            inverse_perm[minimum_degree_node] = ForwardPtr::NextNeg(n_ordered_nodes);
            // nnz_upper_bound += minimum_degree + qsize[minimum_degree_node] - 1;
            if n_ordered_nodes + qsize[minimum_degree_node] >= graph.n_vertices() {
                return numbering(graph.n_vertices(), inverse_perm, qsize);
            }

            // eliminate minimum degree node and perform quotient graph transformation.  Reset tag
            // value if necessary
            // Note: don't reset, the only use in METIS has maxint = max of Index type
            tag += 1;

            debug!("ENTER ELIMINATE");
            let eliminate_result = eliminate(
                minimum_degree_node,
                graph,
                head,
                inverse_perm,
                perm,
                qsize,
                marker,
                tag,
            );
            graph = eliminate_result.0;
            head = eliminate_result.1;
            inverse_perm = eliminate_result.2;
            perm = eliminate_result.3;
            qsize = eliminate_result.4;
            marker = eliminate_result.5;
            debug!("EXIT ELIMINATE");

            n_ordered_nodes += qsize[minimum_degree_node];
            ehead.insert(0, minimum_degree_node);

            // Impossible if DELTA is a usize
            // if DELTA < 0 {
            //     break;
            // }
        }

        // n900
        // update degrees of the nodes involved in the minimum degree nodes elimination
        if n_ordered_nodes > graph.n_vertices() {
            break;
        }

        debug!("ENTER UPDATE");
        let update_result = update(
            ehead,
            &graph,
            DELTA,
            minimum_degree,
            head,
            inverse_perm,
            perm,
            qsize,
            marker,
            tag,
        );
        minimum_degree = update_result.0;
        head = update_result.1;
        inverse_perm = update_result.2;
        perm = update_result.3;
        qsize = update_result.4;
        marker = update_result.5;
        tag = update_result.6;
        debug!("EXIT UPDATE");
    }

    return numbering(graph.n_vertices(), inverse_perm, qsize);
}

#[derive(Clone, Copy, Debug)]
enum BackPtr {
    Previous(usize),
    Degree(usize),
    NegMaxInt,
    None,
}

#[derive(Clone, PartialEq, Copy, Debug)]
enum ForwardPtr {
    Next(usize),
    NextNeg(usize),
    None,
}

#[derive(Clone, Debug)]
enum Marker {
    Maxint,
    Zero,
    Tag(usize),
}

fn initialization(
    graph: &MutableGraph,
) -> (
    Vec<ForwardPtr>,
    Vec<ForwardPtr>,
    Vec<BackPtr>,
    Vec<usize>,
    Vec<Index>,
    Vec<Marker>,
) {
    let qsize = vec![1; graph.n_vertices()];
    let list = vec![-1; graph.n_vertices()];
    let marker = vec![Marker::Zero; graph.n_vertices()];

    /* initialize the degree doubly linked lists */
    let mut head = vec![ForwardPtr::None; graph.n_vertices()];
    let mut forward = vec![ForwardPtr::None; graph.n_vertices()];
    let mut backward = vec![BackPtr::None; graph.n_vertices()];
    for node in 0..graph.n_vertices() {
        let degree = graph.degree(node); // off by one?

        // Insert node into the front of the linked list for its degree
        let front_node = head[degree];
        forward[node] = match front_node {
            ForwardPtr::Next(n) => ForwardPtr::Next(n),
            ForwardPtr::None => ForwardPtr::None,
            _ => panic!(),
        };
        head[degree] = ForwardPtr::Next(node);
        if let ForwardPtr::Next(front_node_i) = front_node {
            backward[front_node_i] = BackPtr::Previous(node);
        }
        backward[node] = BackPtr::Degree(degree);
    }

    (head, forward, backward, qsize, list, marker)
}

fn eliminate(
    minimum_degree_node: usize,
    mut graph: MutableGraph,
    mut head: Vec<ForwardPtr>,
    mut forward: Vec<ForwardPtr>,
    mut backward: Vec<BackPtr>,
    mut qsize: Vec<usize>,
    mut marker: Vec<Marker>,
    tag: usize,
) -> (
    MutableGraph,
    Vec<ForwardPtr>,
    Vec<ForwardPtr>,
    Vec<BackPtr>,
    Vec<usize>,
    Vec<Marker>,
) {
    debug!("---------- AT START OF ELIMINATE -------------");
    debug!("minimum_degree_node: {}", minimum_degree_node);
    debug!("{:?}", graph);
    debug!("head: {:?}", head);
    debug!("forward: {:?}", forward);
    debug!("backward: {:?}", backward);
    debug!("qsize: {:?}", qsize);
    debug!("marker: {:?}", marker);
    debug!("tag: {}", tag);
    debug!("------------ END START OF ELIMINATE --------------");

    // find the reachable set of minimum_degree_node and place it in the data structure
    debug!(
        "215 Setting marker [{}] to tag {:?}",
        minimum_degree_node,
        Marker::Tag(tag)
    );
    marker[minimum_degree_node] = Marker::Tag(tag);

    // element points to the beginning of the list of eliminated neighbors of minimum_degree_node,
    // and rloc gives the storage location for the next reachable node
    let mut eliminated_neighbors = Vec::<usize>::new();
    let mut neighbors_to_keep = Vec::<usize>::new();
    debug!("278 Finding neighbors of {}", minimum_degree_node);
    for &neighbor in graph.neighbors(minimum_degree_node) {
        // I think this check in mmd.c is "are we at the end of the neighbor list", which is unnecessary here
        // if neighbor == 0 {
        //     // this condition is wrong
        //     break;
        // }

        if matches!(marker[neighbor], Marker::Zero)
            || matches!(marker[neighbor], Marker::Tag(t) if t < tag)
        {
            debug!(
                "230 Setting marker [{}] to tag {:?}",
                neighbor,
                Marker::Tag(tag)
            );
            marker[neighbor] = Marker::Tag(tag);
            if matches!(forward[neighbor], ForwardPtr::NextNeg(_)) {
                eliminated_neighbors.push(neighbor);
            } else {
                debug!("297 Keeping neighbor {}", neighbor);
                neighbors_to_keep.push(neighbor)
            }
        }
    }

    // merge with reachable nodes from generalized elements
    while !eliminated_neighbors.is_empty() {
        let link = eliminated_neighbors.pop().unwrap();
        // n400
        // TODO: Rust correctly complains that this might modify the same data structure, but I
        // _think_ this is actually safe
        for node in graph.neighbors_copy(link) {
            if (matches!(marker[node], Marker::Zero)
                || matches!(marker[node], Marker::Tag(t) if t < tag))
                && matches!(forward[node], ForwardPtr::Next(_) | ForwardPtr::None)
            {
                debug!(
                    "256 Setting marker [{}] to tag {:?}",
                    node,
                    Marker::Tag(tag)
                );
                marker[node] = Marker::Tag(tag);
                debug!("320 Keeping neighbor {}", node);
                neighbors_to_keep.push(node);
            }
        }
    }

    graph.adjacency[minimum_degree_node] = neighbors_to_keep;

    // for each node in the reachable set, do the following
    let link = minimum_degree_node;
    for rnode in graph.neighbors_copy(link) {
        // rnode is in the degree list structure
        let previous_node = backward[rnode];
        if !matches!(previous_node, BackPtr::None | BackPtr::NegMaxInt) {
            // this condition is wrong
            // then remove rnode from the structure
            let next_node = forward[rnode];
            if let ForwardPtr::Next(n) = next_node {
                backward[n] = previous_node;
            }

            if let BackPtr::Previous(previous_i) = previous_node {
                forward[previous_i] = next_node;
            }

            if let BackPtr::Degree(d) = previous_node {
                head[d] = next_node;
            }
        }

        // purge inactive quotient neighbors of rnode
        let mut neighbors_to_keep = Vec::new();
        for &neighbor in graph.neighbors(rnode) {
            if matches!(marker[neighbor], Marker::Zero)
                || matches!(marker[neighbor], Marker::Tag(t) if t < tag)
            {
                neighbors_to_keep.push(neighbor);
            }
        }
        graph.adjacency[rnode] = neighbors_to_keep;

        // no active neighbors after the purging
        if graph.degree(rnode) == 0 {
            // merge rnode with minimum_degree_node
            qsize[minimum_degree_node] += qsize[rnode];
            qsize[rnode] = 0;
            debug!(
                "313 Setting marker [{}] to maxint {:?}",
                rnode,
                Marker::Maxint
            );
            marker[rnode] = Marker::Maxint;
            forward[rnode] = ForwardPtr::NextNeg(minimum_degree_node);
            backward[rnode] = BackPtr::NegMaxInt;
        } else {
            // flag rnode for degree update, and add minimum_degree_node as a
            // neighbor of rnode
            forward[rnode] = ForwardPtr::Next(graph.degree(rnode));
            backward[rnode] = BackPtr::None;
            graph.adjacency[rnode].push(minimum_degree_node);
        }
    }

    (graph, head, forward, backward, qsize, marker)
}

fn numbering(n_vertices: usize, in_inverse_perm: Vec<ForwardPtr>, qsize: Vec<usize>) -> MMDResult {
    debug!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    debug!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    debug!("NUMBERING");
    debug!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    debug!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    debug!("in_inverse_perm: {:?}", in_inverse_perm);
    debug!("qsize: {:?}", qsize);

    let mut iperm = vec![ForwardPtr::None; n_vertices];

    let mut perm = vec![ForwardPtr::None; n_vertices];
    for node in 0..n_vertices {
        let i = match in_inverse_perm[node] {
            ForwardPtr::NextNeg(i) => i,
            _ => panic!(),
        };

        if qsize[node] == 0 {
            perm[node] = ForwardPtr::NextNeg(i);
        } else {
            // qsize > 0
            perm[node] = ForwardPtr::Next(i);
        }
    }
    debug!("iperm: {:?}", iperm);
    debug!("perm: {:?}", perm);

    // for each node which has been merged, do the following
    for node in 0..n_vertices {
        if matches!(perm[node], ForwardPtr::NextNeg(_)) {
            // trace the merged tree until one which has not been merged, call it root
            let mut father = node;
            while let ForwardPtr::NextNeg(i) = perm[father] {
                father = i;
            }

            // number node after root
            let root = father;
            let i = match perm[root] {
                ForwardPtr::Next(i) => i,
                _ => panic!(),
            };

            let num = i + 1;
            iperm[node] = ForwardPtr::NextNeg(num);
            perm[root] = ForwardPtr::Next(num);

            // shorten the merged tree
            let mut father = node;
            let mut nextf = perm[father];
            while matches!(nextf, ForwardPtr::NextNeg(_)) {
                perm[father] = ForwardPtr::NextNeg(root);
                father = match nextf {
                    ForwardPtr::NextNeg(i) => i,
                    _ => panic!(),
                };
                nextf = perm[father];
            }
        } else {
            iperm[node] = in_inverse_perm[node];
        }
    }

    debug!("449 YOLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO");
    debug!("iperm: {:?}", iperm);
    debug!("perm: {:?}", perm);

    // ready to compute perm
    for node in 0..n_vertices {
        let num = match iperm[node] {
            ForwardPtr::NextNeg(num) => num,
            _ => panic!(),
        };
        iperm[node] = ForwardPtr::Next(num);
        perm[num] = ForwardPtr::Next(node);
    }

    MMDResult {
        iperm: iperm
            .iter()
            .map(|&f| match f {
                ForwardPtr::Next(i) => i,
                _ => panic!(),
            })
            .collect(),
        perm: perm
            .iter()
            .map(|&f| match f {
                ForwardPtr::Next(i) => i,
                _ => panic!(),
            })
            .collect(),
    }
}

fn update(
    ehead: Vec<usize>,
    graph: &MutableGraph,
    delta: usize,
    mut minimum_degree: usize,
    mut head: Vec<ForwardPtr>,
    mut forward: Vec<ForwardPtr>,
    mut backward: Vec<BackPtr>,
    mut qsize: Vec<usize>,
    mut marker: Vec<Marker>,
    mut tag: usize,
) -> (
    usize,
    Vec<ForwardPtr>,
    Vec<ForwardPtr>,
    Vec<BackPtr>,
    Vec<usize>,
    Vec<Marker>,
    usize,
) {
    let minimum_degree_0 = minimum_degree + delta;
    debug!("Set mdeg0 to mdeg {} + delta {}", minimum_degree, delta);
    // n100
    for element in ehead {
        // for each of the newly formed elements, do the following.  reset tag value if necessary
        let m_tag = tag + minimum_degree_0 + 1; // our md0 is offset by 1
        debug!("Set mtag to tag {} + mdeg0 {}", tag, minimum_degree_0);
        // I _think_ we never have to reset here (assuming we use int64)
        //if m_tag >= max_int {
        //    tag = 1;
        //    for i in 0..graph.n_vertices() {
        //        if !matches!(marker[i], Marker::Maxint) {
        //            marker[i] = Marker::Zero;
        //        }
        //    }

        //    m_tag = tag + minimum_degree_0;
        //}

        // create two linked lists from nodes associated with 'element'
        // one with two neighbors (q2head) in the adjacency structure, and the
        // other with more than two neighbors (qxhead).  also compute 'deg0',
        // number of nodes in this element.
        let mut q2 = Vec::new();
        let mut qx = Vec::new();
        let mut deg0 = -1;

        debug!("Looking at neighbors of element {}", element);

        // n400
        for &enode in graph.neighbors(element) {
            if qsize[enode] != 0 {
                deg0 += qsize[enode] as i32;
                debug!(
                    "497 Setting marker [{}] to tag {:?}",
                    enode,
                    Marker::Tag(m_tag)
                );
                marker[enode] = Marker::Tag(m_tag);

                // enode requires a degree update
                debug!("Checking enode {}", enode);
                if matches!(backward[enode], BackPtr::None) {
                    debug!("Matches");
                    // place either in qx or q2 list
                    debug!("forward: {:?}", forward[enode]);
                    // Not sure what this check is doing, but based on testing it should be 1 not 2 like in mmd.c
                    if forward[enode] != ForwardPtr::Next(1) {
                        debug!("Insert enode {} in qx", enode);
                        qx.insert(0, enode);
                    } else {
                        debug!("Insert enode {} in q2", enode);
                        q2.insert(0, enode);
                    }
                }
            }
        }

        // for each node in q2 list, do the following
        debug!("ENTER q2");
        for enode in q2 {
            if !matches!(backward[enode], BackPtr::None) {
                continue;
            }
            debug!("enode {} in q2", enode);
            tag += 1;
            let mut deg = deg0;

            // identify the other adjacent element neighbor
            let mut neighbor = graph.adjacency[enode][0];
            if neighbor == element {
                neighbor = graph.adjacency[enode][1];
            }

            if matches!(forward[neighbor], ForwardPtr::Next(_) | ForwardPtr::None) {
                // neighbor is uneliminated, increase degree count
                deg += qsize[neighbor] as i32;
            } else {
                // neighbor is eliminated. for each node in the 2nd element, do the following
                for &node in graph.neighbors(neighbor) {
                    if node != enode && qsize[node] != 0 {
                        if matches!(marker[node], Marker::Zero)
                            || matches!(marker[node], Marker::Tag(t) if t < tag)
                        {
                            // node is not yet considered
                            debug!(
                                "549 Setting marker [{}] to tag {:?}",
                                node,
                                Marker::Tag(tag)
                            );
                            marker[node] = Marker::Tag(tag);
                            deg += qsize[node] as i32;
                        } else if matches!(backward[node], BackPtr::None) {
                            if forward[node] == ForwardPtr::Next(1) {
                                // node is indistinguishable from enode.
                                // merge them into a new supernode.
                                qsize[enode] += qsize[node];
                                qsize[node] = 0;
                                debug!(
                                    "559 Setting marker [{}] to maxint {:?}",
                                    node,
                                    Marker::Maxint
                                );
                                marker[node] = Marker::Maxint;
                                forward[node] = ForwardPtr::NextNeg(enode);
                                backward[node] = BackPtr::NegMaxInt;
                            } else {
                                // node is outmatched by enode
                                if matches!(backward[node], BackPtr::None) {
                                    backward[node] = BackPtr::NegMaxInt;
                                }
                            }
                        }
                    }
                }
            }

            // n2100
            // update external degree of enode in the degree structure,
            // and mdeg if necessary
            deg -= qsize[enode] as i32 - 1;
            let fnode = head[deg as usize];
            forward[enode] = fnode;
            backward[enode] = BackPtr::Degree(deg as usize);
            if let ForwardPtr::Next(fnode_i) = fnode {
                backward[fnode_i] = BackPtr::Previous(enode);
            }
            head[deg as usize] = ForwardPtr::Next(enode);
            if deg < minimum_degree as i32 {
                minimum_degree = deg as usize; // eek
            }
        }

        // n1500
        // for each enode in the qx list, do the following
        debug!("ENTER qx");
        for enode in qx {
            debug!("enode {} in qx", enode);
            if !matches!(backward[enode], BackPtr::None) {
                debug!("continue 635, backward: {:?}", backward[enode]);
                continue;
            }
            tag += 1;
            let mut deg = deg0;

            // for each unmarked neighbor of enode, do the following
            for &neighbor in graph.neighbors(enode) {
                if matches!(marker[neighbor], Marker::Zero)
                    || matches!(marker[neighbor], Marker::Tag(t) if t < tag)
                {
                    debug!(
                        "591 Setting marker [{}] to tag {:?}",
                        neighbor,
                        Marker::Tag(tag)
                    );
                    marker[neighbor] = Marker::Tag(tag);
                    if matches!(forward[neighbor], ForwardPtr::None | ForwardPtr::Next(_)) {
                        // if uneliminated, include it in deg count
                        deg += qsize[neighbor] as i32;
                    } else {
                        // if eliminated, include unmarked nodes in this
                        // element into the degree count.
                        for &node in graph.neighbors(neighbor) {
                            if matches!(marker[node], Marker::Zero)
                                || matches!(marker[node], Marker::Tag(t) if t < tag)
                            {
                                debug!(
                                    "609 Setting marker [{}] to tag {:?}",
                                    node,
                                    Marker::Tag(tag)
                                );
                                marker[node] = Marker::Tag(tag);
                                deg += qsize[node] as i32;
                            }
                        }
                    }
                }
            }

            // update external degree of enode in the degree structure,
            // and mdeg if necessary
            deg -= qsize[enode] as i32 - 1;
            let fnode = head[deg as usize];
            forward[enode] = fnode;
            backward[enode] = BackPtr::Degree(deg as usize);
            if let ForwardPtr::Next(fnode_i) = fnode {
                backward[fnode_i] = BackPtr::Previous(enode);
            }
            head[deg as usize] = ForwardPtr::Next(enode);
            if deg < minimum_degree as i32 {
                minimum_degree = deg as usize; // eek
            }
        }

        // get next element in the list
        tag = m_tag;
    }

    (minimum_degree, head, forward, backward, qsize, marker, tag)
}
