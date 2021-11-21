use crate::config::Config;
use crate::graph::WeightedGraph;

pub struct BoundaryInfo {
    pub partition_weights: [i32; 3],
    pub boundary_ind: Vec<usize>,
    pub boundary_ptr: Vec<Option<usize>>,
}

impl BoundaryInfo {
    pub fn delete(&mut self, i: usize) {
        let index_to_update = self.boundary_ptr[i];
        let value_to_move = self.boundary_ind.pop();
        self.boundary_ind[index_to_update.unwrap()] = value_to_move.unwrap();
        self.boundary_ptr[value_to_move.unwrap()] = index_to_update;
        self.boundary_ptr[i] = None;
    }

    pub fn insert(&mut self, i: usize) {
        self.boundary_ind.push(i);
        self.boundary_ptr[i] = Some(self.boundary_ind.len() - 1);
    }
}

pub struct WhereIdEd {
    pub _where: Vec<usize>,
    pub id: Vec<i32>,
    pub ed: Vec<i32>,
}

pub fn compute_two_way_partitioning_params(
    config: &Config,
    graph: WeightedGraph,
    where_id_ed: WhereIdEd,
) -> (i32, WhereIdEd, BoundaryInfo) {
    // ncon had better be 1!!!!

    let mut boundary_info = BoundaryInfo {
        partition_weights: [0, 0, 0], // metis only sets the first 2 to 0???  Maybe we only need 2?
        boundary_ind: Vec::new(),
        boundary_ptr: vec![None; graph.graph.n_vertices()],
    };

    // compute partition_weights
    for i in 0..graph.graph.n_vertices() {
        boundary_info.partition_weights[where_id_ed._where[i]] += graph.vertex_weights.unwrap()[i];
    }

    //compute the required info for refinement
    let mut min_cut = 0;
    for i in 0..graph.graph.n_vertices() {
        let me = where_id_ed._where[i];
        let tid = 0;
        let ted = 0;

        for j in graph.graph.x_adjacency[i]..graph.graph.x_adjacency[i + 1] {
            if me == where_id_ed._where[graph.graph.adjacency_lists[j]] {
                tid += graph.edge_weights.unwrap()[j];
            } else {
                ted += graph.edge_weights.unwrap()[j];
            }
        }

        where_id_ed.id[i] = tid;
        where_id_ed.ed[i] = ted;

        if ted > 0 || graph.graph.degree(i) == 0 {
            boundary_info.boundary_ind.push(i);
            boundary_info.boundary_ptr[i] = Some(boundary_info.boundary_ind.len() - 1);

            min_cut += ted;
        }
    }

    let min_cut = min_cut / 2;
    (min_cut, where_id_ed, boundary_info)
}
