#[derive(Clone, Copy)]
struct Node {
    priority: f32,
    value: usize,
}

pub struct PriorityQueue {
    n_nodes: usize,
    // max_nodes: usize,
    heap: Vec<Node>,
    locator: Vec<Option<usize>>,
}

impl PriorityQueue {
    fn up_heap(&mut self, mut i: Option<usize>, priority: f32, value: usize) {
        while i.is_some() {
            let j = (i.unwrap() - 1) >> 1;
            if priority < self.heap[j].priority {
                self.heap[i.unwrap()] = self.heap[j];
                self.locator[self.heap[i.unwrap()].value] = i;
                i = Some(j);
            } else {
                break;
            }
        }

        self.heap[i.unwrap()] = Node {
            priority: priority,
            value: value,
        };
        self.locator[value] = i;
    }

    fn down_heap(&mut self, mut i: Option<usize>, priority: f32, value: usize) {
        loop {
            let mut j = (i.unwrap() << 1) + 1;
            if !(j < self.n_nodes) {
                break;
            }

            if self.heap[j].priority < priority {
                if j + 1 < self.n_nodes && self.heap[j + 1].priority < self.heap[j].priority {
                    j += 1;
                }

                self.heap[i.unwrap()] = self.heap[j];
                self.locator[self.heap[i.unwrap()].value] = i;
                i = Some(j);
            } else if j + 1 < self.n_nodes && self.heap[j + 1].priority < priority {
                j += 1;

                self.heap[i.unwrap()] = self.heap[j];
                self.locator[self.heap[i.unwrap()].value] = i;
                i = Some(j);
            } else {
                break;
            }
        }

        self.heap[i.unwrap()] = Node {
            priority: priority,
            value: value,
        };
        self.locator[value] = i;
    }

    pub fn create(max_nodes: usize) -> PriorityQueue {
        PriorityQueue {
            n_nodes: 0,
            // max_nodes: max_nodes,
            heap: vec![
                Node {
                    priority: 0.0,
                    value: 0
                };
                max_nodes
            ],
            locator: vec![None; max_nodes],
        }
    }

    pub fn reset(&mut self) {
        for i in 0..self.n_nodes {
            self.locator[self.heap[i].value] = None;
        }
        self.n_nodes = 0;
    }

    pub fn insert(&mut self, value: usize, priority: f32) {
        let i = self.n_nodes;
        self.n_nodes += 1;

        self.up_heap(Some(i), priority, value);
    }

    pub fn delete(&mut self, value: usize) {
        let i = self.locator[value];
        self.locator[value] = None;

        self.n_nodes -= 1;
        if self.n_nodes > 0 && self.heap.last().unwrap().value != value {
            let node = self.heap[self.n_nodes].value;
            let new_key = self.heap[self.n_nodes].priority;
            let old_key = self.heap[i.unwrap()].priority;

            if new_key < old_key {
                self.up_heap(i, new_key, node);
            } else {
                self.down_heap(i, new_key, node);
            }
        }
    }

    pub fn update(&mut self, value: usize, priority: f32) {
        let old_priority = self.heap[self.locator[value].unwrap()].priority;
        if priority == old_priority {
            return;
        }

        let i = self.locator[value];

        if priority < old_priority {
            self.up_heap(i, priority, value);
        } else {
            self.down_heap(i, priority, value);
        }
    }

    pub fn pop(&mut self) -> Option<usize> {
        if self.n_nodes == 0 {
            return None;
        }

        self.n_nodes -= 1;

        let vertex = self.heap[0].value;
        self.locator[vertex] = None;

        let i = self.n_nodes;
        if i > 0 {
            let priority = self.heap[i].priority;
            let value = self.heap[i].value;
            self.down_heap(Some(0), priority, value);
        }
        return Some(vertex);
    }

    pub fn peek(&self) -> Option<usize> {
        match self.heap.first() {
            Some(node) => Some(node.value),
            None => None,
        }
    }
}
