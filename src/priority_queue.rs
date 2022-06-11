#[derive(Clone, Copy, Debug)]
struct Node {
    priority: f32,
    value: usize,
}

/// This is a MAX heap, not a min heap
#[derive(Debug)]
pub struct PriorityQueue {
    n_nodes: usize,
    // max_nodes: usize,
    heap: Vec<Node>,
    locator: Vec<Option<usize>>,
}

impl PriorityQueue {
    fn up_heap(&mut self, mut i: usize, priority: f32, value: usize) {
        while i > 0 {
            let j = (i - 1) >> 1;
            if priority > self.heap[j].priority {
                self.heap[i] = self.heap[j];
                self.locator[self.heap[i].value] = Some(i);
                i = j;
            } else {
                break;
            }
        }

        self.heap[i] = Node {
            priority: priority,
            value: value,
        };
        self.locator[value] = Some(i);
    }

    fn down_heap(&mut self, mut i: usize, priority: f32, value: usize) {
        loop {
            let mut j = (i << 1) + 1;
            if !(j < self.n_nodes) {
                break;
            }

            if self.heap[j].priority > priority {
                if j + 1 < self.n_nodes && self.heap[j + 1].priority > self.heap[j].priority {
                    j += 1;
                }

                self.heap[i] = self.heap[j];
                self.locator[self.heap[i].value] = Some(i);
                i = j;
            } else if j + 1 < self.n_nodes && self.heap[j + 1].priority > priority {
                j += 1;

                self.heap[i] = self.heap[j];
                self.locator[self.heap[i].value] = Some(i);
                i = j;
            } else {
                break;
            }
        }

        self.heap[i] = Node {
            priority: priority,
            value: value,
        };
        self.locator[value] = Some(i);
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

        self.up_heap(i, priority, value);
    }

    pub fn delete(&mut self, value: usize) {
        let i = self.locator[value].unwrap();
        self.locator[value] = None;

        self.n_nodes -= 1;
        if self.n_nodes > 0 && i != self.n_nodes {
            let node = self.heap[self.n_nodes].value;
            let new_key = self.heap[self.n_nodes].priority;
            let old_key = self.heap[i].priority;

            if new_key > old_key {
                self.up_heap(i, new_key, node);
            } else {
                self.down_heap(i, new_key, node);
            }
        }
    }

    pub fn update(&mut self, value: usize, priority: f32) {
        let i = self.locator[value].unwrap();

        let old_priority = self.heap[i].priority;
        if priority == old_priority {
            return;
        }

        if priority > old_priority {
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
            self.down_heap(0, priority, value);
        }
        return Some(vertex);
    }

    pub fn peek(&self) -> Option<usize> {
        if self.n_nodes == 0 {
            None
        } else {
            Some(self.heap.first().unwrap().value)
        }
    }
}
