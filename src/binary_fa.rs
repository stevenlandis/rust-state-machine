use std::collections::{BTreeSet, HashMap, HashSet};

struct NfaNode {
  edges: Vec<NfaEdge>,
}
#[derive(Clone)]
enum NfaEdgeSymbol {
  Empty,
  Symbol(bool),
}
struct NfaEdge {
  symbol: NfaEdgeSymbol,
  to: usize,
}
struct Nfa {
  nodes: Vec<NfaNode>,
  start_nodes: HashSet<usize>,
  end_nodes: HashSet<usize>,
}
struct NfaState {
  nodes: HashSet<usize>,
}

#[test]
fn construct_basic_nfa() {
  let mut nfa = Nfa::new();
  let s0 = nfa.add_node(true, false);
  let s1 = nfa.add_node(false, false);
  let s2 = nfa.add_node(false, false);
  let s3 = nfa.add_node(false, true);
  nfa.add_edge(s0, s1, false);
  nfa.add_empty_edge(s0, s2);
  nfa.add_edge(s1, s3, true);
  nfa.add_edge(s2, s3, false);

  let state = nfa.get_initial_state();
  assert_eq!(nfa.is_end(&state), false);

  let state = nfa.step(&state, false);
  assert_eq!(nfa.is_end(&state), true);

  let state = nfa.step(&state, true);
  assert_eq!(nfa.is_end(&state), true);

  let state = nfa.step(&state, true);
  assert_eq!(nfa.is_end(&state), false);
}

#[test]
fn invert_basic_nfa() {
  let mut nfa = Nfa::new();
  let s0 = nfa.add_node(true, false);
  let s1 = nfa.add_node(false, false);
  let s2 = nfa.add_node(false, false);
  let s3 = nfa.add_node(false, true);
  nfa.add_edge(s0, s1, false);
  nfa.add_empty_edge(s0, s2);
  nfa.add_edge(s1, s3, true);
  nfa.add_edge(s2, s3, false);

  let nfa = nfa.invert();

  let state = nfa.get_initial_state();
  assert_eq!(nfa.is_end(&state), true);

  let state = nfa.step(&state, false);
  assert_eq!(nfa.is_end(&state), false);

  let state = nfa.step(&state, true);
  assert_eq!(nfa.is_end(&state), false);

  let state = nfa.step(&state, true);
  assert_eq!(nfa.is_end(&state), true);
}

#[derive(Clone)]
struct DfaNode {
  t_edge: usize,
  f_edge: usize,
}
struct DfaEdge {
  to: usize,
}
struct Dfa {
  nodes: Vec<DfaNode>,
  start_node: usize,
  end_nodes: HashSet<usize>,
}
struct DfaState {
  node: usize,
}

#[test]
fn convert_basic_nfa_to_dfa() {
  let mut nfa = Nfa::new();
  let s0 = nfa.add_node(true, false);
  let s1 = nfa.add_node(false, false);
  let s2 = nfa.add_node(false, false);
  let s3 = nfa.add_node(false, true);
  nfa.add_edge(s0, s1, false);
  nfa.add_empty_edge(s0, s2);
  nfa.add_edge(s1, s3, true);
  nfa.add_edge(s2, s3, false);

  let dfa = Dfa::from_nfa(&nfa);

  let state = dfa.get_initial_state();
  assert_eq!(dfa.is_end(&state), false);

  let state = dfa.step(&state, false);
  assert_eq!(dfa.is_end(&state), true);

  let state = dfa.step(&state, true);
  assert_eq!(dfa.is_end(&state), true);

  let state = dfa.step(&state, true);
  assert_eq!(dfa.is_end(&state), false);
}

impl Nfa {
  fn new() -> Nfa {
    Nfa {
      nodes: Vec::new(),
      start_nodes: HashSet::new(),
      end_nodes: HashSet::new(),
    }
  }
  pub fn add_node(&mut self, start: bool, end: bool) -> usize {
    let node_idx = self.nodes.len();
    if start {
      self.start_nodes.insert(node_idx);
    }
    if end {
      self.end_nodes.insert(node_idx);
    }
    self.nodes.push(NfaNode { edges: Vec::new() });
    node_idx
  }
  pub fn add_edge(&mut self, from: usize, to: usize, symbol: bool) {
    self.nodes[from].edges.push(NfaEdge {
      symbol: NfaEdgeSymbol::Symbol(symbol),
      to,
    });
  }
  pub fn add_empty_edge(&mut self, from: usize, to: usize) {
    self.nodes[from].edges.push(NfaEdge {
      symbol: NfaEdgeSymbol::Empty {},
      to,
    });
  }
  pub fn take_empty_edges(&self, state: &NfaState) -> NfaState {
    let mut new_nodes = state.nodes.clone();
    let mut stack: Vec<usize> = new_nodes.iter().cloned().collect();
    loop {
      match stack.pop() {
        None => {
          break;
        }
        Some(top) => {
          for edge in &self.nodes[top].edges {
            match edge.symbol {
              NfaEdgeSymbol::Empty => {
                if !new_nodes.contains(&edge.to) {
                  new_nodes.insert(edge.to);
                  stack.push(edge.to);
                }
              }
              _ => (),
            }
          }
        }
      }
    }
    NfaState { nodes: new_nodes }
  }
  fn get_initial_state(&self) -> NfaState {
    self.take_empty_edges(&NfaState {
      nodes: self.start_nodes.clone(),
    })
  }
  fn step(&self, state: &NfaState, symbol: bool) -> NfaState {
    let mut new_nodes = HashSet::<usize>::new();
    for node in &state.nodes {
      for edge in &self.nodes[*node].edges {
        match edge.symbol {
          NfaEdgeSymbol::Symbol(test_symbol) => {
            if symbol == test_symbol {
              new_nodes.insert(edge.to);
            }
          }
          _ => (),
        }
      }
    }
    NfaState { nodes: new_nodes }
  }
  fn is_end(&self, state: &NfaState) -> bool {
    state.nodes.iter().any(|node| self.end_nodes.contains(node))
  }
  fn add_nfa_as_edge(&mut self, from: usize, to: usize, nfa: &Nfa) {
    let offset = self.nodes.len();
    for _ in &nfa.nodes {
      self.add_node(false, false);
    }
    for from in 0..nfa.nodes.len() {
      for edge in &nfa.nodes[from].edges {
        self.nodes[from + offset].edges.push(NfaEdge {
          symbol: edge.symbol.clone(),
          to: edge.to + offset,
        })
      }
    }
    for i in &nfa.start_nodes {
      self.add_empty_edge(from, i + offset);
    }
    for i in &nfa.end_nodes {
      self.add_empty_edge(i + offset, to);
    }
  }
  fn repeat(&self) -> Nfa {
    let mut new_nfa = Nfa::new();
    let s0 = new_nfa.add_node(true, true);
    new_nfa.add_nfa_as_edge(s0, s0, self);
    new_nfa
  }
  fn reverse(&self) -> Nfa {
    let mut new_nfa = Nfa::new();
    for i in 0..self.nodes.len() {
      new_nfa.add_node(self.end_nodes.contains(&i), self.start_nodes.contains(&i));
    }
    let mut all_edges = Vec::<(usize, usize, NfaEdgeSymbol)>::new();
    for from in 0..self.nodes.len() {
      for edge in &self.nodes[from].edges {
        all_edges.push((from, edge.to, edge.symbol.clone()));
      }
    }
    for (to, from, symbol) in all_edges {
      new_nfa.nodes[from].edges.push(NfaEdge { to, symbol });
    }
    new_nfa
  }
  fn union(nfas: &Vec<&Nfa>) -> Nfa {
    let mut new_nfa = Nfa::new();
    let s0 = new_nfa.add_node(true, false);
    let s1 = new_nfa.add_node(false, true);
    for nfa in nfas {
      new_nfa.add_nfa_as_edge(s0, s1, nfa);
    }
    new_nfa
  }
  fn concat(nfas: &Vec<&Nfa>) -> Nfa {
    let mut new_nfa = Nfa::new();
    let connection_nodes: Vec<_> = (0..=nfas.len())
      .map(|i| new_nfa.add_node(i == 0, i == nfas.len()))
      .collect();
    for i in 0..nfas.len() {
      new_nfa.add_nfa_as_edge(connection_nodes[i], connection_nodes[i + 1], nfas[i]);
    }
    new_nfa
  }
  fn from_dfa(dfa: &Dfa) -> Nfa {
    let mut nfa = Nfa::new();
    for i in 0..dfa.nodes.len() {
      nfa.add_node(i == dfa.start_node, dfa.end_nodes.contains(&i));
    }
    for i in 0..dfa.nodes.len() {
      nfa.add_edge(i, dfa.nodes[i].f_edge, false);
      nfa.add_edge(i, dfa.nodes[i].t_edge, true);
    }
    nfa
  }
  fn to_dfa(&self) -> Dfa {
    Dfa::from_nfa(self)
  }
  fn invert(&self) -> Nfa {
    self.to_dfa().invert().to_nfa()
  }
}

impl Dfa {
  fn from_nfa(nfa: &Nfa) -> Dfa {
    let mut state_map = HashMap::<BTreeSet<usize>, usize>::new();
    let mut nodes = Vec::<DfaNode>::new();
    let mut is_node_mapped = Vec::<bool>::new();
    let mut end_nodes = HashSet::<usize>::new();
    let mut stack = Vec::<NfaState>::new();

    let get_key = |state: &NfaState| -> BTreeSet<usize> { state.nodes.iter().cloned().collect() };
    let add_state = |state: &NfaState,
                     state_map: &mut HashMap<BTreeSet<usize>, usize>,
                     nodes: &mut Vec<DfaNode>,
                     is_node_mapped: &mut Vec<bool>,
                     end_nodes: &mut HashSet<usize>|
     -> usize {
      let key = get_key(state);
      let idx = nodes.len();
      state_map.insert(key, idx);
      nodes.push(DfaNode {
        f_edge: 0,
        t_edge: 0,
      });
      is_node_mapped.push(false);
      if nfa.is_end(state) {
        end_nodes.insert(idx);
      }
      idx
    };

    let start_state = nfa.get_initial_state();
    add_state(
      &start_state,
      &mut state_map,
      &mut nodes,
      &mut is_node_mapped,
      &mut end_nodes,
    );
    stack.push(start_state);

    loop {
      match stack.pop() {
        None => break,
        Some(top) => {
          let top_key = get_key(&top);
          let top_idx = *state_map.get(&top_key).unwrap();
          if is_node_mapped[top_idx] {
            continue;
          }
          is_node_mapped[top_idx] = true;
          for symbol in &[false, true] {
            let state = nfa.step(&top, *symbol);
            let key = get_key(&state);
            let idx = match state_map.get(&key) {
              Some(idx) => *idx,
              None => {
                let idx = add_state(
                  &state,
                  &mut state_map,
                  &mut nodes,
                  &mut is_node_mapped,
                  &mut end_nodes,
                );
                stack.push(state);
                idx
              }
            };
            match symbol {
              false => {
                nodes[top_idx].f_edge = idx;
              }
              true => {
                nodes[top_idx].t_edge = idx;
              }
            };
          }
        }
      }
    }

    Dfa {
      nodes,
      start_node: 0,
      end_nodes,
    }
  }
  fn to_nfa(&self) -> Nfa {
    Nfa::from_dfa(self)
  }
  fn get_initial_state(&self) -> DfaState {
    DfaState {
      node: self.start_node,
    }
  }
  fn step(&self, state: &DfaState, symbol: bool) -> DfaState {
    DfaState {
      node: match symbol {
        false => self.nodes[state.node].f_edge,
        true => self.nodes[state.node].t_edge,
      },
    }
  }
  fn is_end(&self, state: &DfaState) -> bool {
    self.end_nodes.contains(&state.node)
  }
  fn invert(&self) -> Dfa {
    Dfa {
      nodes: self.nodes.clone(),
      start_node: self.start_node,
      end_nodes: (0..self.nodes.len())
        .filter(|i| !self.end_nodes.contains(&i))
        .collect(),
    }
  }
  fn reverse(&self) -> Dfa {
    self.to_nfa().reverse().to_dfa()
  }
  fn simplify(&self) -> Dfa {
    // https://en.wikipedia.org/wiki/DFA_minimization
    // Brzozowski's algorithm
    self.reverse().reverse()
  }

  fn print(&self) {
    for i in 0..self.nodes.len() {
      println!(
        "{}: f->{} t->{}, {}",
        i,
        self.nodes[i].f_edge,
        self.nodes[i].t_edge,
        if self.end_nodes.contains(&i) {
          "end"
        } else {
          "not end"
        }
      );
    }
  }
}
