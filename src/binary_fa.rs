use std::collections::{BTreeSet, HashMap, HashSet};

struct NfaNode {
  edges: Vec<NfaEdge>,
  flags: HashSet<u32>,
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
  flags: HashSet<u32>,
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
#[test]
fn simple_nfa_flag() {
  let mut nfa = Nfa::new();
  let s0 = nfa.add_node(true, false);
  let s1 = nfa.add_node_with_flags(false, false, &vec![42]);
  nfa.add_edge(s0, s1, true);
  let state = nfa.get_initial_state();
  assert!(!nfa.get_flags(&state).contains(&42));
  let state = nfa.step(&state, true);
  assert!(nfa.get_flags(&state).contains(&42));
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
    self.add_node_with_flags(start, end, &vec![])
  }
  pub fn add_node_with_flags(&mut self, start: bool, end: bool, flags: &Vec<u32>) -> usize {
    let node_idx = self.nodes.len();
    if start {
      self.start_nodes.insert(node_idx);
    }
    if end {
      self.end_nodes.insert(node_idx);
    }
    self.nodes.push(NfaNode {
      edges: Vec::new(),
      flags: flags.iter().cloned().collect(),
    });
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
    self.take_empty_edges(&NfaState { nodes: new_nodes })
  }
  fn is_end(&self, state: &NfaState) -> bool {
    state.nodes.iter().any(|node| self.end_nodes.contains(node))
  }
  fn get_flags(&self, state: &NfaState) -> HashSet<u32> {
    let mut flags = HashSet::<u32>::new();
    for node_i in &state.nodes {
      for flag in &self.nodes[*node_i].flags {
        flags.insert(*flag);
      }
    }
    flags
  }
  fn add_nfa_as_edge(&mut self, from: usize, to: usize, nfa: &Nfa) {
    let offset = self.nodes.len();
    for node in &nfa.nodes {
      self.add_node_with_flags(false, false, &node.flags.iter().cloned().collect());
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
      nfa.add_node_with_flags(
        i == dfa.start_node,
        dfa.end_nodes.contains(&i),
        &dfa.nodes[i].flags.iter().cloned().collect(),
      );
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

  fn print(&self) {
    for i in 0..self.nodes.len() {
      println!(
        "{}: {} {}, flags: {:?}",
        i,
        if self.start_nodes.contains(&i) {
          "start"
        } else {
          ""
        },
        if self.end_nodes.contains(&i) {
          "end"
        } else {
          ""
        },
        self.nodes[i].flags
      );
      for edge in &self.nodes[i].edges {
        match edge.symbol {
          NfaEdgeSymbol::Empty => {
            println!("  e -> {}", edge.to);
          }
          NfaEdgeSymbol::Symbol(symbol) => {
            println!("  {} -> {}", symbol, edge.to);
          }
        }
      }
    }
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
        flags: nfa.get_flags(state),
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
  fn to_block_dfa(&self, size: usize) -> BlockDfa {
    BlockDfa::from_dfa(self, size)
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
  fn get_flags(&self, state: &DfaState) -> HashSet<u32> {
    self.nodes[state.node].flags.clone()
  }
  fn has_flag(&self, state: &DfaState, flag: &u32) -> bool {
    self.nodes[state.node].flags.contains(&flag)
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
        "{}: f->{} t->{}, {}, flags: {:?}",
        i,
        self.nodes[i].f_edge,
        self.nodes[i].t_edge,
        if self.end_nodes.contains(&i) {
          "end"
        } else {
          "not end"
        },
        self.nodes[i].flags
      );
    }
  }
}

struct BlockDfa {
  size: usize,
  nodes: Vec<BlockDfaNode>,
  start_node: usize,
  end_nodes: HashSet<usize>,
}
struct BlockDfaNode {
  edges: Vec<usize>,
}
impl BlockDfa {
  fn from_dfa(dfa: &Dfa, size: usize) -> BlockDfa {
    let mut node_map = HashMap::<usize, usize>::new();
    let mut nodes = Vec::<BlockDfaNode>::new();
    let mut is_node_mapped = Vec::<bool>::new();
    let mut end_nodes = HashSet::<usize>::new();
    let mut stack = Vec::<usize>::new();

    let add_node = |node: &usize,
                    dfa: &Dfa,
                    node_map: &mut HashMap<usize, usize>,
                    nodes: &mut Vec<BlockDfaNode>,
                    is_node_mapped: &mut Vec<bool>,
                    end_nodes: &mut HashSet<usize>|
     -> usize {
      let new_idx = nodes.len();
      nodes.push(BlockDfaNode { edges: Vec::new() });
      is_node_mapped.push(false);
      node_map.insert(*node, new_idx);
      if dfa.end_nodes.contains(&node) {
        end_nodes.insert(new_idx);
      }
      new_idx
    };

    let start_node = add_node(
      &dfa.start_node,
      dfa,
      &mut node_map,
      &mut nodes,
      &mut is_node_mapped,
      &mut end_nodes,
    );
    stack.push(dfa.start_node);

    loop {
      match stack.pop() {
        None => break,
        Some(dfa_idx) => {
          let block_dfa_idx = *node_map.get(&dfa_idx).unwrap();
          if is_node_mapped[block_dfa_idx] {
            continue;
          }
          is_node_mapped[block_dfa_idx] = true;
          for i in 0..(1 << size) {
            let mut temp_state = dfa_idx;
            for j in 0..size {
              temp_state = dfa
                .step(&DfaState { node: temp_state }, i & (1 << j) != 0)
                .node;
            }
            let to = match node_map.get(&temp_state) {
              Some(idx) => *idx,
              None => {
                let idx = add_node(
                  &temp_state,
                  dfa,
                  &mut node_map,
                  &mut nodes,
                  &mut is_node_mapped,
                  &mut end_nodes,
                );
                stack.push(temp_state);
                idx
              }
            };
            nodes[block_dfa_idx].edges.push(to);
          }
        }
      }
    }

    BlockDfa {
      size,
      nodes,
      start_node,
      end_nodes,
    }
  }

  fn print(&self) {
    for node_i in 0..self.nodes.len() {
      let node = &self.nodes[node_i];
      // map from to-state to symbols
      let mut bins = HashMap::<usize, HashSet<usize>>::new();
      for edge_i in 0..node.edges.len() {
        let to = node.edges[edge_i];
        if !bins.contains_key(&to) {
          bins.insert(to, HashSet::new());
        }
        bins.get_mut(&to).unwrap().insert(edge_i);
      }
      let mut bin_sizes: Vec<_> = bins.iter().map(|(key, val)| (key, val.len())).collect();
      // smallest to largest
      bin_sizes.sort_by_key(|(_, val)| *val);
      println!(
        "{}: {}",
        node_i,
        if self.end_nodes.contains(&node_i) {
          "end"
        } else {
          ""
        },
      );
      for bin_i in 0..(bin_sizes.len() - 1) {
        let (bin_key, _) = bin_sizes[bin_i];
        print!("  ");
        for symbol in bins.get(bin_key).unwrap() {
          print!("{} ", symbol);
        }
        println!("-> {}", bin_key);
      }
      let (last_bin_key, _) = bin_sizes.last().unwrap();
      println!("  else -> {}", last_bin_key);
    }
  }
}

#[derive(Clone)]
enum Component {
  Bit(bool),
  U8(u8),
  String(String),
  Then(Pattern),
  Maybe(Pattern),
  Repeat(Pattern),
  Or(Vec<Pattern>),
  Flag(u32),
}
#[derive(Clone)]
pub struct Pattern {
  components: Vec<Component>,
}
impl Pattern {
  fn new() -> Pattern {
    Pattern {
      components: Vec::new(),
    }
  }
  fn then_bit(mut self, bit: bool) -> Self {
    self.components.push(Component::Bit(bit));
    self
  }
  fn then_u8(mut self, val: u8) -> Self {
    self.components.push(Component::U8(val));
    self
  }
  fn then_str(mut self, string: &str) -> Self {
    self
      .components
      .push(Component::String(String::from(string)));
    self
  }
  fn then(mut self, pattern: Pattern) -> Self {
    self.components.push(Component::Then(pattern));
    self
  }
  fn maybe(mut self, pattern: Pattern) -> Self {
    self.components.push(Component::Maybe(pattern));
    self
  }
  fn repeat(mut self, pattern: Pattern) -> Self {
    self.components.push(Component::Repeat(pattern));
    self
  }
  fn or(mut self, patterns: &Vec<Pattern>) -> Self {
    self
      .components
      .push(Component::Or(patterns.iter().cloned().collect()));
    self
  }
  fn flag(mut self, flag: u32) -> Self {
    self.components.push(Component::Flag(flag));
    self
  }
  fn to_nfa(&self) -> Nfa {
    let component_nfas: Vec<Nfa> = self
      .components
      .iter()
      .map(|component| match component {
        Component::Bit(val) => {
          let mut nfa = Nfa::new();
          let s0 = nfa.add_node(true, false);
          let s1 = nfa.add_node(false, true);
          nfa.add_edge(s0, s1, *val);
          nfa
        }
        Component::U8(num) => {
          let mut pattern = Pattern::new();
          for i in 0..8 {
            pattern = pattern.then_bit(num & 1 << i != 0);
          }
          pattern.to_nfa()
        }
        Component::String(string) => {
          let mut pattern = Pattern::new();
          for byte in string.as_bytes() {
            pattern = pattern.then_u8(*byte);
          }
          pattern.to_nfa()
        }
        Component::Then(pattern) => pattern.to_nfa(),
        Component::Maybe(pattern) => {
          let mut nfa = Nfa::new();
          nfa.add_node(true, true);
          Nfa::union(&vec![&nfa, &pattern.to_nfa()])
        }
        Component::Repeat(pattern) => pattern.to_nfa().repeat(),
        Component::Or(patterns) => {
          let pattern_nfas: Vec<_> = patterns.iter().map(|pattern| pattern.to_nfa()).collect();
          Nfa::union(&pattern_nfas.iter().map(|nfa| nfa).collect())
        }
        Component::Flag(flag) => {
          let mut nfa = Nfa::new();
          nfa.add_node_with_flags(true, true, &vec![*flag]);
          nfa
        }
      })
      .collect();
    Nfa::concat(&component_nfas.iter().map(|nfa| nfa).collect())
  }
  fn to_dfa(&self) -> Dfa {
    self.to_nfa().to_dfa()
  }
}

fn feed_dfa(dfa: &Dfa, symbols: Vec<bool>) -> DfaState {
  let mut state = dfa.get_initial_state();
  for symbol in symbols {
    state = dfa.step(&state, symbol);
  }
  state
}
fn accepts_bits(dfa: &Dfa, symbols: Vec<bool>) -> bool {
  dfa.is_end(&feed_dfa(dfa, symbols))
}
fn accepts_str(dfa: &Dfa, string: &str) -> bool {
  let mut bits = Vec::<bool>::new();
  for byte in string.as_bytes() {
    for i in 0..8 {
      bits.push(byte & (1 << i) != 0);
    }
  }
  accepts_bits(dfa, bits)
}

#[test]
fn basic_pattern() {
  let pattern = Pattern::new()
    .then_bit(false)
    .then_bit(true)
    .then_bit(false);

  let dfa = pattern.to_dfa();
  assert!(accepts_bits(&dfa, vec![false, true, false]));
  assert!(!accepts_bits(&dfa, vec![true, true, false]));
  assert!(!accepts_bits(&dfa, vec![false, false, false]));
  assert!(!accepts_bits(&dfa, vec![false, true, true]));
}
#[test]
fn pattern_then_u8() {
  let pattern = Pattern::new().then_u8(0b01010101);
  let dfa = pattern.to_dfa();
  assert!(accepts_bits(
    &dfa,
    vec![true, false, true, false, true, false, true, false]
  ));
}
#[test]
fn pattern_then_str() {
  let pattern = Pattern::new().then_str("ab");
  let dfa = pattern.to_dfa();
  assert!(accepts_str(&dfa, "ab"));
  assert!(!accepts_str(&dfa, "aa"));
}
#[test]
fn pattern_repeat() {
  let pattern = Pattern::new().repeat(Pattern::new().then_str("ab"));
  let dfa = pattern.to_dfa();
  assert!(accepts_str(&dfa, ""));
  assert!(!accepts_str(&dfa, "a"));
  assert!(accepts_str(&dfa, "ab"));
  assert!(!accepts_str(&dfa, "aba"));
  assert!(accepts_str(&dfa, "abab"));
  assert!(accepts_str(&dfa, "abababababab"));
}
#[test]
fn pattern_maybe() {
  let pattern = Pattern::new()
    .then_str("stuff")
    .maybe(Pattern::new().then_str("and"))
    .then_str("things");
  let dfa = pattern.to_dfa();
  assert!(accepts_str(&dfa, "stuffthings"));
  assert!(accepts_str(&dfa, "stuffandthings"));
  assert!(!accepts_str(&dfa, "suffthings"));
  assert!(!accepts_str(&dfa, "stuffandthing"));
}
#[test]
fn pattern_or() {
  let pattern = Pattern::new().then_str("Hello my name is ").or(&vec![
    Pattern::new().then_str("Alice"),
    Pattern::new().then_str("Bob"),
  ]);
  let dfa = pattern.to_dfa();
  assert!(accepts_str(&dfa, "Hello my name is Alice"));
  assert!(accepts_str(&dfa, "Hello my name is Bob"));
}
#[test]
fn pattern_flag() {
  let pattern = Pattern::new().or(&vec![
    Pattern::new().then_bit(true).then_bit(false).flag(0),
    Pattern::new().then_bit(true).then_bit(true).flag(1),
  ]);
  let dfa = pattern.to_dfa();
  assert!(dfa.has_flag(&feed_dfa(&dfa, vec![true, false]), &0));
  assert!(dfa.has_flag(&feed_dfa(&dfa, vec![true, true]), &1));
}
