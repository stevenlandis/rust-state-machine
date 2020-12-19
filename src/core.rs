use std::collections::{BTreeSet, HashMap, HashSet};

struct NfaState<S, A> {
  edges: Vec<NfaEdge<S, A>>,
  terminal: bool,
}
enum NfaEdge<S, A> {
  Empty {
    actions: Vec<A>,
    to: usize,
  },
  Symbol {
    actions: Vec<A>,
    to: usize,
    symbol: S,
  },
}
struct Nfa<S, A> {
  states: Vec<NfaState<S, A>>,
  initial_states: Vec<usize>,
}
type ParseState<A> = HashMap<usize, Vec<A>>;
impl<S: PartialEq, A: Clone + PartialEq> Nfa<S, A> {
  fn new() -> Nfa<S, A> {
    Nfa {
      states: Vec::new(),
      initial_states: Vec::new(),
    }
  }
  fn add_state(&mut self, initial: bool, terminal: bool) -> usize {
    let id = self.states.len();
    if initial {
      self.initial_states.push(id);
    }
    self.states.push(NfaState {
      edges: Vec::new(),
      terminal,
    });
    id
  }
  fn add_edge(&mut self, from: usize, to: usize, symbol: S, actions: Vec<A>) {
    self.states[from].edges.push(NfaEdge::Symbol {
      actions,
      to,
      symbol,
    })
  }
  fn add_empty_edge(&mut self, from: usize, to: usize, actions: Vec<A>) {
    self.states[from].edges.push(NfaEdge::Empty { actions, to })
  }

  // definitions for parsing
  fn get_initial_parse_state(&self) -> ParseState<A> {
    let mut initial_state: ParseState<A> = HashMap::new();
    for state in &self.initial_states {
      initial_state.insert(*state, Vec::new());
    }
    initial_state
  }
  fn take_empty_edges(&self, state: &ParseState<A>) -> Option<ParseState<A>> {
    let mut next_state: ParseState<A> = state.clone();
    let mut changed = true;
    while changed {
      changed = false;
      let keys: Vec<_> = next_state.keys().cloned().collect();
      for state in keys {
        for edge in &self.states[state].edges {
          match edge {
            NfaEdge::Empty { actions, to } => {
              let prev_actions = next_state.get(&state).unwrap();
              let new_actions = [&prev_actions[..], &actions[..]].concat();
              if next_state.contains_key(&to) {
                if next_state.get(&to).unwrap() != &new_actions {
                  return None;
                }
              } else {
                changed = true;
                next_state.insert(*to, new_actions);
              }
            }
            _ => (),
          }
        }
      }
    }
    return Some(next_state);
  }
  fn step(&self, state: &ParseState<A>, symbol: S) -> Option<ParseState<A>> {
    let mut next_state: ParseState<A> = HashMap::new();
    for state in state.keys() {
      for edge in &self.states[*state].edges {
        match edge {
          NfaEdge::Symbol {
            actions,
            symbol: test_symbol,
            to,
          } => {
            if &symbol == test_symbol {
              let prev_actions = next_state.get(state).unwrap();
              let new_actions = [&prev_actions[..], &actions[..]].concat();
              if next_state.contains_key(to) {
                if next_state.get(to).unwrap() != &new_actions {
                  return None;
                }
              } else {
                next_state.insert(*to, new_actions);
              }
            }
          }
          _ => (),
        }
      }
    }
    self.take_empty_edges(&next_state)
  }
}

struct DfaEdge<A> {
  to: usize,
  actions: Vec<A>,
}
struct DfaState<S, A> {
  edges: HashMap<S, DfaEdge<A>>,
}
struct Dfa<S, A> {
  states: Vec<DfaState<S, A>>,
  initial_state: usize,
}

fn get_parse_set<A>(state: &ParseState<A>) -> BTreeSet<usize> {
  state.keys().cloned().collect()
}
fn get_symbols<S: std::hash::Hash + Eq + Copy, A>(
  nfa: &Nfa<S, A>,
  state: &ParseState<A>,
) -> HashSet<S> {
  let mut symbols: HashSet<S> = HashSet::new();
  for state in state.keys() {
    for edge in &nfa.states[*state].edges {
      match edge {
        NfaEdge::Symbol {
          to: _,
          symbol,
          actions: _,
        } => {
          symbols.insert(*symbol);
        }
        _ => (),
      }
    }
  }
  symbols
}
fn get_common_actions<A: PartialEq + Clone>(
  state: &ParseState<A>,
) -> Option<(Vec<A>, ParseState<A>)> {
  if state.is_empty() {
    return None;
  }
  let model_actions = state.values().next().unwrap();
  let mut prefix_len: usize = 0;
  let max_len = state.values().map(|actions| actions.len()).min().unwrap();
  while prefix_len < max_len {
    if state
      .values()
      .any(|actions| actions[prefix_len] != model_actions[prefix_len])
    {
      break;
    }
    prefix_len += 1;
  }
  if prefix_len == 0 {
    return None;
  }
  let common_actions: Vec<_> = model_actions[..prefix_len].iter().cloned().collect();
  let mut new_state: ParseState<A> = HashMap::new();
  for (state_idx, old_actions) in state {
    new_state.insert(
      *state_idx,
      old_actions[prefix_len..].iter().cloned().collect(),
    );
  }
  Some((common_actions, new_state))
}

impl<S: Eq + std::hash::Hash + Copy, A: Clone + PartialEq> Dfa<S, A> {
  fn from_nfa(nfa: &Nfa<S, A>) -> Option<Dfa<S, A>> {
    struct StateInfo<A> {
      state: usize,
      actions: ParseState<A>,
    }
    let mut state_map: HashMap<BTreeSet<usize>, StateInfo<A>> = HashMap::new();
    let mut states: Vec<DfaState<S, A>> = Vec::new();
    let mut stack: Vec<StateInfo<A>> = Vec::new();
    loop {
      match stack.pop() {
        None => break,
        Some(top) => {
          if state_map.contains_key(&get_parse_set(&top.actions)) {
            continue;
          }
          for symbol in get_symbols(nfa, &top.actions) {
            match nfa.step(&top.actions, symbol) {
              None => return None,
              Some(next_state) => {
                let (common_actions, next_state): (Vec<A>, ParseState<A>) =
                  match get_common_actions(&next_state) {
                    None => (Vec::new(), next_state),
                    Some((common_actions, new_state)) => (common_actions, new_state),
                  };
                let key = get_parse_set(&next_state);
                let to = match state_map.get(&key) {
                  None => {
                    let state_idx = states.len();
                    state_map.insert(
                      key,
                      StateInfo {
                        state: state_idx,
                        actions: next_state,
                      },
                    );
                    state_idx
                  }
                  Some(info) => {
                    if info.actions != next_state {
                      return None;
                    }
                    info.state
                  }
                };
                states[top.state].edges.insert(
                  symbol,
                  DfaEdge {
                    to,
                    actions: common_actions,
                  },
                );
              }
            }
          }
        }
      }
    }    Some(Dfa {
      states,
      initial_state: 0,
    })
  }
}
