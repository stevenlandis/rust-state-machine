mod nfa {
  use std::collections::{BTreeSet, HashMap, HashSet};
  struct State<S, A> {
    edges: Vec<Edge<S, A>>,
    terminal: bool,
  }
  enum Edge<S, A> {
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
  pub struct Nfa<S, A> {
    states: Vec<State<S, A>>,
    initial_states: Vec<usize>,
  }
  pub type ParseState<A> = HashMap<usize, Vec<A>>;

  pub fn get_parse_set<A>(state: &ParseState<A>) -> BTreeSet<usize> {
    state.keys().cloned().collect()
  }
  pub fn get_symbols<S: std::hash::Hash + Eq + Copy, A>(
    nfa: &Nfa<S, A>,
    state: &ParseState<A>,
  ) -> HashSet<S> {
    let mut symbols: HashSet<S> = HashSet::new();
    for state in state.keys() {
      for edge in &nfa.states[*state].edges {
        match edge {
          Edge::Symbol {
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
  pub fn get_common_actions<A: PartialEq + Clone>(
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

  impl<S: PartialEq, A: Clone + PartialEq> Nfa<S, A> {
    pub fn new() -> Nfa<S, A> {
      Nfa {
        states: Vec::new(),
        initial_states: Vec::new(),
      }
    }
    pub fn add_state(&mut self, initial: bool, terminal: bool) -> usize {
      let id = self.states.len();
      if initial {
        self.initial_states.push(id);
      }
      self.states.push(State {
        edges: Vec::new(),
        terminal,
      });
      id
    }
    pub fn add_edge(&mut self, from: usize, to: usize, symbol: S, actions: Vec<A>) {
      self.states[from].edges.push(Edge::Symbol {
        actions,
        to,
        symbol,
      })
    }
    pub fn add_empty_edge(&mut self, from: usize, to: usize, actions: Vec<A>) {
      self.states[from].edges.push(Edge::Empty { actions, to })
    }

    // definitions for parsing
    pub fn get_initial_parse_state(&self) -> Option<ParseState<A>> {
      let mut initial_state: ParseState<A> = HashMap::new();
      for state in &self.initial_states {
        initial_state.insert(*state, Vec::new());
      }
      self.take_empty_edges(&initial_state)
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
              Edge::Empty { actions, to } => {
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
    pub fn step(&self, state: &ParseState<A>, symbol: S) -> Option<ParseState<A>> {
      let mut next_state: ParseState<A> = HashMap::new();
      for state_idx in state.keys() {
        for edge in &self.states[*state_idx].edges {
          match edge {
            Edge::Symbol {
              actions,
              symbol: test_symbol,
              to,
            } => {
              if &symbol == test_symbol {
                let prev_actions = state.get(state_idx).unwrap();
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
    pub fn is_terminal(&self, state: &ParseState<A>) -> bool {
      state
        .keys()
        .any(|state_idx| self.states[*state_idx].terminal)
    }
  }
}

mod dfa {
  use super::nfa::{get_common_actions, get_parse_set, get_symbols, Nfa, ParseState};
  use std::collections::{BTreeSet, HashMap};

  struct Edge<A> {
    to: usize,
    actions: Vec<A>,
  }
  struct State<S, A> {
    edges: HashMap<S, Edge<A>>,
  }
  struct Dfa<S, A> {
    states: Vec<State<S, A>>,
    initial_state: usize,
  }

  impl<S: Eq + std::hash::Hash + Copy, A: Clone + PartialEq> Dfa<S, A> {
    fn from_nfa(nfa: &Nfa<S, A>) -> Option<Dfa<S, A>> {
      struct StateInfo<A> {
        state: usize,
        actions: ParseState<A>,
      }
      let mut state_map: HashMap<BTreeSet<usize>, StateInfo<A>> = HashMap::new();
      let mut states: Vec<State<S, A>> = Vec::new();
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
                    Edge {
                      to,
                      actions: common_actions,
                    },
                  );
                }
              }
            }
          }
        }
      }
      Some(Dfa {
        states,
        initial_state: 0,
      })
    }
  }
}

#[cfg(test)]
mod nfa_tests {
  use super::nfa::*;

  fn assert_state(state: &ParseState<u32>, correct: &Vec<(usize, Vec<u32>)>) {
    let mut keys: Vec<_> = state.keys().cloned().collect();
    keys.sort();
    let mut correct_keys: Vec<_> = correct
      .iter()
      .map(|(state_idx, _)| state_idx)
      .cloned()
      .collect();
    correct_keys.sort();
    assert_eq!(keys, correct_keys);

    for (state_idx, correct_actions) in correct {
      let actions = state.get(state_idx).unwrap();
      assert_eq!(actions, correct_actions);
    }
  }

  #[test]
  fn construct() {
    let mut nfa: Nfa<u32, u32> = Nfa::new();
    let s0 = nfa.add_state(true, false);
    let s1 = nfa.add_state(false, false);
    let s2 = nfa.add_state(false, false);
    let s3 = nfa.add_state(false, true);
    nfa.add_edge(s0, s1, 1, vec![42]);
    nfa.add_empty_edge(s0, s2, vec![1, 2]);
    nfa.add_edge(s1, s3, 2, vec![]);
    nfa.add_edge(s2, s3, 1, vec![3]);

    let state = nfa.get_initial_parse_state().unwrap();
    assert_eq!(nfa.is_terminal(&state), false);
    assert_state(&state, &vec![(0, vec![]), (2, vec![1, 2])]);

    let state = nfa.step(&state, 1).unwrap();
    assert_eq!(nfa.is_terminal(&state), true);
    assert_state(&state, &vec![(1, vec![42]), (3, vec![1, 2, 3])]);
  }

  #[test]
  fn fail_on_empty_loop() {
    let mut nfa: Nfa<u32, u32> = Nfa::new();
    let s0 = nfa.add_state(true, false);
    let s1 = nfa.add_state(false, false);
    let s2 = nfa.add_state(false, false);
    let s3 = nfa.add_state(false, false);
    nfa.add_empty_edge(s0, s1, vec![0]);
    nfa.add_empty_edge(s1, s2, vec![]);
    nfa.add_empty_edge(s2, s3, vec![]);
    nfa.add_empty_edge(s3, s0, vec![]);

    let state = nfa.get_initial_parse_state();
    assert_eq!(state, None);
  }

  #[test]
  fn pass_on_empty_loop_without_actions() {
    let mut nfa: Nfa<u32, u32> = Nfa::new();
    let s0 = nfa.add_state(true, false);
    let s1 = nfa.add_state(false, false);
    let s2 = nfa.add_state(false, false);
    let s3 = nfa.add_state(false, false);
    nfa.add_empty_edge(s0, s1, vec![]);
    nfa.add_empty_edge(s1, s2, vec![]);
    nfa.add_empty_edge(s2, s3, vec![]);
    nfa.add_empty_edge(s3, s0, vec![]);

    let state = nfa.get_initial_parse_state().unwrap();
    assert_state(
      &state,
      &vec![(0, vec![]), (1, vec![]), (2, vec![]), (3, vec![])],
    )
  }

  #[test]
  fn fail_on_action_collision() {
    let mut nfa: Nfa<u32, u32> = Nfa::new();
    let s0 = nfa.add_state(true, false);
    let s1 = nfa.add_state(false, false);
    let s2 = nfa.add_state(false, false);
    let s3 = nfa.add_state(false, false);
    nfa.add_empty_edge(s0, s1, vec![0]);
    nfa.add_empty_edge(s0, s2, vec![1]);
    nfa.add_edge(s1, s3, 1, vec![]);
    nfa.add_edge(s2, s3, 1, vec![]);

    let state = nfa.get_initial_parse_state().unwrap();
    assert_state(&state, &vec![(0, vec![]), (1, vec![0]), (2, vec![1])]);

    let state = nfa.step(&state, 1);
    assert_eq!(state, None);
  }
}
