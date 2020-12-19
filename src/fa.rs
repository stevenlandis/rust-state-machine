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
  use super::nfa;
  use std::collections::{BTreeSet, HashMap, HashSet};

  struct Edge<A> {
    to: usize,
    actions: Vec<A>,
  }
  struct State<S, A> {
    edges: HashMap<S, Edge<A>>,
  }
  pub struct Dfa<S, A> {
    states: Vec<State<S, A>>,
    initial_state: usize,
  }
  pub struct ParseState {
    state: usize,
  }

  impl<S: Eq + std::hash::Hash + Copy, A: Clone + PartialEq> Dfa<S, A> {
    pub fn from_nfa(nfa: &nfa::Nfa<S, A>) -> Option<Dfa<S, A>> {
      let initial_state = match nfa.get_initial_parse_state() {
        None => {
          return None;
        }
        Some(state) => state,
      };

      let mut state_map = HashMap::<BTreeSet<usize>, usize>::new();
      let mut nfa_states = Vec::<nfa::ParseState<A>>::new();
      let mut dfa_states = Vec::<State<S, A>>::new();
      let mut stack = HashSet::<usize>::new();

      state_map.insert(nfa::get_parse_set(&initial_state), 0);
      nfa_states.push(initial_state);
      dfa_states.push(State {
        edges: HashMap::new(),
      });
      stack.insert(0);

      loop {
        println!("stack len={}", stack.len());
        match stack.iter().next() {
          None => break,
          Some(state_idx) => {
            let state_idx = *state_idx;
            stack.remove(&state_idx);
            println!(
              "on state {}, symbols={}",
              state_idx,
              nfa::get_symbols(nfa, &nfa_states[state_idx]).len()
            );
            for symbol in nfa::get_symbols(nfa, &nfa_states[state_idx]) {
              match nfa.step(&nfa_states[state_idx], symbol) {
                None => return None,
                Some(next_state) => {
                  let (common_actions, next_state): (Vec<A>, nfa::ParseState<A>) =
                    match nfa::get_common_actions(&next_state) {
                      None => (Vec::new(), next_state),
                      Some((common_actions, new_state)) => (common_actions, new_state),
                    };
                  let key = nfa::get_parse_set(&next_state);
                  let to = match state_map.get(&key) {
                    None => {
                      println!("edge to new state");
                      let to_state_idx = dfa_states.len();
                      nfa_states.push(next_state);
                      dfa_states.push(State {
                        edges: HashMap::new(),
                      });
                      state_map.insert(key, to_state_idx);
                      stack.insert(to_state_idx);
                      to_state_idx
                    }
                    Some(to_state_idx) => {
                      println!("edge to existing state");
                      if nfa_states[*to_state_idx] != next_state {
                        return None;
                      }
                      *to_state_idx
                    }
                  };
                  dfa_states[state_idx].edges.insert(
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
      println!("result states={}", dfa_states.len());
      Some(Dfa {
        states: dfa_states,
        initial_state: 0,
      })
    }

    pub fn get_initial_parse_state(&self) -> ParseState {
      ParseState {
        state: self.initial_state,
      }
    }

    pub fn step(&self, state: ParseState, symbol: S) -> Option<(ParseState, &Vec<A>)> {
      match self.states[state.state].edges.get(&symbol) {
        None => None,
        Some(Edge { to, actions }) => Some((ParseState { state: *to }, actions)),
      }
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

  #[test]
  fn nfa_that_cannot_be_dfa_from_actions() {
    let mut nfa = Nfa::<u32, u32>::new();
    let s0 = nfa.add_state(true, false);
    let s1 = nfa.add_state(true, false);
    let s2 = nfa.add_state(false, false);
    nfa.add_edge(s0, s0, 1, vec![0]);
    nfa.add_edge(s0, s2, 2, vec![]);
    nfa.add_edge(s1, s1, 1, vec![1]);
    nfa.add_edge(s2, s2, 3, vec![]);

    let state = nfa.get_initial_parse_state().unwrap();
    assert_state(&state, &vec![(0, vec![]), (1, vec![])]);

    let state = nfa.step(&state, 1).unwrap();
    assert_state(&state, &vec![(0, vec![0]), (1, vec![1])]);

    let state = nfa.step(&state, 1).unwrap();
    let state = nfa.step(&state, 1).unwrap();
    assert_state(&state, &vec![(0, vec![0, 0, 0]), (1, vec![1, 1, 1])]);

    let state = nfa.step(&state, 2).unwrap();
    assert_state(&state, &vec![(2, vec![0, 0, 0])]);
  }
}

#[cfg(test)]
mod dfa_tests {
  use super::dfa;
  use super::nfa;

  #[test]
  fn construct_simple_dfa_from_nfa() {
    let mut nfa = nfa::Nfa::<u32, u32>::new();
    let s0 = nfa.add_state(true, false);
    let s1 = nfa.add_state(false, false);
    let s2 = nfa.add_state(false, false);
    nfa.add_edge(s0, s1, 1, vec![1]);
    nfa.add_edge(s1, s2, 2, vec![2]);

    let dfa = dfa::Dfa::from_nfa(&nfa).unwrap();
    let state = dfa.get_initial_parse_state();

    let (state, actions) = dfa.step(state, 1).unwrap();
    assert_eq!(actions, &vec![1]);

    let (state, actions) = dfa.step(state, 2).unwrap();
    assert_eq!(actions, &vec![2]);

    assert!(dfa.step(state, 3).is_none());
  }

  #[test]
  fn construct_multiple_states() {
    let mut nfa = nfa::Nfa::<u32, u32>::new();
    let s0 = nfa.add_state(true, false);
    let s1 = nfa.add_state(false, false);
    let s2 = nfa.add_state(false, false);
    let s3 = nfa.add_state(false, true);
    nfa.add_edge(s0, s1, 1, vec![1]);
    nfa.add_empty_edge(s0, s2, vec![1]);
    nfa.add_edge(s1, s3, 2, vec![2]);
    nfa.add_edge(s2, s3, 1, vec![]);

    let dfa = dfa::Dfa::from_nfa(&nfa).unwrap();
    let state = dfa.get_initial_parse_state();

    let (state, actions) = dfa.step(state, 1).unwrap();
    assert_eq!(actions, &vec![1]);

    let (state, actions) = dfa.step(state, 2).unwrap();
    assert_eq!(actions, &vec![2]);

    assert!(dfa.step(state, 42).is_none());
  }

  #[test]
  fn construct_fail_same_state_different_actions() {
    let mut nfa = nfa::Nfa::<u32, u32>::new();
    let s0 = nfa.add_state(true, false);
    let s1 = nfa.add_state(true, false);
    let s2 = nfa.add_state(false, false);
    nfa.add_edge(s0, s0, 1, vec![0]);
    nfa.add_edge(s0, s2, 2, vec![]);
    nfa.add_edge(s1, s1, 1, vec![1]);
    nfa.add_edge(s2, s2, 3, vec![]);

    assert!(dfa::Dfa::from_nfa(&nfa).is_none());
  }
}
