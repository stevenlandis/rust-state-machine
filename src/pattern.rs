use crate::fa;

#[derive(Clone)]
enum Component {
  String(String),
  Action(u32),
  Then(Pattern),
  Maybe(Pattern),
  Repeat(Pattern),
  Or(Vec<Pattern>),
}

#[derive(Clone)]
pub struct Pattern {
  components: Vec<Component>,
}

type PatternNfa = fa::nfa::Nfa<u8, u32>;
type PatternDfa = fa::dfa::Dfa<u8, u32>;

impl Pattern {
  fn new() -> Pattern {
    Pattern {
      components: Vec::new(),
    }
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
  fn action(mut self, action: u32) -> Self {
    self.components.push(Component::Action(action));
    self
  }

  fn to_nfa(&self) -> PatternNfa {
    let component_nfas: Vec<PatternNfa> = self
      .components
      .iter()
      .map(|component| match component {
        Component::String(string) => {
          let string_bytes = string.as_bytes();
          let mut nfa = PatternNfa::new();
          let states: Vec<_> = (0..=string_bytes.len())
            .map(|i| nfa.add_state(i == 0, i == string_bytes.len()))
            .collect();
          for i in 0..string_bytes.len() {
            nfa.add_edge(states[i], states[i + 1], string_bytes[i], vec![])
          }
          nfa
        }
        Component::Action(action) => {
          let mut nfa = PatternNfa::new();
          let s0 = nfa.add_state(true, false);
          let s1 = nfa.add_state(false, true);
          nfa.add_empty_edge(s0, s1, vec![*action]);
          nfa
        }
        Component::Then(pattern) => pattern.to_nfa(),
        Component::Maybe(pattern) => {
          let mut nfa = PatternNfa::new();
          nfa.add_state(true, true);
          fa::nfa::Nfa::union(&vec![&nfa, &pattern.to_nfa()])
        }
        Component::Repeat(pattern) => pattern.to_nfa().repeat(),
        Component::Or(patterns) => {
          let pattern_nfas: Vec<_> = patterns.iter().map(|pattern| pattern.to_nfa()).collect();
          fa::nfa::Nfa::union(&pattern_nfas.iter().map(|nfa| nfa).collect())
        }
      })
      .collect();
    fa::nfa::Nfa::concat(&component_nfas.iter().map(|nfa| nfa).collect())
  }
}

struct Parser {
  dfa: PatternDfa,
  state: Option<fa::dfa::ParseState>,
  actions: Vec<u32>,
}
impl Parser {
  pub fn from_pattern(pattern: &Pattern) -> Option<Parser> {
    let dfa = match fa::dfa::Dfa::from_nfa(&pattern.to_nfa()) {
      Some(dfa) => dfa,
      None => {
        return None;
      }
    };
    let initial_state = dfa.get_initial_parse_state();
    Some(Parser {
      dfa,
      state: Some(initial_state),
      actions: Vec::new(),
    })
  }
  pub fn feed_str(&mut self, string: &str) {
    for byte in string.as_bytes() {
      self.state = match &self.state {
        None => {
          break;
        }
        Some(state) => match self.dfa.step(&state, *byte) {
          None => None,
          Some((new_state, new_actions)) => {
            for action in new_actions {
              self.actions.push(*action);
            }
            Some(new_state)
          }
        },
      };
    }
  }
  pub fn drain_actions(&mut self) -> Vec<u32> {
    let actions = self.actions.clone();
    self.actions = Vec::new();
    actions
  }
  pub fn is_done(&self) -> bool {
    match &self.state {
      None => false,
      Some(state) => self.dfa.is_terminal(state),
    }
  }
}

#[cfg(test)]
mod pattern_tests {
  use super::{Parser, Pattern};

  #[test]
  fn make_simple_pattern() {
    let pattern = Pattern::new()
      .then_str("stuff")
      .then_str("and")
      .then_str("things");

    let mut parser = Parser::from_pattern(&pattern).unwrap();
    parser.feed_str("stuffandthings");
    assert!(parser.is_done());
    parser.feed_str("x");
    assert!(!parser.is_done());
  }

  #[test]
  fn disambiguate_similar_patterns() {
    let pattern = Pattern::new().or(&vec![
      Pattern::new().then_str("stuff"),
      Pattern::new().then_str("stufx"),
    ]);

    let mut parser = Parser::from_pattern(&pattern).unwrap();
    parser.feed_str("stuff");
    assert!(parser.is_done());
    // assert_eq!(parser.drain_actions(), vec![1]);

    let mut parser = Parser::from_pattern(&pattern).unwrap();
    parser.feed_str("stufx");
    assert!(parser.is_done());

    let mut parser = Parser::from_pattern(&pattern).unwrap();
    parser.feed_str("stuf");
    assert!(!parser.is_done());
    // assert_eq!(parser.drain_actions(), vec![2]);
  }
}
