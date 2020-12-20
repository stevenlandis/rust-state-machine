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

impl Pattern {
  fn new() -> Pattern {
    Pattern {
      components: Vec::new(),
    }
  }
  fn then_str(&mut self, string: &str) -> &mut Pattern {
    self
      .components
      .push(Component::String(String::from(string)));
    self
  }
  fn then(&mut self, pattern: Pattern) -> &mut Pattern {
    self.components.push(Component::Then(pattern));
    self
  }
  fn maybe(&mut self, pattern: Pattern) -> &mut Pattern {
    self.components.push(Component::Maybe(pattern));
    self
  }
  fn repeat(&mut self, pattern: Pattern) -> &mut Pattern {
    self.components.push(Component::Repeat(pattern));
    self
  }
  fn or(&mut self, patterns: &Vec<Pattern>) -> &mut Pattern {
    self
      .components
      .push(Component::Or(patterns.iter().cloned().collect()));
    self
  }
  fn action(&mut self, action: u32) -> &mut Pattern {
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
  nfa: PatternNfa,
  state: Option<fa::dfa::ParseSate>,
}
impl Parser {
  pub fn from_pattern(pattern: &Pattern) -> Parser {
    Parser {
      nfa: pattern.to_nfa(),
    }
  }
}

#[cfg(test)]
mod pattern_tests {
  use super::Pattern;

  #[test]
  fn make_simple_pattern() {
    let pattern = Pattern::new()
      .then_str("stuff")
      .then_str("and")
      .then_str("things");
  }
}

/*
let pat_struct = pattern::new()
  .then(pat_pub)
  .then(pat_ws)
  .then(pat_struct_token)
  .then(pat_ws)
  .then(pat_identifier)
  .then(pat_ws)
  .then(pat_braket_open)
  .maybe(pattern::new()
    .then(pattern::new()
      .then(pat_identifier)
      .then(pat_ws)
      .then(pat_colon)
      .then(pat_ws)
      .then(pat_type)
    )
    .separted_by(pat_comma)
  )
  .then(pat_braket_end)
*/
