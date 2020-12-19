use std::collections::HashSet;

fn main() {
    println!("Hello, world!");
}

fn nfa_main() {
    let mut nfa = Nfa::new();
    let s0 = nfa.add_state(true, false);
    let s1 = nfa.add_state(false, false);
    let s2 = nfa.add_state(false, false);
    let s3 = nfa.add_state(false, true);
    nfa.add_edge(s0, 1, s1);
    nfa.add_empty_edge(s0, s2);
    nfa.add_edge(s1, 2, s3);
    nfa.add_edge(s2, 1, s3);

    let mut parser = NfaParser::new(&nfa);
    parser.take_empty_edges();
    dbg!(parser.is_terminal());
    parser.step(1);
    dbg!(parser.is_terminal());
    parser.step(3);
    dbg!(parser.is_terminal());
    dbg!(parser.states);
}

struct NfaState<T> {
    edges: Vec<NfaEdge<T>>,
    terminal: bool,
}

enum NfaEdge<T> {
    Empty { to: usize },
    Symbol { symbol: T, to: usize },
}

struct Nfa<T> {
    states: Vec<NfaState<T>>,
    initial_states: Vec<usize>,
}

impl<T> Nfa<T> {
    fn new() -> Nfa<T> {
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

    fn add_edge(&mut self, state: usize, symbol: T, to: usize) {
        self.states[state]
            .edges
            .push(NfaEdge::Symbol { symbol, to });
    }

    fn add_empty_edge(&mut self, state: usize, to: usize) {
        self.states[state].edges.push(NfaEdge::Empty { to });
    }
}

struct NfaParser<'a, T> {
    nfa: &'a Nfa<T>,
    states: HashSet<usize>,
}

impl<T: PartialEq> NfaParser<'_, T> {
    fn new(nfa: &Nfa<T>) -> NfaParser<T> {
        NfaParser {
            nfa,
            states: nfa.initial_states.iter().cloned().collect(),
        }
    }

    fn take_empty_edges(&mut self) {
        let mut stack: Vec<usize> = self.states.iter().cloned().collect();
        loop {
            match stack.pop() {
                None => break,
                Some(top) => {
                    for edge in &self.nfa.states[top].edges {
                        match edge {
                            NfaEdge::Empty { to } => {
                                if !(self.states.contains(to)) {
                                    self.states.insert(*to);
                                }
                            }
                            _ => (),
                        }
                    }
                }
            }
        }
    }

    fn step(&mut self, symbol: T) {
        let mut next_states: HashSet<usize> = HashSet::new();
        for state in &self.states {
            for edge in &self.nfa.states[*state].edges {
                match edge {
                    NfaEdge::Symbol {
                        symbol: test_symbol,
                        to,
                    } => {
                        if *test_symbol == symbol {
                            next_states.insert(*to);
                        }
                    }
                    _ => (),
                }
            }
        }
        self.states = next_states;
        self.take_empty_edges();
    }

    fn is_terminal(&self) -> bool {
        self.states
            .iter()
            .any(|state| self.nfa.states[*state].terminal)
    }
}
