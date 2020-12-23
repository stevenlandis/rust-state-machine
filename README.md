This is a library for doing cool things with finite automota, otherwise known as state machines. While these machines are used everywhere from parsing to device drivers, the tools to manipulate state machines are often hard to find.

This library introduces and Nfa and Dfa class that make it easy to do cool things like

- invert state machines
- concatenate state machines
- find the union of a bunch of state machines
- convert Nfa <-> Dfa
- simplify Dfas to their simplest form
- repeat state machines

One quirk of this setup is these state machines are specifically restricted to being binary state machines: They only make transitions on true or false. This makes some operations like nfa -> dfa conversion a little easier without losing generality. The cool thing about binary is that it can be used to represent just about anything. That means binary state machines can provide a core to parse everything from text to strange file formats to some made-up symbols. They just need to be convertible to binary to work!

This repo is still a work in progress, but eventually it will be turned into a library that can be easily used.

This libary makes most operations immutable. For example, calling nfa.invert() produces a new nfa with inverted behavior.
