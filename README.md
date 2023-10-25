# rsgm

A small library for importing Bayesian networks into Rust. Primarily
used for research purposes and for [rsdd](https://github.com/neuppl/rsdd).

## Usage

### Converter

The `converter` directory contains a `bif_to_json.py` file that converts an
arbitrary `bif` file into the JSON format that can be read by the library.
To use it, first install the `pgmpy` python package. Then, it can be invoked as:

```
python bif_to_json.py input.bif > output.json
```

### Bayesian Networks

The RSGM library contains tools for parsing and manipulating Bayesian networks
from a JSON representation. The example in the `examples` directory provides
the most basic example to do so; this simply prints the JSON representation.

```
cargo run --example compile -- -f bayesian_networks/sachs.json -m print
```

In addition, it demonstrates how to compile the Bayesian Network into a CNF,
which can then be represented by various decision diagram formats:

```
cargo run --example compile -- -f bayesian_networks/sachs.json -m bdd
cargo run --example compile -- -f bayesian_networks/sachs.json -m sdd
```

Here is a "kitchen sink" example that showcases the public API:

```rs
fn test_public_api_e2e() {
    /// models the collider A, B -> C
    static NETWORK: &str = r#"{
        "network": "toy_network",
        "variables": ["A", "B", "C"],
        "cpts": {
            "A": [[0.5], [0.5]],
            "B": [[0.25], [0.75]],
            "C": [[0.9, 0.8, 0.3, 0.4], [0.1, 0.2, 0.7, 0.6]]
        },
        "states": {
            "A": ["F", "T"],
            "B": ["F", "T"],
            "C": ["F", "T"]
        },
        "parents" :{
            "A": [],
            "B": [],
            "C": ["A", "B"]
        }
    }"#;

    let bayesian_network = BayesianNetwork::from_json(NETWORK);

    // parents
    assert!(bayesian_network.parents("A").is_empty());
    assert!(bayesian_network.parents("B").is_empty());
    assert_eq!(bayesian_network.parents("C").len(), 2);
    assert!(bayesian_network.parents("C").iter().any(|s| s == "A"));
    assert!(bayesian_network.parents("C").iter().any(|s| s == "B"));

    // parent_assignments
    assert_eq!(
        bayesian_network.parent_assignments("A"),
        vec![HashMap::new()]
    );
    assert_eq!(
        bayesian_network.parent_assignments("B"),
        vec![HashMap::new()]
    );
    assert_eq!(bayesian_network.parent_assignments("C").len(), 4);
    assert!(bayesian_network
        .parent_assignments("C")
        .iter()
        .any(|s| s["A"] == "F" && s["B"] == "F"));
    assert!(bayesian_network
        .parent_assignments("C")
        .iter()
        .any(|s| s["A"] == "F" && s["B"] == "T"));
    assert!(bayesian_network
        .parent_assignments("C")
        .iter()
        .any(|s| s["A"] == "T" && s["B"] == "F"));
    assert!(bayesian_network
        .parent_assignments("C")
        .iter()
        .any(|s| s["A"] == "T" && s["B"] == "T"));

    // variables
    assert_eq!(bayesian_network.variables().len(), 3);
    assert!(bayesian_network.variables().iter().any(|s| s == "A"));
    assert!(bayesian_network.variables().iter().any(|s| s == "B"));
    assert!(bayesian_network.variables().iter().any(|s| s == "C"));

    // assignments
    assert_eq!(bayesian_network.all_possible_assignments("A").len(), 2);
    assert!(bayesian_network
        .all_possible_assignments("A")
        .iter()
        .any(|s| s == "T"));
    assert!(bayesian_network
        .all_possible_assignments("A")
        .iter()
        .any(|s| s == "F"));

    // conditionals
    assert_eq!(
        bayesian_network.conditional_probability("A", "T", &HashMap::from([])),
        0.5
    );
    assert_eq!(
        bayesian_network.conditional_probability(
            "C",
            "T",
            &HashMap::from([
                (String::from("A"), String::from("T")),
                (String::from("B"), String::from("T"))
            ])
        ),
        0.6
    );

    // topo sort
    assert_eq!(bayesian_network.topological_sort().len(), 3);
    assert_eq!(bayesian_network.topological_sort()[0], "A");
    assert_eq!(bayesian_network.topological_sort()[1], "B");
    assert_eq!(bayesian_network.topological_sort()[2], "C");
}
```
