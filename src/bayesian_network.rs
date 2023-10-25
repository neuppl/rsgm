//! A graphical representation of a Bayesian network

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// maps each variable name to a Conditional Probability Table (CPT)
/// - rows are indexed by the current variable's possible values
///   this index comes from the order given in `states`
/// - columns are indexed by the parent's possible values
///   this index comes from the order given in the `parents` table
///
/// Example:
///
/// Suppose we have a Bayesian network (a) -> (c) <- (b)
/// - state: {"a" -> ["a1", "a2"], "b" -> ["b1", "b2", "b3"]}
/// - parents: {"a" -> [], "b" -> [], "c" -> ["a", "b"]}
/// - cpts: ```{"a" -> [[0.1], [0.9]],        /// says "a" has a prior probability  of value "a1" with prob 0.1
///          "b" -> [[0.3], [0.2], [0.5]],
///          "c" -> [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
///                  [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]]}```
/// says Pr(c=c1 | a=a1, b=b1) = 0.1
///      Pr(c=c1 | a=a2, b=b1) = 0.4
///      Pr(c=c1 | a=a1, b=b3) = 0.3
type ConditionalProbabilityTable = HashMap<String, Vec<Vec<f64>>>;
/// maps each variable name to a list of that variable's possible values
type States = HashMap<String, Vec<String>>;
/// maps each variable name to a list of that variable's parents
type Parents = HashMap<String, Vec<String>>;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BayesianNetwork {
    network: String,
    variables: Vec<String>,
    cpts: ConditionalProbabilityTable,
    states: States,
    parents: Parents,
}

impl BayesianNetwork {
    /// Generate a Bayesian Network from a JSON string.
    /// The JSON string needs to have (in JSON types):
    /// - `network`: `String`
    /// - `variables`: String[]
    /// - `cpts`: { String: Number[][] }
    /// - `states`: { String: String[] }
    /// - `parents`: { String: String[] }
    pub fn from_json(str: &str) -> BayesianNetwork {
        match serde_json::from_str(str) {
            Ok(bn) => bn,
            Err(err) => panic!("Error parsing JSON: {err}"),
        }
    }

    fn state_index(&self, variable: &str, assignment: &str) -> usize {
        let cur_s = self
            .states
            .get(variable)
            .unwrap_or_else(|| panic!("could not find variable {variable}"));
        cur_s
            .iter()
            .position(|x| *x == *assignment)
            .unwrap_or_else(|| {
                panic!("could not find assignment {assignment} for variable {variable}")
            })
    }

    fn num_states(&self, variable: &str) -> usize {
        let cur_s = self
            .states
            .get(variable)
            .unwrap_or_else(|| panic!("could not find variable {variable}"));
        cur_s.len()
    }

    /// get a list of all parents for `variable`
    /// ```
    /// use rsgm::BayesianNetwork;
    ///
    /// // models the collider A, B -> C
    /// static NETWORK: &str = r#"{
    ///     "network": "toy_network",
    ///     "variables": ["A", "B", "C"],
    ///     "cpts": {
    ///         "A": [[0.5], [0.5]],
    ///         "B": [[0.25], [0.75]],
    ///         "C": [[0.9, 0.8, 0.3, 0.4], [0.1, 0.2, 0.7, 0.6]]
    ///     },
    ///     "states": {
    ///         "A": ["F", "T"],
    ///         "B": ["F", "T"],
    ///         "C": ["F", "T"]
    ///     },
    ///     "parents" :{
    ///         "A": [],
    ///         "B": [],
    ///         "C": ["A", "B"]
    ///     }
    /// }"#;
    ///
    /// let bayesian_network = BayesianNetwork::from_json(NETWORK);
    ///
    /// assert!(bayesian_network.parents("A").is_empty());
    /// assert!(bayesian_network.parents("B").is_empty());
    /// assert_eq!(bayesian_network.parents("C").len(), 2);
    /// assert!(bayesian_network.parents("C").iter().any(|s| s == "A"));
    /// assert!(bayesian_network.parents("C").iter().any(|s| s == "B"));
    /// ```
    pub fn parents(&self, variable: &str) -> &Vec<String> {
        &self.parents[variable]
    }

    fn parent_h(&self, mut cur_parents: Vec<String>) -> Vec<HashMap<String, String>> {
        if cur_parents.is_empty() {
            return vec![HashMap::new()];
        }

        let mut r: Vec<HashMap<String, String>> = Vec::new();
        let p = cur_parents.pop().unwrap();
        let cur_values = self.all_possible_assignments(&p);
        let sub = self.parent_h(cur_parents);

        // add each assignment onto
        for v in cur_values {
            for sub_i in sub.iter() {
                let mut new_s = sub_i.clone();
                new_s.insert(p.clone(), v.clone());
                r.push(new_s);
            }
        }
        r
    }

    /// get a vector of all possible assignments to the parents of this variable
    /// ```
    /// use rsgm::BayesianNetwork;
    /// use std::collections::HashMap;
    ///
    /// // models the collider A, B -> C
    /// static NETWORK: &str = r#"{
    ///     "network": "toy_network",
    ///     "variables": ["A", "B", "C"],
    ///     "cpts": {
    ///         "A": [[0.5], [0.5]],
    ///         "B": [[0.25], [0.75]],
    ///         "C": [[0.9, 0.8, 0.3, 0.4], [0.1, 0.2, 0.7, 0.6]]
    ///     },
    ///     "states": {
    ///         "A": ["F", "T"],
    ///         "B": ["F", "T"],
    ///         "C": ["F", "T"]
    ///     },
    ///     "parents" :{
    ///         "A": [],
    ///         "B": [],
    ///         "C": ["A", "B"]
    ///     }
    /// }"#;
    ///
    /// let bayesian_network = BayesianNetwork::from_json(NETWORK);
    ///
    /// assert_eq!(bayesian_network.parent_assignments("A"), vec![HashMap::new()]);
    /// assert_eq!(bayesian_network.parent_assignments("B"), vec![HashMap::new()]);
    /// assert_eq!(bayesian_network.parent_assignments("C").len(), 4);
    /// assert!(bayesian_network.parent_assignments("C").iter().any(|s| s["A"] == "F" && s["B"] == "F"));
    /// assert!(bayesian_network.parent_assignments("C").iter().any(|s| s["A"] == "F" && s["B"] == "T"));
    /// assert!(bayesian_network.parent_assignments("C").iter().any(|s| s["A"] == "T" && s["B"] == "F"));
    /// assert!(bayesian_network.parent_assignments("C").iter().any(|s| s["A"] == "T" && s["B"] == "T"));
    /// ```
    pub fn parent_assignments(&self, variable: &str) -> Vec<HashMap<String, String>> {
        self.parent_h(self.parents(variable).clone())
    }

    /// get all variables defined in this Bayesian network
    /// ```
    /// use rsgm::BayesianNetwork;
    ///
    /// // models the collider A, B -> C
    /// static NETWORK: &str = r#"{
    ///     "network": "toy_network",
    ///     "variables": ["A", "B", "C"],
    ///     "cpts": {
    ///         "A": [[0.5], [0.5]],
    ///         "B": [[0.25], [0.75]],
    ///         "C": [[0.9, 0.8, 0.3, 0.4], [0.1, 0.2, 0.7, 0.6]]
    ///     },
    ///     "states": {
    ///         "A": ["F", "T"],
    ///         "B": ["F", "T"],
    ///         "C": ["F", "T"]
    ///     },
    ///     "parents" :{
    ///         "A": [],
    ///         "B": [],
    ///         "C": ["A", "B"]
    ///     }
    /// }"#;
    ///
    /// let bayesian_network = BayesianNetwork::from_json(NETWORK);
    ///
    /// assert_eq!(bayesian_network.variables().len(), 3);
    /// assert!(bayesian_network.variables().iter().any(|s| s == "A"));
    /// assert!(bayesian_network.variables().iter().any(|s| s == "B"));
    /// assert!(bayesian_network.variables().iter().any(|s| s == "C"));
    /// ```
    pub fn variables(&self) -> &Vec<String> {
        &self.variables
    }

    /// get all possible assignments to `variable`
    /// ```
    /// use rsgm::BayesianNetwork;
    ///
    /// // models the collider A, B -> C
    /// static NETWORK: &str = r#"{
    ///     "network": "toy_network",
    ///     "variables": ["A", "B", "C"],
    ///     "cpts": {
    ///         "A": [[0.5], [0.5]],
    ///         "B": [[0.25], [0.75]],
    ///         "C": [[0.9, 0.8, 0.3, 0.4], [0.1, 0.2, 0.7, 0.6]]
    ///     },
    ///     "states": {
    ///         "A": ["F", "T"],
    ///         "B": ["F", "T"],
    ///         "C": ["F", "T"]
    ///     },
    ///     "parents" :{
    ///         "A": [],
    ///         "B": [],
    ///         "C": ["A", "B"]
    ///     }
    /// }"#;
    ///
    /// let bayesian_network = BayesianNetwork::from_json(NETWORK);
    ///
    /// assert_eq!(bayesian_network.all_possible_assignments("A").len(), 2);
    /// assert!(bayesian_network.all_possible_assignments("A").iter().any(|s| s == "T"));
    /// assert!(bayesian_network.all_possible_assignments("A").iter().any(|s| s == "F"));
    pub fn all_possible_assignments(&self, variable: &str) -> &Vec<String> {
        &self.states[variable]
    }

    /// Get the conditional probability Pr(variable = variable_value | parent_assignment)
    /// ```
    /// use rsgm::BayesianNetwork;
    /// use std::collections::HashMap;
    ///
    /// // models the collider A, B -> C
    /// static NETWORK: &str = r#"{
    ///     "network": "toy_network",
    ///     "variables": ["A", "B", "C"],
    ///     "cpts": {
    ///         "A": [[0.5], [0.5]],
    ///         "B": [[0.25], [0.75]],
    ///         "C": [[0.9, 0.8, 0.3, 0.4], [0.1, 0.2, 0.7, 0.6]]
    ///     },
    ///     "states": {
    ///         "A": ["F", "T"],
    ///         "B": ["F", "T"],
    ///         "C": ["F", "T"]
    ///     },
    ///     "parents" :{
    ///         "A": [],
    ///         "B": [],
    ///         "C": ["A", "B"]
    ///     }
    /// }"#;
    ///
    /// let bayesian_network = BayesianNetwork::from_json(NETWORK);
    ///
    /// assert_eq!(bayesian_network.conditional_probability("A", "T", &HashMap::from([])), 0.5);
    /// assert_eq!(bayesian_network.conditional_probability("C", "T", &HashMap::from([
    ///     (String::from("A"), String::from("T")),
    ///     (String::from("B"), String::from("T"))
    /// ])), 0.6);
    /// ```
    pub fn conditional_probability(
        &self,
        variable: &str,
        variable_value: &str,
        parent_assignment: &HashMap<String, String>,
    ) -> f64 {
        let var_idx = self.state_index(variable, variable_value);
        let row = &self.cpts[variable][var_idx];
        // compute the index into the row
        let parents = self.parents.get(variable).unwrap();
        let mut cur_stride = 1;
        let mut idx = 0;
        for parent in parents.iter().rev() {
            let parent_assgn = &parent_assignment[parent];
            let parent_asggn_idx = self.state_index(parent, parent_assgn);
            idx += cur_stride * parent_asggn_idx;
            let parent_sz = self.num_states(parent);
            cur_stride *= parent_sz;
        }
        row[idx]
    }

    /// Produces a list of variables in topological order;
    /// breaks ties with the order of `variables`
    /// ```
    /// use rsgm::BayesianNetwork;
    ///
    /// // models the collider A, B -> C
    /// static NETWORK: &str = r#"{
    ///     "network": "toy_network",
    ///     "variables": ["A", "B", "C"],
    ///     "cpts": {
    ///         "A": [[0.5], [0.5]],
    ///         "B": [[0.25], [0.75]],
    ///         "C": [[0.9, 0.8, 0.3, 0.4], [0.1, 0.2, 0.7, 0.6]]
    ///     },
    ///     "states": {
    ///         "A": ["F", "T"],
    ///         "B": ["F", "T"],
    ///         "C": ["F", "T"]
    ///     },
    ///     "parents" :{
    ///         "A": [],
    ///         "B": [],
    ///         "C": ["A", "B"]
    ///     }
    /// }"#;
    ///
    /// let bayesian_network = BayesianNetwork::from_json(NETWORK);
    ///
    /// assert_eq!(bayesian_network.topological_sort().len(), 3);
    /// assert_eq!(bayesian_network.topological_sort()[0], "A");
    /// assert_eq!(bayesian_network.topological_sort()[1], "B");
    /// assert_eq!(bayesian_network.topological_sort()[2], "C");
    /// ```
    pub fn topological_sort(&self) -> Vec<String> {
        // super naive toposort
        let mut result: Vec<String> = Vec::new();
        let mut cur_vars: BTreeMap<String, Vec<String>> = self
            .variables
            .iter()
            .map(|v| (v.clone(), self.parents[v].clone()))
            .collect();

        while !cur_vars.is_empty() {
            // find a variable with no parents, remove it, add it to the result
            // list, and remove it as a parent from all other nodes
            let topvar: String = {
                let (topvar, _) = cur_vars
                    .iter()
                    .find(|(_, value)| value.is_empty())
                    .unwrap_or_else(|| panic!("graph not topologically sortable"));
                topvar.clone()
            };
            result.push(topvar.clone());
            cur_vars.remove(&topvar);
            for (_, value) in cur_vars.iter_mut() {
                value.retain(|v| *v != topvar);
            }
        }
        result
    }
}

#[test]
fn test_conditional() {
    let sachs = include_str!("../bayesian_networks/sachs.json");
    let network = BayesianNetwork::from_json(sachs);
    let parent_assgn = HashMap::from([
        (String::from("Erk"), String::from("HIGH")),
        (String::from("PKA"), String::from("AVG")),
    ]);
    assert_eq!(
        network.conditional_probability(&String::from("Akt"), &String::from("LOW"), &parent_assgn),
        0.177105936
    );
}

#[test]
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
