//! A graphical representation of a Bayesian network

use std::collections::HashMap;

use serde::{Serialize, Deserialize};
use serde_json::{Result, Value};

/// maps each variable name to a CPT table
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
/// - cpts: ```{"a" -> [[0.1], [0.9]],        // says "a" has a prior probability  of value "a1" with prob 0.1
///          "b" -> [[0.3], [0.2], [0.5]], 
///          "c" -> [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],      
///                  [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]]}``` 
/// says Pr(c=c1 | a=a1, b=b1) = 0.1
///      Pr(c=c1 | a=a2, b=b1) = 0.4
///      Pr(c=c1 | a=a1, b=b3) = 0.3
type cpt = HashMap<String, Vec<Vec<f64>>>;
/// maps each variable name to a list of that variable's possible values
type states = HashMap<String, Vec<String>>;
/// maps each variable name to a list of that variable's parents
type parents = HashMap<String, Vec<String>>;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BayesianNetwork {
    network: String,
    variables: Vec<String>,
    cpts: cpt, 
    states: states,
    parents: parents
}

impl BayesianNetwork {
    pub fn from_file(fname: &str) -> BayesianNetwork {
        serde_json::from_str(&fname).unwrap()
    }

    fn get_state_index(&self, variable: &String, assignment: &String) -> usize {
        let cur_s = self.states.get(variable).unwrap_or_else(|| panic!("could not find variable {variable}"));
        cur_s.into_iter().position(|x| *x == *assignment).unwrap_or_else(|| panic!("could not find assignment {assignment} for variable {variable}"))
    }

    fn get_num_states(&self, variable: &String) -> usize {
        let cur_s = self.states.get(variable).unwrap_or_else(|| panic!("could not find variable {variable}"));
        cur_s.len()
    }

    /// Get the conditional probability Pr(variable = variable_value | parent_assignment)
    pub fn get_conditional_prob(&self, variable: &String, variable_value: &String, parent_assignment: &HashMap<String, String>) -> f64 {
        let var_idx = self.get_state_index(variable, variable_value);
        let row = &self.cpts[variable][var_idx];
        // compute the index into the row
        let parents = self.parents.get(variable).unwrap();
        let mut cur_stride = 1;
        let mut idx = 0;
        for parent in parents.into_iter().rev() {
            let parent_assgn = &parent_assignment[parent];
            let parent_asggn_idx = self.get_state_index(parent, parent_assgn);
            idx += cur_stride * parent_asggn_idx;
            let parent_sz = self.get_num_states(parent);
            cur_stride = cur_stride * parent_sz;
        }
        row[idx]
    }

    /// Produces a list of variables in topological order
    pub fn topological_sort(&self) -> Vec<String> {
        // super naive toposort
        let mut result : Vec<String> = Vec::new();
        let mut cur_vars: HashMap<String, Vec<String>> = self.variables.iter().map(|v| {
            (v.clone(), self.parents[v].clone())
        }).collect();

        while !cur_vars.is_empty() {
            // find a variable with no parents, remove it, add it to the result
            // list, and remove it as a parent from all other nodes
            let topvar : String = {
                let (topvar, _) = cur_vars.iter().find(|(_, value)| { value.is_empty() }).unwrap_or_else(|| panic!("graph not topologically sortable"));
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
