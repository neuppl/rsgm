use std::collections::HashMap;

use rsdd::{
    repr::{
        cnf::Cnf,
        var_label::{Literal, VarLabel},
        wmc::WmcParams,
    },
    util::semirings::{RealSemiring, Semiring},
};

use crate::BayesianNetwork;

/// Contains a Bayesian network that was compiled to a CNF
#[derive(Debug, Clone)]
pub struct BayesianNetworkCNF {
    cnf: Cnf,
    /// maps Variable Name -> (Variable Assignment -> Label)
    indicators: HashMap<String, HashMap<String, VarLabel>>,
    params: WmcParams<RealSemiring>,
}

impl BayesianNetworkCNF {
    pub fn from_bayesian_network(network: &BayesianNetwork) -> BayesianNetworkCNF {
        let mut clauses: Vec<Vec<Literal>> = Vec::new();
        let mut wmc_params: HashMap<VarLabel, (RealSemiring, RealSemiring)> = HashMap::new();
        let mut var_count = 0;

        // create one indicator for every variable assignment
        // maps Variable Name -> (Variable Assignment -> Label)
        let mut indicators: HashMap<String, HashMap<String, VarLabel>> = HashMap::new();

        for variable in network.topological_sort() {
            // create this variable's indicators and parameter clauses
            let mut cur_indic: Vec<Literal> = Vec::new();
            indicators.insert(variable.clone(), HashMap::new());
            for variable_assignment in network.all_possible_assignments(&variable) {
                let cur_var = VarLabel::new_usize(var_count);
                let new_indic = Literal::new(cur_var, true);
                wmc_params.insert(cur_var, (RealSemiring::one(), RealSemiring::one()));
                cur_indic.push(new_indic);
                indicators
                    .get_mut(&variable)
                    .unwrap()
                    .insert(variable_assignment.clone(), cur_var);
                var_count += 1;

                for parent_assignment in network.parent_assignments(&variable) {
                    let cur_param = VarLabel::new_usize(var_count);
                    let cur_prob = network.conditional_probability(
                        &variable,
                        variable_assignment,
                        &parent_assignment,
                    );
                    wmc_params.insert(cur_param, (RealSemiring::one(), RealSemiring(cur_prob)));
                    var_count += 1;

                    // build cur_param <=> cur_assgn /\ cur_indic
                    let mut indic_vec: Vec<Literal> = parent_assignment
                        .iter()
                        .map(|(varname, varval)| {
                            let label = indicators[varname][varval];
                            Literal::new(label, true)
                        })
                        .collect();
                    indic_vec.push(new_indic);

                    let mut imp1 = implies(&[Literal::new(cur_param, true)], &indic_vec);
                    let mut imp2 = implies(&indic_vec, &[Literal::new(cur_param, true)]);
                    clauses.append(&mut imp1);
                    clauses.append(&mut imp2);
                }
            }
            // build exactly-one for indicator clause
            clauses.append(&mut exactly_one(cur_indic));
        }
        BayesianNetworkCNF {
            cnf: Cnf::new(clauses),
            indicators,
            params: WmcParams::new(wmc_params),
        }
    }

    pub fn indicator(&self, var: &String, value: &String) -> VarLabel {
        self.indicators[var][value]
    }

    pub fn cnf(&self) -> &Cnf {
        &self.cnf
    }

    pub fn params(&self) -> &WmcParams<RealSemiring> {
        &self.params
    }
}

/// construct a CNF for the two TERMS (i.e., conjunctions of literals) t1 => t2
fn implies(t1: &[Literal], t2: &[Literal]) -> Vec<Vec<Literal>> {
    let mut r: Vec<Vec<Literal>> = Vec::new();
    // negate the lhs
    let lhs: Vec<Literal> = t1
        .iter()
        .map(|l| Literal::new(l.get_label(), !l.get_polarity()))
        .collect();
    for v in t2.iter() {
        let mut new_clause = lhs.clone();
        new_clause.push(*v);
        r.push(new_clause);
    }
    r
}

/// constructs a CNF constraint where exactly one of `lits` is true
fn exactly_one(lits: Vec<Literal>) -> Vec<Vec<Literal>> {
    let mut r: Vec<Vec<Literal>> = Vec::new();
    // one must be true
    r.push(lits.clone());
    // pairwise constraints
    for x in 0..(lits.len()) {
        for y in (x + 1)..(lits.len()) {
            r.push(vec![lits[x].negated(), lits[y].negated()]);
        }
    }
    r
}
