import argparse
import imitation_learning.env_configs
import pdb
import numpy as np

from functools import reduce
from collections import Counter
from rich import print
from imitation_learning.il import get_average_reward, get_average_reward_with_std
from qtree import QNode, QLeaf, load_tree
from statsmodels.stats.proportion import proportion_confint
from imitation_learning.utils import printv, str_avg
from imitation_learning.dt_structure_viz import viztree2qtree, load_viztree
from imitation_learning.utils import load_dataset

from scipy.special import betaincinv

def UCF(n, k):
    return betaincinv(k + 1, n - k, 1 - 0.25)

def parse_label_list(y, rule):
    N = len(y)
    y_C = y.count(rule.consequent)
    y_I = N - y_C
    ucf = UCF(N, y_I)

    return N, y_C, y_I, ucf

class Rulelist:
    def __init__(self, config, rules=[], default=0):
        self.config = config
        self.rules = rules
        self.default = default
    
    def evaluate(self, state):
        for rule in self.rules:
            if rule.evaluate(state):
                return rule.consequent
        return self.default
    
    def predict_batch(self, X):
        return [self.evaluate(x) for x in X]
    
    def act(self, state):
        return self.evaluate(state)
    
    def generalize_single_rules(self, X, y, verbose=False):
        printv(f"[yellow]Generalizing rules by evaluating pessimist error predictions..." +
            f" Dataset has size {len(X)}.[/yellow]", verbose)

        assignment_memory = np.zeros(len(X))
        for rule_id, rule in enumerate(self.rules):
            remaining_data = [(X[i], y[i]) for i in range(len(X)) if assignment_memory[i] == 0]
            if not remaining_data:
                printv(f"  No remaining data.")
                break
        
            X, y = zip(*remaining_data)
            assignment_memory = np.zeros(len(y))

            N, y_C, y_I = 0, 0, 0
            for i in range(len(X)):
                if rule.evaluate(X[i]):
                    if y[i] == rule.consequent:
                        y_C += 1
                    else:
                        y_I += 1

                    assignment_memory[i] = 1
                    N += 1

            if not N:
                printv(f"Rule #{rule_id} handles 0 observations.", verbose)
                continue

            ucf_rule = UCF(N, y_I)
            
            printv(f"{str(N).rjust(5, ' ')} observations handled by rule {rule_id}; " +
                f" \t C: {str(y_C).rjust(5, ' ')}, " +
                f"I: {str(y_I).rjust(5, ' ')}, " + 
                f"UCF: {'{:.2f}'.format(ucf_rule * 100)}%", verbose)

            while len(rule.antecedents) > 1:
                least_error = 1
                best_antecedent, best_antecedent_id = None, -1

                potential_assignment_memories = []
                for antecedent_id, antecedent in enumerate(rule.antecedents):
                    potential_assignment_memories.append(np.zeros(len(y)))

                    N_minus, y_C_minus, y_I_minus = 0, 0, 0
                    for i in range(len(X)):
                        if rule.evaluate_minus_antecedent(X[i], antecedent):
                            if y[i] == rule.consequent:
                                y_C_minus += 1
                            else:
                                y_I_minus += 1

                            N_minus += 1
                            potential_assignment_memories[antecedent_id][i] = 1

                    ucf_minus = UCF(N_minus, y_I_minus)
                    if ucf_minus < least_error:
                        least_error = ucf_minus
                        best_antecedent = antecedent
                        best_antecedent_id = antecedent_id

                    printv(f"    [bright_black]W/out antecedent {antecedent}:\t" +
                        f"N = {str(N_minus).rjust(5, ' ')}, " + 
                        f"C: {str(y_C_minus).rjust(5, ' ')}, " +
                        f"I: {str(y_I_minus).rjust(5, ' ')}, " +
                        f"UCF = {'{:.2f}'.format(ucf_minus * 100)}%[/bright_black]",
                        verbose)
                
                if least_error <= ucf_rule:
                    printv(f"   [green]Removing antecedent {best_antecedent}.[/green]", verbose)
                    rule.remove_antecedent(best_antecedent)
                    assignment_memory = potential_assignment_memories[best_antecedent_id]
                    ucf_rule = least_error
                else:
                    break
            
            # self.update_default(X, y)
            
            y_predict = self.predict_batch(X)
            matches = [(1 if y[i] == y_predict[i] else 0) for i in range(len(y))]
            accuracy = np.mean(matches)
            print(f"[yellow]In-sample accuracy for this ruleset is {accuracy}.[/yellow]")

            avg, std = get_average_reward_with_std(self.config, self, episodes=100)
            print(f"[yellow]Average reward for this rulelist is {str_avg(avg, std)}.[/yellow]")
        
        self.update_default(X, y, verbose)
    
    def update_default(self, X, y, verbose=False):
        y_default = [y_i for x_i, y_i in zip(X, y) if not any([r.evaluate(x_i) for r in self.rules])]
        N_default = len(y_default)

        if N_default:
            self.default = Counter(y_default).most_common(1)[0][0]
        
        printv(f" {str(N_default).rjust(5, ' ')} observations handled by default. " +
            f"Best action is {self.config['actions'][self.default]}", verbose)
    
    def reward_prune_rules(self, X, y, episodes, comp_threshold, verbose=False):
        printv(f"[yellow]Pruning rulelist by comparing average rewards "
            + f"over {episodes} episodes:[/yellow]", verbose)
        base_avg, base_std = get_average_reward_with_std(
            self.config, self,
            episodes=episodes,
            verbose=False)
        
        for i, rule in enumerate(self.rules):
            printv(f"  Evaluating impact of removing rule {i}...", verbose)

            rule_id = self.rules.index(rule)
            self.rules.pop(rule_id)
            potential_avg, potential_std = get_average_reward_with_std(
                self.config, self,
                episodes=episodes,
                verbose=False)

            printv(f"  [cyan]Average reward went from {str_avg(base_avg, base_std)} "
                + f"to {str_avg(potential_avg, potential_std)}.[/cyan]")
            
            comp_mult = (comp_threshold if base_avg > 0 else (2 - comp_threshold))

            if potential_avg >= base_avg * comp_mult:
                base_avg, base_std = potential_avg, potential_std
                self.update_default(X, y)
            else:
                self.rules.insert(rule_id, rule)
                printv(f"  [cyan]Undoing rule removal...[/cyan]", verbose)

    def is_observation_assigned_to_previous_rule(self, x, rule):
        for r in self.rules:
            if r == rule:
                return False
            if r.evaluate(x):
                return True
        raise Exception("This should not be happening.")
    
    def sort_rules(self, X, y, verbose=False):
        printv(f"[yellow]Ordering rulelist's {len(self.rules)} rules...[/yellow]", verbose)
        self.rules.sort(key=lambda r : r.consequent)

        action_ucf_pairs = []
        dataset_size = len(y)
        assignment_memory = np.zeros(dataset_size)
        total_N = 0

        for action_id, action in enumerate(self.config['actions']):
            ruleset = [r for r in self.rules if r.consequent == action_id]
            rule_ucf_pairs = []
            best_ucf = 1

            for rule_id, rule in enumerate(ruleset):
                remaining_data = [(X[i], y[i]) for i in range(len(X)) if assignment_memory[i] == 0]
                if not remaining_data:
                    printv(f"  No remaining data.")
                    break
            
                X, y = zip(*remaining_data)
                assignment_memory = np.zeros(len(y))

                y_C, y_I = 0, 0
                for i in range(len(X)):
                    if rule.evaluate(X[i]):
                        if y[i] == rule.consequent:
                            y_C += 1
                        else:
                            y_I += 1

                        assignment_memory[i] = 1

                N = y_C + y_I
                total_N += N

                if not N:
                    printv(f"  Rule #{rule_id} of action '{action}' has 0 observations.")
                    continue

                ucf_rule = UCF(N, y_I)
                rule_ucf_pairs.append((rule, ucf_rule))
                printv(f"  Rule #{rule_id} of action '{action}' has "
                    + f"UCF {'{:.2f}'.format(ucf_rule * 100)}%, "
                    + f"N: {N}, y_C: {y_C}, y_I: {y_I}")

                if ucf_rule < best_ucf:
                    best_ucf = ucf_rule
            
            rule_ucf_pairs.sort(key=lambda r : r[1])
            rules, _ = zip(*rule_ucf_pairs)
            
            action_ucf_pairs.append((rules, best_ucf))
        
        printv(f"Total observations handled: {total_N}. Default handles {dataset_size - total_N}.")
        action_ucf_pairs.sort(key=lambda r : r[1])
        rulesets, _ = zip(*action_ucf_pairs)
        self.rules = list(reduce(lambda x, y: x+y, rulesets))

    def load_qtree(self, qtree):
        leaves = qtree.get_leaves()
        
        rules = []
        for leaf in leaves:
            ancestors = leaf.get_ancestors()

            antecedents = [Antecedent(a.attribute, a.value, is_left) for a, is_left in ancestors]
            consequent = leaf.get_best_action()

            rule = Rule(antecedents, consequent)
            rules.append(rule)
        
        self.rules = rules

    def load_txt(self, filename):
        self.rules = []
        actions = [a.lower() for a in self.config['actions']]
        attributes = [name.lower() for name, _, _, _ in self.config['attributes']]

        with open(filename, 'r') as f:
            content = f.read()
        
        rules = content.split("\n\n")
        for rule in rules[:-1]:
            lines = rule.split("\n")
            
            parsed_antecedents = []
            for antecedent in lines[:-1]:
                test = antecedent[antecedent.find("\"") + 1 : antecedent.rfind("\"")]
                operator = "<=" if "<=" in test else ">"
                attribute, threshold = test.split(operator)

                attribute = attributes.index(attribute.strip().lower())
                threshold = float(threshold)
                lesser_than = operator == "<="

                parsed_antecedents.append(Antecedent(attribute, threshold, lesser_than))
            
            consequent = lines[-1]
            parsed_consequent = actions.index(consequent[5:].strip())
            self.rules.append(Rule(parsed_antecedents, parsed_consequent))

        default = rules[-1]
        parsed_default = actions.index(default[5:].strip())
        self.default = parsed_default
    
    def save_txt(self, filepath):
        with open(filepath, 'w') as f:
            f.write(str(self))
    
    def __str__(self):
        output = ""

        for rule in self.rules:
            output += "IF "
            
            for antecedent in rule.antecedents:
                attr_name = self.config['attributes'][antecedent.attribute][0]
                thresh = antecedent.threshold
                operator = '<=' if antecedent.lesser_than else '>'

                output += f"\"{attr_name} {operator} {thresh}\""
                if antecedent != rule.antecedents[-1]:
                    output += "\n   "
            
            action = self.config['actions'][rule.consequent]
            output += f"\nTHEN {action}"
            output += "\n\n"

        output += f"ELSE {self.config['actions'][self.default]}"
        return output

class Rule:
    def __init__(self, antecedents, consequent):
        self.antecedents = antecedents
        self.consequent = consequent
    
    def evaluate(self, state):
        return all([a.evaluate(state) for a in self.antecedents])
    
    def evaluate_minus_antecedent(self, state, antecedent):
        return all([a.evaluate(state) for a in self.antecedents if a is not antecedent])
    
    def remove_antecedent(self, antecedent):
        self.antecedents.remove(antecedent)
    
class Antecedent:
    def __init__(self, attribute, threshold, lesser_than):
        self.attribute = attribute
        self.threshold = threshold
        self.lesser_than = lesser_than
    
    def evaluate(self, state):
        if self.lesser_than:
            return state[self.attribute] <= self.threshold
        else:
            return state[self.attribute] > self.threshold
    
    def __str__(self):
        return f"x[{self.attribute}] {'<=' if self.lesser_than else '>'} {'{:.3f}'.format(self.threshold)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rulelists')
    parser.add_argument('-t','--task',help="Which task to run?", required=True)
    parser.add_argument('-f','--filename', help='Filepath for expert', required=True)
    parser.add_argument('-c','--class', help='Which type of file was loaded?', required=True)
    parser.add_argument('-o','--output', help='Filepath to output converted tree', required=False)
    parser.add_argument('--should_prune', help='Should prune loaded tree?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--comp_threshold', help='The comparison threshold for pruning', required=False, default=1, type=float)
    parser.add_argument('--grading_episodes', help='Number of episodes used during pruning.', required=False, default=10, type=int)
    parser.add_argument('--dataset', help='Dataset filename', required=False, default="")
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())
    
    config = imitation_learning.env_configs.get_config(args['task'])
    rulelist = Rulelist(config)

    if args['class'] == "QTree":
        qtree = load_tree(args['filename'])
        rulelist.load_qtree(qtree)
    elif args['class'] == "VizTree":
        string = load_viztree(args['filename'])
        qtree = viztree2qtree(config, string)
        rulelist.load_qtree(qtree)
    elif args['class'] == "Rulelist":
        rulelist.load_txt(args['filename'])
    
    print(f"[yellow]Loaded a rulelist with {len(rulelist.rules)} rules.[/yellow]")

    if args['should_prune']:
        X, y = load_dataset(args['dataset'])
        
        rulelist.sort_rules(
            X, y, verbose=args['verbose'])
        print("")
        rulelist.generalize_single_rules(
            X, y, verbose=args['verbose'])
        print("")
        rulelist.reward_prune_rules(
            X, y, args['grading_episodes'],
            comp_threshold=args['comp_threshold'],
            verbose=args['verbose'])

        print("\n[yellow]Final rulelist:[/yellow]")
        print(rulelist)
    
    if args['output']:
        rulelist.save_txt(args['output'])
    
    print("")
    avg, std = get_average_reward_with_std(
        config, rulelist, 
        episodes=args['grading_episodes'],
        verbose=False)
    print(f"[yellow]Average reward for this rulelist is {str_avg(avg, std)}.[/yellow]")