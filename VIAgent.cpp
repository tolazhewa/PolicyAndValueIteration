#include "VIAgent.hpp"

/**
 * Constructor
 **/
VIAgent::VIAgent(Environment env, double theta)
: env(env)
, theta(theta)
, number_of_iter(0)
, state_values({{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}}) {
    for (int i = 0; i < 4; i++) {
        std::vector<std::vector<int>> tmp;
        for (int j = 0; j < 4; j++) {
            std::vector<int> tmp2;
            for (auto a : env.get_actions()){
                tmp2.push_back(a);
            }
            tmp.push_back(tmp2);
        }
        policy.push_back(tmp);
    }
}

/**
 * return the max value of a vector
 **/
double VIAgent::max(std::vector<double> arr) {
    if (arr.size() <= 0)
        return -1;
        
    double max = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

/**
 * returns the max argument
 **/
double VIAgent::argmax(std::vector<double> arr){
    if (arr.size() <= 0)
        return -1;
    int argmax = 0;
    double max = arr[0];
    for (int i = 1; i < arr.size(); i++){
        if (arr[i] > max){
            argmax = i;
            max = arr[i];
        }
    }
    return argmax;
}

/**
 * returns the absolute value
 **/
double VIAgent::abs(double val){
    return (val >= 0) ? val : val * -1;
}

/**
 * Value Iteration
 **/
void VIAgent::value_iter(){
    std::vector<std::vector<std::vector<int>>> 
    pol(
        state_values.size(),
        std::vector<std::vector<int>>(
            state_values.size(),
            std::vector<int>(0, 0)
        )
    );

    double delta, temp;
    do {
        number_of_iter++;
        delta = 0.0;
        for (int i = 0; i < state_values.size(); i++){
            for (int j = 0; j < state_values[i].size(); j++){
                temp = state_values[i][j];
                state_values[i][j] = calc_action_state_value(i,j);
                delta = max({ delta, abs(temp - state_values[i][j]) });
            }
        }
    } while(delta >= theta);

    for (int i = 0; i < pol.size(); i++){
        for (int j = 0; j < pol[i].size(); j++){
            pol[i][j] = get_greedy_action(i,j);
        }
    }
    policy = pol;
}

/**
 * Returns the greedy action for a state
 **/
std::vector<int> VIAgent::get_greedy_action(int i, int j){
    if ((i == 0 && j == 0) || (i == 3 && j == 3))
        return {-1};

    std::vector<int> optimal_actions;
    std::vector<int> temp = get_next_state(i,j,env.get_actions()[0]);
    if (!validate_state(temp)){
        temp = get_next_state(i,j,env.get_actions()[1]);
    }

    double max = state_values[temp[0]][temp[1]];

    for (auto a : env.get_actions()) {
        temp = get_next_state(i,j,a);
        if (!validate_state(temp))
            continue;

        if (state_values[temp[0]][temp[1]] == max){
            optimal_actions.push_back(a);
        } else if (state_values[temp[0]][temp[1]] > max) {
            max = state_values[temp[0]][temp[1]];
            optimal_actions = { a };
        }
    }
    return optimal_actions;
}

/**
 * Calculates the action-state value
 **/
double VIAgent::calc_action_state_value(int i, int j){
    if ((i == 0 && j == 0) || (i == 3 && j == 3))
        return 0;
    
    std::vector<double> action_values;
    double sum;
    for (auto a : env.get_actions()){
        if (!validate_state(get_next_state(i,j,a)))
            continue;
        sum = 0.0;
        for (auto s : prod_prob_arr(i,j,a)) {
            if (s[0] <= 0.0)
                continue;
            sum += s[0] * (env.get_action_rewards()[a] + env.get_discount() * state_values[(int)s[1]][(int)s[2]]);
        }
        action_values.push_back(sum);
    }
    return max(action_values);
}

/**
 * Returns next state given action
 **/
std::vector<int> VIAgent::get_next_state(int i, int j, int action){
    std::vector<int> next_state;
    switch (action) {
        case LEFT:
            next_state.push_back(i);
            next_state.push_back(j-1);
            break;

        case RIGHT:
            next_state.push_back(i);
            next_state.push_back(j+1);
            break;
        
        case UP:
            next_state.push_back(i-1);
            next_state.push_back(j);
            break;

        case DOWN:
            next_state.push_back(i+1);
            next_state.push_back(j);
            break;

        default:
            break;
    }
    return next_state;
}

/**
 * Validates the state (checks bounds)
 **/
bool VIAgent::validate_state(std::vector<int> state){
    if (state[0] < 0 || state[0] >= state_values.size() || state[1] < 0 || state[1] >= state_values.size())
        return false;
    return true;
}

/**
 * Produces the actions based off probabilities of the environment
 **/
std::vector<std::vector<double>> VIAgent::prod_prob_arr(int i, int j, int action){
    std::vector<std::vector<double>> states;
    std::vector<double> next_state = { env.get_p1() }, 
                        curr_state = { env.get_p2() }, 
                        adj1 = { (1.0 - env.get_p1() - env.get_p2()) / 2 }, 
                        adj2 = { (1.0 - env.get_p1() - env.get_p2()) / 2 };
    std::vector<int> temp;

    temp = get_next_state(i,j,action);
    next_state.push_back(temp[0]);
    next_state.push_back(temp[1]);

    curr_state.push_back(i);
    curr_state.push_back(j);
    
    if (action == UP || action == DOWN){
        adj1.push_back(next_state[1]);
        adj1.push_back(next_state[2] + 1);
        adj2.push_back(next_state[1]);
        adj2.push_back(next_state[2] - 1);
    } else {
        adj1.push_back(next_state[1] + 1);
        adj1.push_back(next_state[2]);
        adj2.push_back(next_state[1] - 1);
        adj2.push_back(next_state[2]);
    }

    if (!validate_state({ (int)adj1[1], (int)adj1[2] })) {
        adj1[0] = 0.0;
        adj2[0] = 1.0 - env.get_p1() - env.get_p2();
    } else if (!validate_state({ (int)adj2[1], (int)adj2[2] })) {
        adj1[0] = 1.0 - env.get_p1() - env.get_p2();
        adj2[0] = 0.0;
    }

    states.push_back(next_state);
    states.push_back(curr_state);
    states.push_back(adj1);
    states.push_back(adj2);

    return states;
}

/**
 * Prints the state values
 **/
void VIAgent::print_state_values(std::ostream &file) {
    file << "  |--------------------------|  " << std::endl;
    file << "  |------ State Values ------|  " << std::endl;
    file << "  |--------------------------|  " << std::endl;
    for (int i = 0; i < state_values.size(); i++) {
        file << SMLSPACE << "|";
        for (int j = 0; j < state_values[i].size(); j++){
            if (j == 0)
                file << MEDSPACE << state_values[i][j];
            else
                file << BIGSPACE << state_values[i][j];
        }
        file << "|" << std::endl;
    }
    file << "  |--------------------------|  \n" << std::endl;
}

/**
 * Prints the policy actions
 **/
void VIAgent::print_policy_actions(std::ostream &file) {
    file << "  |--------------------------|  " << std::endl;
    file << "  |----- Policy Actions -----|" << std::endl;
    file << "  |--------------------------|  " << std::endl;
            
    for (int i = 0; i < policy.size(); i++) {
        file << SMLSPACE << "|";
        for (int j = 0; j < policy[i].size(); j++) {
            for (int k = 0; k < policy[i][j].size(); k++){
                if (j == 0)
                    file << MEDSPACE << get_action_text(policy[i][j][k]);
                else
                    file << BIGSPACE << get_action_text(policy[i][j][k]);
            }
        }
        file << "|" << std::endl;
    }
    file << "  |--------------------------|  \n" << std::endl;
}

/**
 * Returns the text of the action (e.g 0 -> UP)
 **/
std::string VIAgent::get_action_text(int a){
    switch(a){
        case UP:
            return "UP";
        case DOWN:
            return "DOWN";
        case LEFT:
            return "LEFT";
        case RIGHT:
            return "RIGHT";
        default:
            return "STAY";
    }
}

/**
 * Getter for the number of iterations
 **/
int VIAgent::get_number_of_iter(){
    return number_of_iter;
}