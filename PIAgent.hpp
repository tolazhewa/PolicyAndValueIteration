#ifndef PI_AGENT_CLASS
#define PI_AGENT_CLASS

#include <vector>
#include <iomanip>
#include <string>
#include <ctime>
#include <cstdlib>

#include "Environment.hpp"

/**
 * The defines the mapping between actions to numerical values
 **/
#define UP      0
#define DOWN    1
#define LEFT    2
#define RIGHT   3

// Output spacing support
#define BIGSPACE std::setw(7) 
#define MEDSPACE std::setw(5)
#define SMLSPACE std::setw(3)

/**
 * Policy Iteration Agent class
 **/
class PIAgent {
    private:
        // The acceptable bound for the algorithm
        double theta;
        // Values of each state v(s)
        std::vector<std::vector<double>> state_values;
        // Policy for each space (pi)
        std::vector<std::vector<std::vector<int>>> policy;
        // Environment to work with
        Environment env;
        // Number of iterations
        int number_of_iter;

    public:
        /**
         * Constructor
         **/
        PIAgent(Environment env, double theta = 0.001);

        // Policy Iteration
        void policy_iter();

        // return the max value of a vector
        double max(std::vector<double>);

        // returns the absolute value
        double abs(double);

        // returns the max argument
        double argmax(std::vector<double>);

        // Calculates the action-state value
        double calc_action_state_value(int, int);

        // Returns next state given action
        std::vector<int> get_next_state(int, int, int);

        // Produces the actions based off probabilities of the environment
        std::vector<std::vector<double>> prod_prob_arr(int,int,int);

        // Validates the state (checks bounds)
        bool validate_state(std::vector<int>);

        // Prints the state values
        void print_state_values(std::ostream &file = std::cout);

        // Prints the policy actions
        void print_policy_actions(std::ostream &file = std::cout);

        // Getter for the number of iterations
        int get_number_of_iter();

        // Returns the greedy action for a state
        int get_greedy_action(int,int);

        // Returns the action for a state (probabilistic)
        int get_action(int,int);

        // Returns the text of the action (e.g 0 -> UP)
        std::string get_action_text(int);
};

#endif