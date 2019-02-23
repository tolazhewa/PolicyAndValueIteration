#ifndef ENVIRONMENT_CLASS
#define ENVIRONMENT_CLASS

#include <vector>
#include <iostream>

/**
 * The defines the mapping between actions to numerical values
 **/
#define UP      0
#define DOWN    1
#define LEFT    2
#define RIGHT   3

/**
 * Enviroment class representing the attributes of the game played
 **/
class Environment{
    private:
        /**
         * p1: Probability of transitioning to the next state
         * p2: Probability of staying in the current state
         * discount: Discount rate (value of future rewards)
         * actions: All actions available
         * action_rewards: Reward for each action
         **/
        double p1, p2, discount;
        std::vector<int> actions;
        std::vector<double> action_rewards;
    
    public:
        /**
         * Environment Constructor
         **/
        Environment(double p1, double p2, double rup, double rdown, double rleft, double rright, double discount = 0.95); 
        
        /**
         * Getters
         **/
        std::vector<int>    get_actions();
        std::vector<double> get_action_rewards();
        double              get_discount();
        double              get_p1();
        double              get_p2();
};

#endif