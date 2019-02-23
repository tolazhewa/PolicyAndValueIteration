#include "Environment.hpp"

/**
 * Constructor setting values
 **/
Environment::Environment(double p1, double p2, double rup, double rdown, double rleft, double rright, double discount)
: action_rewards({rup, rdown, rleft, rright})
, p1(p1)
, p2(p2)
, discount(discount)
, actions({UP, DOWN, LEFT, RIGHT})
{}

/**
 * Returns the available actions
 **/
std::vector<int> Environment::get_actions(){
    return actions;
}

/**
 * Returns the discount rate
 **/
double Environment::get_discount(){
    return discount;
}

/**
 * Returns the action rewards
 **/
std::vector<double> Environment::get_action_rewards(){
    return action_rewards;
}

/**
 * Returns probability of transitioning to the next state
 **/
double Environment::get_p1(){
    return p1;
}

/**
 * Returns probability of staying in the current state
 **/
double Environment::get_p2(){
    return p2;
}