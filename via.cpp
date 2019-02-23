#include <iostream>
#include <ctime>
#include "Environment.hpp"
#include "VIAgent.hpp"

int main(int argc, char **argv) {
    // Set up random number generator seed to time
    srand(time(NULL));

    // Format standard output
    std::cout << std::fixed;
    std::cout << std::setprecision(2);

    // Declarations
    int start, end;
    double p1,p2,rup,rdown,rleft,rright;

    // Inputs
    std::cin >> p1;
    std::cin >> p2;
    std::cin >> rup;
    std::cin >> rdown;
    std::cin >> rleft;
    std::cin >> rright;

    // Declarations based off variables inputted
    Environment env(p1, p2, rup, rdown, rleft, rright);
    VIAgent agent(env);

    // Timing + iterations
    start = std::clock();
    agent.value_iter();
    end = std::clock();

    // Outputs
    std::cout << "\n\tValue Iteration\n" << std::endl;
    agent.print_state_values();
    agent.print_policy_actions();
    std::cout << "Number of iterations: " << agent.get_number_of_iter() << std::endl;
    std::cout << "Execution time: " << (end - start)/double(CLOCKS_PER_SEC) * 1000 << " ms\n";
}