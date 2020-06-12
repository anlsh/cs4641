package assignment4.util;

import assignment4.BasicGridWorld;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;

public class ComplexTerminalFunction implements TerminalFunction {

    int[][] terminals;

    public ComplexTerminalFunction(int[][] terminals) {
        this.terminals = terminals;
    }

    @Override
    public boolean isTerminal(State s) {

        // get location of agent in next state
        ObjectInstance agent = s.getFirstObjectOfClass(BasicGridWorld.CLASSAGENT);
        int ax = agent.getIntValForAttribute(BasicGridWorld.ATTX);
        int ay = agent.getIntValForAttribute(BasicGridWorld.ATTY);

        // are they at goal location?
        for (int[] loc : terminals) {
            if (ax == loc[0] && ay == loc[1]) {
                return true;
            }
        }
        return false;
    }

}
