package com.example.mas.agents;

import com.example.mas.MapModel;
import jade.core.Agent;
import jade.core.behaviours.TickerBehaviour;
import jade.lang.acl.ACLMessage;

import java.util.List;
import java.util.Random;
import jade.core.AID;
public class TrafficAgent extends Agent {
    private String ambulance;
    private Random rand = new Random();

    protected void setup() {
        ambulance = (String) getArguments()[0];
        addBehaviour(new TickerBehaviour(this, 4000) {
            protected void onTick() {
                if(MapModel.activePaths.containsKey(ambulance) && rand.nextDouble() < 0.4) {
                    List<int[]> path = MapModel.activePaths.get(ambulance);
                    if(path.size() >= 4) {
                        int[] blockedPoint = path.get(rand.nextInt(path.size() - 2) + 1);
                        MapModel.activeBlockages.put(ambulance, blockedPoint);

                        ACLMessage alert = new ACLMessage(ACLMessage.INFORM);
                        alert.addReceiver(new AID(ambulance, AID.ISLOCALNAME));
                        alert.setContent(blockedPoint[0] + "," + blockedPoint[1]);
                        alert.setOntology("Traffic-Alert");
                        send(alert);

                        System.out.println(getLocalName() + " reported blockage at [" +
                                blockedPoint[0] + "," + blockedPoint[1] + "] for " + ambulance);
                    }
                }
            }
        });
    }
}
