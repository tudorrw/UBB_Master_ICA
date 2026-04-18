package com.example.mas.agents;

import com.example.mas.GUI;
import com.example.mas.MapModel;
import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.MessageTemplate;

import java.util.List;

public class TracingAgent extends Agent {
    private String ambulance;
    protected void setup() {
        ambulance = (String) getArguments()[0];
        addBehaviour(new CyclicBehaviour() {
            public void action() {
                MessageTemplate mt = MessageTemplate.MatchPerformative(ACLMessage.INFORM);
                ACLMessage msg = receive(mt);
                if (msg != null) {
                    String[] data = msg.getContent().split(":");
                    int originalBid = Integer.parseInt(data[1]);

                    // History check from MapModel
                    int actualSteps = MapModel.traveledPaths.get(ambulance).size();

                    double efficiency = (double) originalBid / actualSteps;
                    String status = efficiency >= 1.0 ? "PERFECT" : (efficiency > 0.8 ? "GOOD" : "DELAYED");

                    String report = String.format("[%s Trace] Efficiency: %.2f | Result: %s",
                            ambulance, efficiency, status);

                    // Color code by performance
                    String hexColor = (efficiency >= 0.9) ? "#3498db" : (efficiency > 0.7 ? "#f1c40f" : "#e74c3c");
                    GUI.addLog(report, hexColor);
                } else { block(); }
            }
        });
    }
}
