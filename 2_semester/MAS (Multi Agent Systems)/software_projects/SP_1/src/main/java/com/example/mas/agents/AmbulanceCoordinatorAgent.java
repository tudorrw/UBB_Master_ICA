package com.example.mas.agents;

import com.example.mas.MapModel;
import com.example.mas.entities.Patient;

import jade.core.Agent;
import jade.core.behaviours.TickerBehaviour;
import jade.domain.DFService;
import jade.domain.FIPAAgentManagement.DFAgentDescription;
import jade.domain.FIPAAgentManagement.ServiceDescription;
import jade.domain.FIPAException;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.MessageTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class AmbulanceCoordinatorAgent extends Agent {
    protected void setup() {
        addBehaviour(new TickerBehaviour(this, 5000) {
            protected void onTick() {

                boolean allPatientsAssigned = true;
                for (Patient p : MapModel.patients) {
                    if (!p.isAssigned) {
                        allPatientsAssigned = false;
                        break;
                    }
                }

                if (allPatientsAssigned) {
                    System.out.println("AC: All patients assigned. Mission coordination complete. Terminating...");
                    stop();
                    myAgent.doDelete();
                    return;
                }


                List<Patient> remainingPatients = new ArrayList<>(MapModel.patients);
                Random rand = new Random();

                while (!remainingPatients.isEmpty()) {
                    int index = rand.nextInt(remainingPatients.size());
                    Patient p = remainingPatients.get(index);
                    remainingPatients.remove(index);
                    if(!p.isAssigned) {
                        // 1. Perceive: Find unassigned patient
                        System.out.println("AC: Initiating auction for patient at " + p.x + "," + p.y);

                        ACLMessage cfp = new ACLMessage(ACLMessage.CFP);
                        cfp.setContent(p.x + "," + p.y);

                        int expectedBids = 0;
                        try {
                            DFAgentDescription template = new DFAgentDescription();
                            ServiceDescription sd = new ServiceDescription();
                            sd.setType("ambulance-service");
                            template.addServices(sd);
                            DFAgentDescription[] result = DFService.search(myAgent, template);
                            for (DFAgentDescription agent : result) {
                                cfp.addReceiver(agent.getName());
                                expectedBids++;
                            }
                        } catch (FIPAException fe) { fe.printStackTrace(); }

                        send(cfp);
                        // 2. Reason: Select best bidder
                        ACLMessage bestPropose = null;
                        int minSteps = Integer.MAX_VALUE;

                        for(int i = 0; i < expectedBids; i++) {
                            ACLMessage propose = blockingReceive(MessageTemplate.MatchPerformative(ACLMessage.PROPOSE), 1000);

                            if (propose != null) {
                                int currentBid = Integer.parseInt(propose.getContent());
                                System.out.println("AC: Received bid from " + propose.getSender().getLocalName() + " (" + currentBid + " steps)");

                                if (currentBid < minSteps) {
                                    minSteps = currentBid;
                                    bestPropose = propose;
                                }
                            }
                        }
                        // 3. Act: Assign the winning ambulance
                        if (bestPropose != null) {
                            System.out.println("AC: WINNER is " + bestPropose.getSender().getLocalName() + " with MINIMUM " + minSteps + " steps.");
                            ACLMessage accept = bestPropose.createReply();
                            accept.setPerformative(ACLMessage.ACCEPT_PROPOSAL); // [cite: 58]
                            accept.setContent(p.x + "," + p.y);
                            send(accept);

                            p.isAssigned = true;
                            p.assignedAmbulance = bestPropose.getSender().getLocalName();
                            System.out.println("AC: Assigned " + p.assignedAmbulance + " to patient.\n");
                        }
                        break;
                    }
                }
            }
        });
    }

    @Override
    protected void takeDown() {
        System.out.println("Coordinator Agent is shutting down.");
    }
}
