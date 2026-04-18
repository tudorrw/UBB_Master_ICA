package com.example.mas.agents.ambulance;

import com.example.mas.MapModel;
import jade.core.AID;
import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.core.behaviours.TickerBehaviour;
import jade.domain.DFService;
import jade.domain.FIPAAgentManagement.DFAgentDescription;
import jade.domain.FIPAAgentManagement.ServiceDescription;
import jade.domain.FIPAException;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.MessageTemplate;

import java.util.ArrayList;
import java.util.List;

public class AmbulanceAgent extends Agent {
    private GpsSystem gpsSystem = new GpsSystem();
    private boolean isMoving = false;
    private List<GpsSystem.Node> currentPathNodes;
    private int currentStepIdx = 0;

    private int originalBid;

    protected void setup() {
        Object[] args = getArguments();
        int startX = (int) args[0];
        int startY = (int) args[1];

        MapModel.ambulancePositions.put(getLocalName(), new int[]{startX, startY});

        DFAgentDescription dfd = new DFAgentDescription();
        dfd.setName(getAID());
        ServiceDescription sd = new ServiceDescription();
        sd.setType("ambulance-service");
        sd.setName(getLocalName());
        dfd.addServices(sd);
        try {
            DFService.register(this, dfd);
        } catch (FIPAException fe) { fe.printStackTrace(); }

        //pune bid pentru coodinator
        addBehaviour(new CyclicBehaviour() {
            public void action() {
                // CFP
                MessageTemplate mt = MessageTemplate.MatchPerformative(ACLMessage.CFP);
                ACLMessage msg = receive(mt);
                if (msg != null && !isMoving) {
                    // Compute path
                    String[] coords = msg.getContent().split(",");
                    int tx = Integer.parseInt(coords[0]), ty = Integer.parseInt(coords[1]);
                    int[] curr = MapModel.ambulancePositions.get(getLocalName());

                    List<GpsSystem.Node> path = gpsSystem.computeAStar(curr[0], curr[1], tx, ty);

                    if (path != null) {
                        ACLMessage reply = msg.createReply();
                        reply.setPerformative(ACLMessage.PROPOSE);
                        originalBid = path.size();
                        reply.setContent(String.valueOf(originalBid));
                        System.out.println(getLocalName() + ": Bidding " + originalBid + " steps for patient at [" + coords[0] + "," + coords[1] + "]");
                        send(reply);
                    }
                } else { block(); }
            }
        });

        addBehaviour(new CyclicBehaviour() {
            public void action() {
                MessageTemplate mt = MessageTemplate.MatchPerformative(ACLMessage.ACCEPT_PROPOSAL);
                ACLMessage msg = receive(mt);
                if (msg != null) {
                    startMoving(msg.getContent());
                } else { block(); }
            }
        });

        //aici e relatia cu traffic agent-ul
        //path-ul isi da update in functie de ce modificari (blockages in traffic) apar intr timp
        addBehaviour(new CyclicBehaviour() {
            public void action() {
                MessageTemplate mt = MessageTemplate.MatchOntology("Traffic-Alert");
                ACLMessage msg = receive(mt);
                if(msg != null) {
                    System.out.println(getLocalName() + " re-calculating path due to traffic blockage...");

                    int[] currentPos = MapModel.ambulancePositions.get(getLocalName());
                    GpsSystem.Node target = currentPathNodes.get(currentPathNodes.size() - 1);

                    List<GpsSystem.Node> newPath = gpsSystem.computeAStar(currentPos[0], currentPos[1], target.x, target.y);
                    if (newPath != null) {
                        // Update active path for UI and navigation
                        currentPathNodes = newPath;
                        currentStepIdx = 0;

                        List<int[]> pathCoords = new ArrayList<>();
                        for(GpsSystem.Node n : newPath) pathCoords.add(new int[]{n.x, n.y});
                        MapModel.activePaths.put(getLocalName(), pathCoords);
                    }
                } else {
                    block();
                }
            }
        });
    }

    private void startMoving(String target) {
        isMoving = true;
        String[] coords = target.split(",");
        int tx = Integer.parseInt(coords[0]), ty = Integer.parseInt(coords[1]);
        int[] curr = MapModel.ambulancePositions.get(getLocalName());

        currentPathNodes = gpsSystem.computeAStar(curr[0], curr[1], tx, ty);
        currentStepIdx = 0;
        // Store path for visual rendering
        List<int[]> pathCoords = new ArrayList<>();

        for(GpsSystem.Node n : currentPathNodes){
            pathCoords.add(new int[]{n.x, n.y});
        }

        MapModel.activePaths.put(getLocalName(), pathCoords);
        MapModel.traveledPaths.put(getLocalName(), new ArrayList<>());

        addBehaviour(new TickerBehaviour(this, 1900) {
            protected void onTick() {
                if (currentPathNodes != null && currentStepIdx < currentPathNodes.size()) {
                    GpsSystem.Node next = currentPathNodes.get(currentStepIdx++);
                    int[] newPos = new int[]{next.x, next.y};

                    // Maintain history: append current position to traveled path
                    MapModel.traveledPaths.get(getLocalName()).add(newPos);
                    MapModel.ambulancePositions.put(getLocalName(), newPos);
                } else {
                    isMoving = false;
                    MapModel.activePaths.remove(getLocalName());
                    String myTracker = getLocalName().replace("Ambulance", "TracingAgent");

                    ACLMessage report = new ACLMessage(ACLMessage.INFORM);
                    report.addReceiver(new AID(myTracker, AID.ISLOCALNAME));
                    report.setContent(getLocalName() + ":" + originalBid);
                    send(report);

                    stop();

                }
            }
        });
    }
}
