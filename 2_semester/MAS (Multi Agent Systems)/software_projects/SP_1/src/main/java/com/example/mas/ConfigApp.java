package com.example.mas;

public class ConfigApp {
    public static void main(String[] args) {
        jade.core.Runtime rt = jade.core.Runtime.instance();
        jade.core.Profile p = new jade.core.ProfileImpl();
        jade.wrapper.ContainerController cc = rt.createMainContainer(p);
        try {
            cc.createNewAgent("Coordinator", "com.example.mas.agents.AmbulanceCoordinatorAgent", null).start();

            // Instantiate Team Agents using MapModel as the single source of truth [cite: 19]
            MapModel.initialize();
            for (int i = 0; i < MapModel.START_POINTS.length; i++) {
                String ambName = "Ambulance_" + (i + 1);
                String trafficName = "TrafficAgent_" + (i + 1);
                String tracingName = "TracingAgent_" + (i + 1);

                Object[] startPos = { MapModel.START_POINTS[i][0], MapModel.START_POINTS[i][1] };
                cc.createNewAgent(ambName, "com.example.mas.agents.ambulance.AmbulanceAgent", startPos).start();

                cc.createNewAgent(trafficName, "com.example.mas.agents.TrafficAgent", new Object[]{ambName}).start();
                cc.createNewAgent(tracingName, "com.example.mas.agents.TracingAgent", new Object[]{ambName}).start();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
