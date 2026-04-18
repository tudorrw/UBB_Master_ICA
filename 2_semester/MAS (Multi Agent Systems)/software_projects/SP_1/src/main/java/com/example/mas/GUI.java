package com.example.mas;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import java.util.*;

public class GUI extends Application {
    private TerrainView terrainView;
    private final Map<String, Label> ambulanceStatusLabels = new HashMap<>();
    private final VBox logContainer = new VBox(5);
    private Timer timer;

    private static GUI instance;

    public GUI() {
        instance = this;
    }

    @Override
    public void start(Stage primaryStage) {
        // --- ROOT LAYOUT (Horizontal) ---
        HBox mainRoot = new HBox(20);
        mainRoot.setStyle("-fx-padding: 20; -fx-background-color: #1e1e1e;");

        // left side- map
        VBox mapSection = new VBox(10);
        Label mapHeader = new Label("City Emergency Map");
        mapHeader.setStyle("-fx-text-fill: white; -fx-font-size: 18px; -fx-font-weight: bold;");


        terrainView = new TerrainView();
        terrainView.initializeTerrain();

        mapSection.getChildren().addAll(mapHeader, terrainView);

        // dashboard & logs
        VBox dashboard = new VBox(15);
        dashboard.setMinWidth(300);
        dashboard.setStyle("-fx-background-color: #2d2d2d; -fx-padding: 15; -fx-background-radius: 10;");

        Label dashHeader = new Label("Agent Dispatch Monitor");
        dashHeader.setStyle("-fx-text-fill: #00ff00; -fx-font-weight: bold;");

        // Ambulance Status Section
        VBox statusBox = new VBox(8);
        Label statusTitle = new Label("Ambulance Locations:");
        statusTitle.setStyle("-fx-text-fill: lightgray; -fx-font-style: italic;");
        statusBox.getChildren().add(statusTitle);

        // We create labels for the ambulances (m units) [cite: 25]
        for (int i = 1; i <= MapModel.NUM_AMBULANCES; i++) {
            Label lbl = new Label("Ambulance_" + i + ": Calculating...");
            lbl.setStyle("-fx-text-fill: #3498db;");
            ambulanceStatusLabels.put("Ambulance_" + i, lbl);
            statusBox.getChildren().add(lbl);
        }

        // Real-time Logs (Auction results, traces, etc.)
        Label logTitle = new Label("Mission Logs:");
        logTitle.setStyle("-fx-text-fill: lightgray; -fx-font-style: italic;");
        ScrollPane scrollLogs = new ScrollPane(logContainer);
        scrollLogs.setPrefHeight(300);
        scrollLogs.setStyle("-fx-background: #1e1e1e; -fx-border-color: #444;");

        dashboard.getChildren().addAll(dashHeader, new Separator(), statusBox, new Separator(), logTitle, scrollLogs);

        // Combine both sides
        mainRoot.getChildren().addAll(mapSection, dashboard);

        Scene scene = new Scene(mainRoot, 1000, 750);
        primaryStage.setTitle("Ambulance Multi-Agent System (PAGES Model)");
        primaryStage.setScene(scene);
        primaryStage.show();

        timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                Platform.runLater(() -> {
                    updateTerrain();
                });
            }
        }, 0, 1000);
    }

    private void updateTerrain() {
        terrainView.drawTerrain(); // Redraw map with current positions

        for (String id : ambulanceStatusLabels.keySet()) {
            int[] pos = MapModel.ambulancePositions.get(id);
            if (pos != null) {
                ambulanceStatusLabels.get(id).setText(id + " Pos: (" + pos[0] + "," + pos[1] + ")");
            }
        }
    }

    public static void addLog(String message, String color) {
        if (instance == null) return;
        Platform.runLater(() -> {
            Label label = new Label("> " + message);
            label.setStyle("-fx-text-fill: " + color + "; -fx-font-size: 11px;");
            instance.logContainer.getChildren().add(0, label); // Newest on top
        });
    }
    @Override
    public void stop() {
        if (timer != null) timer.cancel();
        System.exit(0);
    }
    public static void main(String[] args) throws InterruptedException{
        new Thread(() -> ConfigApp.main(args)).start();
        Thread.sleep(5000);
        launch(args);
    }
}
