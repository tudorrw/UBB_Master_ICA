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

    @Override
    public void start(Stage primaryStage) {
        // --- ROOT LAYOUT (Horizontal) ---
        HBox mainRoot = new HBox(20);
        mainRoot.setStyle("-fx-padding: 20; -fx-background-color: #1e1e1e;");

        // left side- map
        VBox mapSection = new VBox(10);
        Label mapHeader = new Label("City Emergency Map");
        mapHeader.setStyle("-fx-text-fill: white; -fx-font-size: 18px; -fx-font-weight: bold;");

        MapModel.initialize();
        terrainView = new TerrainView();
        terrainView.drawTerrain();

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
        for (int i = 1; i <= 3; i++) {
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
                    updateUI();
                });
            }
        }, 0, 500);
    }

    private void updateUI() {
        // Redraw the grid to show movement
        //terrainView.drawTerrain();

        // Update the status labels using our shared MapModel
        // This is where we "Perceive" the Real-time Position (Gt)
        for (String id : ambulanceStatusLabels.keySet()) {
            // In a real implementation, you'd pull actual X,Y from MapModel
            // ambulanceStatusLabels.get(id).setText(id + " Pos: (" + x + "," + y + ")");
        }
    }

    @Override
    public void stop() {
        if (timer != null) timer.cancel();
        System.exit(0);
    }
    public static void main(String[] args) {
        new Thread(() -> {
            try {
                // MainApp.main(args);
            } catch (Exception e) { e.printStackTrace(); }
        }).start();

        launch(args);
    }
}
