package com.example.mas;

import javafx.scene.Group;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Line;
import javafx.scene.shape.Rectangle;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;


public class TerrainView extends Pane {
    private final int CELL_SPACING = 24;
    private final int NODE_SIZE = 12;
    private final int OFFSET = 20; // Top/Left padding

    private final Group staticLayer = new Group();
    private final Group dynamicLayer = new Group();

    public TerrainView() {
        this.getChildren().addAll(staticLayer, dynamicLayer);
        this.setStyle("-fx-background-color: #000000;");
    }
    public void initializeTerrain() {
        staticLayer.getChildren().clear();
        // 1. Draw the "Road" lines (Edges) first so they appear behind nodes
        for (int i = 0; i < MapModel.GRID_SIZE; i++) {
            for (int j = 0; j < MapModel.GRID_SIZE; j++) {
                double x1 = i * CELL_SPACING + OFFSET;
                double y1 = j * CELL_SPACING + OFFSET;

                // Horizontal connection
                if (i < MapModel.GRID_SIZE - 1) { // Horizontal
                    staticLayer.getChildren().add(createLine(x1, y1, x1 + CELL_SPACING, y1));
                }
                if (j < MapModel.GRID_SIZE - 1) { // Vertical
                    staticLayer.getChildren().add(createLine(x1, y1, x1, y1 + CELL_SPACING));
                }

                if (i < MapModel.GRID_SIZE - 1 && j < MapModel.GRID_SIZE - 1) {
                    // Down-Right Diagonal (\)
                    staticLayer.getChildren().add(createLine(x1, y1, x1 + CELL_SPACING, y1 + CELL_SPACING));

                    // Down-Left Diagonal (/)
                    double x2 = (i + 1) * CELL_SPACING + OFFSET;
                    staticLayer.getChildren().add(createLine(x2, y1, x1, y1 + CELL_SPACING));
                }
                //squares representing the states of the paths
                Rectangle node = new Rectangle(NODE_SIZE, NODE_SIZE);
                node.setX(i * CELL_SPACING + 20 - (NODE_SIZE / 2.0));
                node.setY(j * CELL_SPACING + 20 - (NODE_SIZE / 2.0));

                TileType tile = TileType.fromInt(MapModel.grid[i][j]);

                switch (tile) {
                    case OBSTACLE:
                        node.setFill(Color.GRAY);
                        break;
                    case START:
                        node.setFill(Color.LIME);
                        break;
                    case END:
                        node.setFill(Color.RED);
                        break;
                    case ROAD:
                        node.setFill(Color.BLUE);
                        break;
                }

                staticLayer.getChildren().add(node);
            }

        }
        double size = (MapModel.GRID_SIZE - 1) * CELL_SPACING + 2 * OFFSET;
        this.setPrefSize(size, size);
    }

    private Line createLine(double x1, double y1, double x2, double y2) {
        Line line = new Line(x1, y1, x2, y2);
        line.setStroke(Color.DARKBLUE);
        line.setStrokeWidth(1);
        return line;
    }

    private void drawActiveAmbulances() {
        for (String name : MapModel.ambulancePositions.keySet()) {
            int[] pos = MapModel.ambulancePositions.get(name);
            Rectangle amb = new Rectangle(NODE_SIZE + 4, NODE_SIZE + 4);
            amb.setX(pos[0] * CELL_SPACING + OFFSET - ((NODE_SIZE + 4) / 2.0));
            amb.setY(pos[1] * CELL_SPACING + OFFSET - ((NODE_SIZE + 4) / 2.0));
            amb.setFill(Color.CYAN);
            amb.setStroke(Color.WHITE);
            dynamicLayer.getChildren().add(amb);
        }
    }

    private void drawBlockages() {
        for (int[] pos : MapModel.activeBlockages.values()) {
            Rectangle block = new Rectangle(NODE_SIZE, NODE_SIZE);
            block.setX(pos[0] * CELL_SPACING + OFFSET - (NODE_SIZE / 2.0));
            block.setY(pos[1] * CELL_SPACING + OFFSET - (NODE_SIZE / 2.0));
            block.setFill(Color.DARKGOLDENROD); // Darker gray for dynamic blockages
            dynamicLayer.getChildren().add(block);
        }
    }

    private void drawPaths(ConcurrentHashMap<String, List<int[]>> paths, Color color, int strokeLineWidth) {
        for (String name : paths.keySet()) {
            List<int[]> path = paths.get(name);
            for (int i = 0; i < path.size() - 1; i++) {
                int[] p1 = path.get(i);
                int[] p2 = path.get(i+1);
                Line line = new Line(p1[0]*CELL_SPACING+OFFSET, p1[1]*CELL_SPACING+OFFSET,
                        p2[0]*CELL_SPACING+OFFSET, p2[1]*CELL_SPACING+OFFSET);
                line.setStroke(color);
                line.setStrokeWidth(strokeLineWidth);
                dynamicLayer.getChildren().add(line);
            }
        }
    }

    private void drawPlannedTrajectories() {
        drawPaths(MapModel.activePaths, Color.YELLOW, 2);
    }

    private void drawTraveledPaths() {
        drawPaths(MapModel.traveledPaths, Color.RED, 3);
    }

    public void drawTerrain() {
        dynamicLayer.getChildren().clear();
        drawBlockages();
        drawActiveAmbulances();
        drawPlannedTrajectories();
        drawTraveledPaths();
    }
}
