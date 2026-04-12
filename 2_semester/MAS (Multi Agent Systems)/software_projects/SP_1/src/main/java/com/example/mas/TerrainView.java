package com.example.mas;

import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Line;
import javafx.scene.shape.Rectangle;

public class TerrainView extends Pane {
    private final int CELL_SPACING = 30;
    private final int NODE_SIZE = 12;
    private final int OFFSET = 20; // Top/Left padding

    public void drawTerrain() {
        this.getChildren().clear();
        this.setStyle("-fx-background-color: #000000;"); // Black background like the image

        // 1. Draw the "Road" lines (Edges) first so they appear behind nodes
        for (int i = 0; i < MapModel.GRID_SIZE; i++) {
            for (int j = 0; j < MapModel.GRID_SIZE; j++) {
                double x = i * CELL_SPACING + OFFSET;
                double y = j * CELL_SPACING + OFFSET;

                // Horizontal connection
                if (i < MapModel.GRID_SIZE - 1) {
                    Line hLine = new Line(x, y, x + CELL_SPACING, y);
                    hLine.setStroke(Color.DARKSLATEGRAY);
                    this.getChildren().add(hLine);
                }
                // Vertical connection
                if (j < MapModel.GRID_SIZE - 1) {
                    Line vLine = new Line(x, y, x, y + CELL_SPACING);
                    vLine.setStroke(Color.DARKSLATEGRAY);
                    this.getChildren().add(vLine);
                }
            }
        }

        // 2. Draw the Nodes (Cells)
        for (int i = 0; i < MapModel.GRID_SIZE; i++) {
            for (int j = 0; j < MapModel.GRID_SIZE; j++) {
                Rectangle node = new Rectangle(NODE_SIZE, NODE_SIZE);
                node.setX(i * CELL_SPACING + 20 - (NODE_SIZE / 2.0));
                node.setY(j * CELL_SPACING + 20 - (NODE_SIZE / 2.0));

                switch (MapModel.grid[i][j]) {
                    case 1:
                        node.setFill(Color.GRAY);      // Obstacle
                        break;
                    case 2:
                        node.setFill(Color.LIME);      // Start
                        break;
                    case 3:
                        node.setFill(Color.ORANGERED); // End
                        break;
                    default:
                        node.setFill(Color.BLUE);      // Actual Road
                        break;
                }
                this.getChildren().add(node);
            }
        }
        double totalWidth = (MapModel.GRID_SIZE - 1) * CELL_SPACING + 2 * OFFSET;
        double totalHeight = (MapModel.GRID_SIZE - 1) * CELL_SPACING + 2 * OFFSET;
        this.setPrefSize(totalWidth, totalHeight);
    }
}
