package com.example.mas.agents.ambulance;

import com.example.mas.MapModel;
import com.example.mas.TileType;

import java.util.*;


public class GpsSystem {

    public List<Node> computeAStar(int startX, int startY, int targetX, int targetY) {
        PriorityQueue<Node> openSet = new PriorityQueue<>(Comparator.comparingDouble(n -> n.f));
        Map<String, Node> allNodes = new HashMap<>();

        Node startNode = new Node(startX, startY);
        startNode.g = 0;
        startNode.h = calculateHeuristic(startX, startY, targetX, targetY);
        startNode.f = startNode.g + startNode.h;
        openSet.add(startNode);
        allNodes.put(startX + "," + startY, startNode);
        while (!openSet.isEmpty()) {
            Node current = openSet.poll();

            if (current.x == targetX && current.y == targetY) {
                return reconstructPath(current);
            }

            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;

                    int nextX = current.x + dx;
                    int nextY = current.y + dy;

                    if (isValid(nextX, nextY)) {
                        double weight = (dx != 0 && dy != 0) ? 1.41 : 1.0;
                        double tentativeG = current.g + weight;

                        Node neighbor = allNodes.getOrDefault(nextX + "," + nextY, new Node(nextX, nextY));

                        if (tentativeG < neighbor.g) {
                            neighbor.parent = current;
                            neighbor.g = tentativeG;
                            neighbor.h = calculateHeuristic(nextX, nextY, targetX, targetY);
                            neighbor.f = neighbor.g + neighbor.h;

                            if (!openSet.contains(neighbor)) openSet.add(neighbor);
                            allNodes.put(nextX + "," + nextY, neighbor);
                        }
                    }
                }
            }
        }
        return null;
    }

    private double calculateHeuristic(int x1, int y1, int x2, int y2) {
        double dx = Math.abs(x1 - x2);
        double dy = Math.abs(y1 - y2);
        return (dx + dy) + (Math.sqrt(2) - 2) * Math.min(dx, dy);
    }

    private boolean isValid(int x, int y) {
        return x >= 0 && x < MapModel.GRID_SIZE && y >= 0 && y < MapModel.GRID_SIZE
                && MapModel.grid[x][y] != TileType.OBSTACLE.getValue()
                && !MapModel.isBlocked(x, y);
    }

    private List<Node> reconstructPath(Node node) {
        List<Node> path = new ArrayList<>();
        while (node != null) {
            path.add(0, node);
            node = node.parent;
        }
        return path;
    }
    static class Node {
        int x, y;
        double g = Double.MAX_VALUE, h, f;
        Node parent;
        Node(int x, int y) { this.x = x; this.y = y; }
    }

}
