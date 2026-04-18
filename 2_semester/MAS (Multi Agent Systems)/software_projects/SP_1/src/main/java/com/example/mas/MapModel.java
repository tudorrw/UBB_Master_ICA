package com.example.mas;

import com.example.mas.entities.Patient;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

public class MapModel {
    public static final int GRID_SIZE = 25;
    public static int[][] grid = new int[GRID_SIZE][GRID_SIZE];

    public static final int NUM_PATIENTS = 3;
    public static final int NUM_AMBULANCES = 3;

    public static int[][] START_POINTS; // now dynamic

    public static List<Patient> patients = new ArrayList<>();

    public static ConcurrentHashMap<String, int[]> ambulancePositions = new ConcurrentHashMap<>();
    public static ConcurrentHashMap<String, List<int[]>> activePaths = new ConcurrentHashMap<>();
    public static ConcurrentHashMap<String, List<int[]>> traveledPaths = new ConcurrentHashMap<>();
    public static ConcurrentHashMap<String, int[]> activeBlockages = new ConcurrentHashMap<>();

    public static void initialize() {
        patients.clear();

        List<int[]> validRoads = new ArrayList<>();

        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                if(i == j || i % 4 == 0 || j % 4 == 0) {
                    validRoads.add(new int[]{i, j});
                }
                else {
                    grid[i][j] = TileType.OBSTACLE.getValue();
                }
            }
        }
        Random rand = new Random();
        Collections.shuffle(validRoads, rand); //randomizes the entire road lis

        int total = NUM_AMBULANCES + NUM_PATIENTS;
        if (validRoads.size() < total) {
            throw new IllegalStateException("Not enough road tiles for ambulances and patients.");
        }
        START_POINTS = new int[NUM_AMBULANCES][];
        for (int i = 0; i < NUM_AMBULANCES; i++) {
            int[] pos = validRoads.get(i);
            START_POINTS[i] = pos;
            grid[pos[0]][pos[1]] = TileType.START.getValue();
        }

        for (int i = 0; i < NUM_PATIENTS; i++) {
            int[] pos = validRoads.get(NUM_AMBULANCES + i);
            patients.add(new Patient("Patient_" + (i + 1), pos[0], pos[1]));
            grid[pos[0]][pos[1]] = TileType.END.getValue();
        }

        for (int i = total; i < validRoads.size(); i++) {
            int[] pos = validRoads.get(i);
            grid[pos[0]][pos[1]] = TileType.ROAD.getValue();
        }
    }

    public static boolean isBlocked(int x, int y) {
        for (int[] pos : activeBlockages.values()) {
            if (pos[0] == x && pos[1] == y) return true;
        }
        return false;
    }
}

