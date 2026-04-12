package com.example.mas;

import java.util.Random;

public class MapModel {
    public static final int GRID_SIZE = 20;
    public static int[][] grid = new int[GRID_SIZE][GRID_SIZE];

    public static void initialize() {
        Random rand = new Random();
        for(int i = 0; i < GRID_SIZE; i++) {
            for(int j = 0; j < GRID_SIZE; j++) {
                grid[i][j] = (rand.nextDouble() < 0.15) ? 1 : 0;
            }
        }
        grid[2][2] = 2;
        grid[GRID_SIZE - 3][GRID_SIZE - 3] = 3;
    }


}
