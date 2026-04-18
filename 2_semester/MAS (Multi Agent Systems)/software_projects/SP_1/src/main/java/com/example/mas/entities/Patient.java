package com.example.mas.entities;

public class Patient {
    public int x, y;
    public String name;
    public boolean isAssigned = false;
    public String assignedAmbulance = null;

    public Patient(String name, int x, int y) {
        this.name = name;
        this.x = x;
        this.y = y;
    }
}
