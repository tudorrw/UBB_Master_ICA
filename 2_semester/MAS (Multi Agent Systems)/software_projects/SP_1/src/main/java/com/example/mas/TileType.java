package com.example.mas;

public enum TileType {
    ROAD(0),
    OBSTACLE(1),
    START(2),
    END(3);

    private final int value;

    TileType(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public static TileType fromInt(int value) {
        for (TileType t : TileType.values()) {
            if (t.value == value) return t;
        }
        return ROAD;
    }
}