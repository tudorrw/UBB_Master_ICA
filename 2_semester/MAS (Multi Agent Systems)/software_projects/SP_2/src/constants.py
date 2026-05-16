# Window
SCREEN_SCALE_W = 0.55
SCREEN_SCALE_H = 0.65 
FPS = 60
TITLE = "(Ultimate) Tic-Tac-Toe"


# Colours
C_BG = (15, 20, 35)
C_PANEL = (25, 38, 62)
C_PANEL_HOVER = (35, 50, 90)
C_BORDER = (50, 80, 120)
C_BORDER_HOVER = (0, 200, 180)
C_TEXT = (220, 235, 245)
C_TEXT_DIM = (100, 130, 160)
C_ACCENT = (0, 200, 180)
C_ACCENT_HOVER = (0, 240, 210)


# Font
FONT_PATH = "assets/fonts/dystopian.otf"

#SVGs
SVG_PATHS = {
    "startingBackground": "assets/images/startingBackground.svg",
    "playBackground": "assets/images/playBackground.svg",
}

#Buttons


# Game modes
MODE_PVA = "player_vs_agent_normal"
MODE_AVA  = "agent_vs_agent_normal"       
MODE_PVP  = "player_vs_player_ultimate" 
MODE_AVAU = "agent_vs_agent_ultimate"

MODES = [MODE_PVA, MODE_AVA, MODE_PVP, MODE_AVAU]
 
MODE_LABELS = {
    MODE_PVA:  ("Player vs Agent",  "Normal Tic-Tac-Toe"),
    MODE_AVA:  ("Agent vs Agent",   "Normal Tic-Tac-Toe"),
    MODE_PVP:  ("Player vs Player", "Ultimate Tic-Tac-Toe"),
    MODE_AVAU: ("Agent vs Agent",   "Ultimate Tic-Tac-Toe"),
}

# Players
X = "X"
O = "O"