import sys
import pygame

from src.constants import SCREEN_SCALE_W, SCREEN_SCALE_H, FPS, TITLE, MODE_PVA
from src.screens.menu  import MenuScreen
from src.screens.mode_select import ModeSelectScreen
from src.screens.play import PlayScreen

class App:
    def __init__(self):
        pygame.init()

        info = pygame.display.Info()
        w = int(info.current_w * SCREEN_SCALE_W)
        h = int(info.current_h * SCREEN_SCALE_H)
        self.size = (w, h)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(TITLE)

        self.clock = pygame.time.Clock()
        self.selected_mode = MODE_PVA   # default, overwritten by mode_select

        # Screens are built on demand and cached
        self._cache: dict = {}
        self._active: str = ""
        self.set_screen("menu")

    # Screen management

    def set_screen(self, name: str):
        """Switch to a screen by name, building it if not yet cached."""
        if name not in self._cache:
            self._cache[name] = self._build(name)
        self._active = name

    def _build(self, name: str):
        return {
            "menu": lambda: MenuScreen(self),
            "mode_select": lambda: ModeSelectScreen(self),
            "play": lambda: PlayScreen(self),
        }[name]()

    @property
    def _screen(self):
        return self._cache[self._active]

    # Main loop
    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                self._screen.handle_event(event)

            self._screen.update(dt)
            self._screen.draw(self.screen)
            pygame.display.flip()