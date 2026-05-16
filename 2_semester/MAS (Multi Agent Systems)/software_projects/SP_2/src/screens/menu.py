import sys
import pygame

from src.constants import C_BG, C_ACCENT_HOVER, FONT_PATH, SVG_PATHS
from src.utils.assets import get_font, load_svg, load_image
from src.utils.button import Button


class MenuScreen:
    def __init__(self, app):
        self.app = app
        sw, sh = app.size

        bg_svg = load_svg(SVG_PATHS["startingBackground"], sw, sh)
        self._bg = bg_svg

        cx = sw // 2
        self._title = get_font(90, FONT_PATH).render("MAIN MENU", True, "#dceced")
        self._title_rect = self._title.get_rect(center=(cx, int(sh * 0.14)))

        f = get_font(58, FONT_PATH)
        gap = int(sh * 0.24)
        y0  = int(sh * 0.36)

        play_img = load_image("assets/images/PlayBox.png")
        options_img = load_image("assets/images/OptionsBox.png")
        quit_img = load_image("assets/images/QuitBox.png")

        self._buttons = [
            Button("PLAY",(cx, y0), f, image=play_img,hover_color=C_ACCENT_HOVER),
            Button("QUIT", (cx, y0 + gap), f, image=quit_img, hover_color=C_ACCENT_HOVER),
        ]

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mp = event.pos
            label = next((b.text for b in self._buttons if b.is_hovered(mp)), None)
            if label == "PLAY":
                self.app.set_screen("mode_select")
            elif label == "QUIT":
                pygame.quit(); sys.exit()

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit(); sys.exit()

    def update(self, dt: float): pass

    def draw(self, surface: pygame.Surface):
        if self._bg:
            surface.blit(self._bg, (0, 0))
        else:
            surface.fill(C_BG)

        surface.blit(self._title, self._title_rect)

        mp = pygame.mouse.get_pos()
        for btn in self._buttons:
            btn.draw(surface, mp)