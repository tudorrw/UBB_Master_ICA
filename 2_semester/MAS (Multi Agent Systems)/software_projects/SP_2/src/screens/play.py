import pygame

from src.constants import C_BG, C_TEXT, C_TEXT_DIM, C_ACCENT, MODE_LABELS, FONT_PATH
from src.utils.assets import get_font
from src.utils.button import Button


class PlayScreen:
    def __init__(self, app):
        self.app = app
        sw, sh = app.size
        self._sw, self._sh = sw, sh

        self._f_title = get_font(52, FONT_PATH)
        self._f_sub   = get_font(28, FONT_PATH)
        self._f_note  = get_font(22, FONT_PATH)

        self._back = Button(
            "BACK", (int(sw * 0.12), int(sh * 0.93)),
            get_font(26, FONT_PATH), color=C_TEXT_DIM, hover_color=C_TEXT,
        )

    # Lifecycle

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._back.is_hovered(event.pos):
                self.app.set_screen("mode_select")
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.set_screen("mode_select")

    def update(self, dt: float): pass

    def draw(self, surface: pygame.Surface):
        surface.fill(C_BG)
        sw, sh = self._sw, self._sh
        cx = sw // 2

        mode = self.app.selected_mode
        top_label, sub_label = MODE_LABELS[mode]

        # Mode title
        title_surf = self._f_title.render(top_label, True, C_ACCENT)
        surface.blit(title_surf, title_surf.get_rect(center=(cx, int(sh * 0.28))))

        sub_surf = self._f_sub.render(sub_label, True, C_TEXT)
        surface.blit(sub_surf, sub_surf.get_rect(center=(cx, int(sh * 0.38))))

        # "Coming soon" note
        note_surf = self._f_note.render("Game board coming in the next step…", True, C_TEXT_DIM)
        surface.blit(note_surf, note_surf.get_rect(center=(cx, int(sh * 0.58))))

        mp = pygame.mouse.get_pos()
        self._back.draw(surface, mp)