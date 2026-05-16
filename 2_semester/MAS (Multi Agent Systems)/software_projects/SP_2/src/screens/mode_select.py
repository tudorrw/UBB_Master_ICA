import pygame

from src.constants import (
    C_BG, C_PANEL, C_PANEL_HOVER, C_BORDER, C_BORDER_HOVER,
    C_TEXT, C_TEXT_DIM, C_ACCENT, FONT_PATH,
    MODES, MODE_LABELS, FONT_PATH
)
from src.utils.assets import get_font, load_svg
from src.utils.button import Button


class ModeSelectScreen:
    def __init__(self, app):
        self.app = app
        sw, sh = app.size

        bg_svg = load_svg("assets/images/startingBackground.svg", sw, sh)
        self._bg = bg_svg

        # Title
        self._title = get_font(64, FONT_PATH).render("SELECT MODE", True, C_ACCENT)
        self._title_rect = self._title.get_rect(center=(sw // 2, int(sh * 0.11)))

        # 2×2 card grid
        card_w  = int(sw * 0.37)
        card_h  = int(sh * 0.26)
        gap_x   = int(sw * 0.08)
        gap_y   = int(sh * 0.07)
        total_w = card_w * 2 + gap_x
        total_h = card_h * 2 + gap_y
        ox      = (sw - total_w) // 2
        oy      = int(sh * 0.26)

        self._cards: list[tuple[pygame.Rect, str]] = []
        for i, mode in enumerate(MODES):
            col = i % 2
            row = i // 2
            x = ox + col * (card_w + gap_x)
            y = oy + row * (card_h + gap_y)
            self._cards.append((pygame.Rect(x, y, card_w, card_h), mode))

        # Fonts for card labels
        self._f_top = get_font(26, FONT_PATH)
        self._f_sub = get_font(18, FONT_PATH)

        # Back button
        self._back = Button(
            "BACK", (int(sw * 0.12), int(sh * 0.93)),
            get_font(26, FONT_PATH), color=C_TEXT_DIM, hover_color=C_TEXT,
        )

    # Helpers

    def _hovered_idx(self, mp) -> int | None:
        for i, (rect, _) in enumerate(self._cards):
            if rect.collidepoint(mp):
                return i
        return None

    # Lifecycle

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mp = event.pos
            for rect, mode in self._cards:
                if rect.collidepoint(mp):
                    self.app.selected_mode = mode
                    self.app.set_screen("play")
                    return
            if self._back.is_hovered(mp):
                self.app.set_screen("menu")

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.set_screen("menu")

    def update(self, dt: float): pass

    def draw(self, surface: pygame.Surface):
        if self._bg:
            surface.blit(self._bg, (0, 0))
        else:
            surface.fill(C_BG)

        surface.blit(self._title, self._title_rect)

        mp = pygame.mouse.get_pos()
        hi = self._hovered_idx(mp)


        for i, (rect, mode) in enumerate(self._cards):
            hovered = i == hi
            fill   = C_PANEL_HOVER if hovered else C_PANEL
            border = C_BORDER_HOVER if hovered else C_BORDER

            card_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(card_surf, (*fill, 230), card_surf.get_rect(), border_radius=14)
            pygame.draw.rect(card_surf, (*border, 255), card_surf.get_rect(),
                             width=2, border_radius=14)
            surface.blit(card_surf, rect.topleft)

            # Labels
            top_label, sub_label = MODE_LABELS[mode]
            t_col = C_ACCENT if hovered else C_TEXT
            s_col = C_TEXT   if hovered else C_TEXT_DIM

            top_surf = self._f_top.render(top_label, True, t_col)
            sub_surf = self._f_sub.render(sub_label, True, s_col)

            top_rect = top_surf.get_rect(center=(rect.centerx, rect.centery - 16))
            sub_rect = sub_surf.get_rect(center=(rect.centerx, rect.centery + 18))

            surface.blit(top_surf, top_rect)
            surface.blit(sub_surf, sub_rect)

        self._back.draw(surface, mp)