import pygame

class Button:
    def __init__(self, text: str, pos: tuple[int, int], font: pygame.font.Font, color=(220, 235, 245), hover_color=(0, 240, 210),
        image: pygame.Surface | None = None,
    ):
        self.text = text
        self.font = font
        self.color = color
        self.hover_color = hover_color
        self.image = image
        self.pos = pos

        self._render(color)

    def _render(self, colour):
        self._surf = self.font.render(self.text, True, colour)
        if self.image:
            self.rect = self.image.get_rect(center=self.pos)
        else:
            self.rect = self._surf.get_rect(center=self.pos)
        self._text_rect = self._surf.get_rect(center=self.pos)


    def is_hovered(self, mouse_pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos)

    def draw(self, surface: pygame.Surface, mouse_pos: tuple[int, int]):
        colour = self.hover_color if self.is_hovered(mouse_pos) else self.color
        self._render(colour)
        if self.image:
            surface.blit(self.image, self.rect)
        surface.blit(self._surf, self._text_rect)