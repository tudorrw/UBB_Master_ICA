import io, os, cairosvg
import pygame



def get_font(size: int, path: str) -> pygame.font.Font:
    return pygame.font.Font(path, size)



def load_svg(path: str, w: int, h: int) -> pygame.Surface | None:
    data = cairosvg.svg2png(url=path, output_width=w, output_height=h)
    return pygame.image.load(io.BytesIO(data))


def load_image(path: str) -> pygame.Surface | None:
    return pygame.image.load(path)