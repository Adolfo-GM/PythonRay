# PythonRay.py - A raycasting engine in Python
# Author: Adolfo GM
# Date: 2024 DEC 15
# ========================================================

import numpy as np
import pygame as pg
import turtle

wn = turtle.Screen()
wn.title("Adolfo GM")

def main():
    size = 25
    posx, posy, rot = (1, np.random.randint(1, size - 1), 1)
    mapc, maph, mapr, ex, ey = mazeGenerator(posx, posy, size)
    width = 320
    mod = width / 60
    height = int(width * 0.75)

    running = True
    pg.init()
    wn.listen()

    font = pg.font.SysFont("Arial", 18)
    pg.mouse.set_visible(False)
    pg.mouse.set_pos([320, 240])
    screen = pg.display.set_mode((800, 600))
    clock = pg.time.Clock()

    gradient = np.linspace(0, 1, int(height / 2 - 1))
    sky = np.asarray([gradient * 0 + 0.53, gradient * 0 + 0.81, gradient * 0 + 0.98]).T
    floor = np.asarray([gradient * 0 + 0.13, gradient * 0.55, gradient * 0.13]).T

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                running = False

        pixels = np.zeros([height, width, 3])

        for i in range(width):
            rot_i = rot + np.deg2rad(i / mod - 30)
            pixels[0:len(sky), i] = sky
            pixels[int(height / 2):int(height / 2) + len(floor), i] = floor

            x, y = (posx, posy)
            sin, cos = (0.04 * np.sin(rot_i) / mod, 0.04 * np.cos(rot_i) / mod)

            n, half = 0, None
            c, h, x, y, n, half, ty, tc = caster(x, y, i / mod, ex, ey, maph, mapc, sin, cos, n, half, mod)
            if mapr[int(x)][int(y)]:
                pixels, ty, tc = reflection(x, y, i, ex, ey, maph, mapc, sin, cos, n, c, h, half, pixels, ty, tc, height, mod)
            else:
                pixels[int((height - h * height) / 2):int((height + h * height) / 2), i] = [0.5, 0.5, 0.5]
                if half is not None:
                    pixels[int(height / 2):int((height + half[0] * height) / 2), i] = half[1]
            if len(ty) > 0:
                ty = (np.asarray(ty) * height / 2 + height / 2).astype(int)
                ty2, ind = np.unique(ty, return_index=True)
                pixels[ty2, i] = (np.asarray(tc)[ind] / 2 + pixels[ty2, i]) / 2

        if int(posx) == ex and int(posy) == ey:
            break

        pressed_keys = pg.key.get_pressed()
        posx, posy, rot = movement(pressed_keys, posx, posy, rot, maph, clock.tick() / 500)
        pg.mouse.set_pos([320, 240])

        surf = pg.surfarray.make_surface(np.rot90(pixels * 255).astype('uint8'))
        surf = pg.transform.scale(surf, (800, 600))
        screen.blit(surf, (0, 0))
        fps = font.render(str(int(clock.get_fps())), 1, pg.Color("coral"))
        screen.blit(fps, (10, 0))
        pg.display.update()

    pg.quit()

def mazeGenerator(x, y, size):
    mapc = np.random.uniform(0, 1, (size, size, 3))
    mapr = np.random.choice([0, 0, 0, 0, 1], (size, size))
    maph = np.random.choice([0, 0, 0, 0, 1], (size, size))
    maph[0, :], maph[size - 1, :], maph[:, 0], maph[:, size - 1] = (1, 1, 1, 1)
    mapc[x][y], maph[x][y], mapr[x][y] = (0, 0, 0)
    count = 0
    while True:
        testx, testy = (x, y)
        if np.random.uniform() > 0.5:
            testx += np.random.choice([-1, 1])
        else:
            testy += np.random.choice([-1, 1])
        if testx > 0 and testx < size - 1 and testy > 0 and testy < size - 1:
            if maph[testx][testy] == 0 or count > 5:
                count = 0
                x, y = (testx, testy)
                mapc[x][y], maph[x][y], mapr[x][y] = (0, 0, 0)
                if x == size - 2:
                    ex, ey = (x, y)
                    break
            else:
                count += 1
    return mapc, maph, mapr, ex, ey

def movement(pressed_keys, posx, posy, rot, maph, et):
    x, y = (posx, posy)
    if pressed_keys[pg.K_UP] or pressed_keys[ord('w')]:
        x, y = (x + et * np.cos(rot), y + et * np.sin(rot))
    if pressed_keys[pg.K_DOWN] or pressed_keys[ord('s')]:
        x, y = (x - et * np.cos(rot), y - et * np.sin(rot))
    if pressed_keys[pg.K_LEFT] or pressed_keys[ord('a')]:
        rot += 0.1 
    if pressed_keys[pg.K_RIGHT] or pressed_keys[ord('d')]:
        rot -= 0.1 
    if rot < 0:
        rot += 2 * np.pi
    if rot > 2 * np.pi:
        rot -= 2 * np.pi

    if maph[int(x)][int(y)] == 0:
        posx, posy = (x, y)
    return posx, posy, rot

def caster(x, y, i, ex, ey, maph, mapc, sin, cos, n, half, mod):
    zz = 1 if half is None else 0.5
    x, y, n, tc, ty = fastRay(x, y, zz, cos, sin, maph, n, i, ex, ey, mod)
    h, c = shader(n, maph, mapc, sin, cos, x, y, i, mod)
    if maph[int(x)][int(y)] == 0.5 and half is None:
        half = [h, c, n]
        x, y, n, tc2, ty2 = fastRay(x, y, 1, cos, sin, maph, n, i, ex, ey, mod)
        ty, tc = ty + ty2, tc + tc2
        h, c = shader(n, maph, mapc, sin, cos, x, y, i, mod)
    return (c, h, x, y, n, half, ty, tc)

def fastRay(x, y, z, cos, sin, maph, n, i, ex, ey, mod):
    ty, tc = [], []
    while True:
        n += 1
        x, y = x + cos, y + sin
        if maph[int(x)][int(y)] >= z:
            break
    return x, y, n, tc, ty

def shader(n, maph, mapc, sin, cos, x, y, i, mod):
    h = np.clip(1 / (0.04 / mod * n * np.cos(np.deg2rad(i / mod - 30))), 0, 1)
    c = [0.5, 0.5, 0.5]
    return h, c

def reflection(x, y, i, ex, ey, maph, mapc, sin, cos, n, c, h, half, pixels, ty, tc, height, mod):
    hor = int(height / 2)
    hh = int((h * height) / 2)
    pixels[hor - hh:hor + hh, i] = np.add(pixels[hor - hh:hor + hh, i], [c] * (hh * 2)) / 2
    if maph[int(x + cos)][int(y - sin)] != 0:
        cos = -cos
    else:
        sin = -sin
    c2, h2, x, y, n2, half2, ty2, tc2 = caster(x, y, i, ex, ey, maph, mapc, sin, cos, n, half, mod)
    ty, tc = ty + ty2, tc + tc2
    hh = int((h2 * height) / 2)
    if half2 is not None and half is None:
        hh = int((half2[0] * height) / 2)
        pixels[hor:hor + hh, i] = (c + half2[1]) / 2
    elif half is not None:
        hh = int((half[0] * height) / 2)
        pixels[hor:hor + hh, i] = half[1]
    return pixels, ty, tc

if __name__ == '__main__':
    main()
