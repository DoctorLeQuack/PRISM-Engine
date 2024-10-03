import os
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)

import pygame as pg
from game import Game
from icons import ICON

pg.init() 

pg.display.set_icon(ICON)
pg.display.set_caption('ULTRA Engine')

ultra = Game()                  

ultra.main()