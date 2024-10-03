from math import tan, atan

PI = 3.14159265358979323846
PI_2 = PI / 2
PI_4 = PI / 4
TAU = PI * 2

SCREEN_WIDTH = int(1080)
SCREEN_WIDTH_2 = SCREEN_WIDTH // 2
SCREEN_HEIGHT = int(675)
SCREEN_HEIGHT_2 = SCREEN_HEIGHT // 2
SCREEN_RATIO_WIDTH_I = (SCREEN_HEIGHT/SCREEN_WIDTH) - 1
SCREEN_RATIO_WIDTH = SCREEN_HEIGHT/SCREEN_WIDTH
SCREEN_RATIO_HEIGHT = SCREEN_WIDTH/SCREEN_HEIGHT

WINDOW_WIDTH = int(1080)
WINDOW_HEIGHT = int(675)

SCALED_SCREEN_WIDTH = SCREEN_WIDTH
SCALED_SCREEN_HEIGHT = int(SCREEN_WIDTH * (WINDOW_HEIGHT / WINDOW_WIDTH))
SCALED_SCREEN_WIDTH_2 = SCALED_SCREEN_WIDTH / 2
SCALED_SCREEN_HEIGHT_2 = SCALED_SCREEN_HEIGHT / 2

ZNEAR = 0.000000001
ZFAR = 128

HFOV = PI_2
HFOV_2 = HFOV / 2
VFOV = 2 * atan(tan(HFOV_2) / SCREEN_RATIO_HEIGHT)
VFOV_2 = VFOV / 2
PROJECTION_PLANE_DISTANCE_H = (SCREEN_WIDTH_2 / tan(HFOV_2))
PROJECTION_PLANE_DISTANCE_V = (SCREEN_WIDTH_2 / tan(VFOV_2))
MAX_CAMERA_PITCH = SCREEN_HEIGHT - SCREEN_WIDTH

GRAVITY = -9.8
COLLISION_THRESHOLD = 0.1

TEXTURE_MAX = 256
SPRITE_MAX = 128
SECTOR_MAX = 128
SKYBOX_MAX = 128
WALL_MAX = 128
FONT_MAX = 16
ENVIRONMENT_ANIMATION_MAX = 128
ACTOR_ANIMATION_MAX = 128
BILLBOARD_MAX = 128
ACTOR_MAX = 128
VERTEX_MAX = WALL_MAX*2
QUEUE_MAX = 128
PATH_MAX = 128
ANIMATION_FRAMES_MAX = 256
MASKED_MAX = 128

SCREEN_PLAYER_ANGLE_LENGTH = 1

SCREEN_VERTEX_SIZE = 2
SCREEN_VERTEX_SELECT_RANGE = 2

SCREEN_BILLBOARD_SIZE = 4
SCREEN_BILLBOARD_SELECT_RANGE = 8

SCREEN_SELECT_ICON_NORMAL = None
SCREEN_MOVE_ICON_NORMAL = None