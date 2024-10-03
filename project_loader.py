from numpy import array, zeros, full, uint8, int32, float32
from settings import TEXTURE_MAX, SKYBOX_MAX, SPRITE_MAX, PATH_MAX, SECTOR_MAX, WALL_MAX, ENVIRONMENT_ANIMATION_MAX, ANIMATION_FRAMES_MAX
from pygame import surfarray, image

MODE_TEXTURES = "[TEXTURES_START]"
MODE_TEXTURES_END = "[TEXTURES_END]"
MODE_FONTS = "[FONTS_START]"
MODE_FONTS_END = "[FONTS_END]"
MODE_ANIMATIONS = "[ANIMATIONS_START]"
MODE_ANIMATIONS_END = "[ANIMATIONS_END]"
MODE_MAPS = "[MAPS_START]"
MODE_MAPS_END = "[MAPS_END]"

SUBMODE_TEXTURES = "<TEXTURES_START>"
SUBMODE_TEXTURES_END = "<TEXTURES_END>"
SUBMODE_ENVIRONMENT_ANIMATIONS = "<ENVIRONMENT_ANIMATIONS_START>"
SUBMODE_ENVIRONMENT_ANIMATIONS_END = "<ENVIRONMENT_ANIMATIONS_END>"
SUBMODE_SPRITE_ANIMATIONS = "<SPRITE_ANIMATIONS_START>"
SUBMODE_SPRITE_ANIMATIONS_END = "<SPRITE_ANIMATIONS_END>"
SUBMODE_SPRITES = "<SPRITES_START>"
SUBMODE_SPRITES_END = "<SPRITES_END>"
SUBMODE_MAP = "<MAP_START="
SUBMODE_MAP_END = "<MAP_END="
SUBMODE_FONT = "<FONT_START="
SUBMODE_FONT_END = "<FONT_END="
SUBMODE_SKYBOX = "<SKYBOX_START>"
SUBMODE_SKYBOX_END = "<SKYBOX_END>"

SCAN_TEXTURES = "SCAN_TEXTURES"
SCAN_FONTS = "SCAN_FONTS"
SCAN_ANIMATIONS = "SCAN_ANIMATIONS"
SCAN_MAPS = "SCAN_MAPS"

SUBSCAN_MAP = "SUBSCAN_MAP"
SUBSCAN_FONT = "SUBSCAN_FONT"
SUBSCAN_TEXTURES = "SUBSCAN_TEXTURES"
SUBSCAN_SPRITES = "SUBSCAN_SPRITES"
SUBSCAN_SKYBOX = "SUBSCAN_SKYBOX"

SUBSCAN_MAP_INFO = "SUBSCAN_MAP_INFO"
SUBSCAN_MAP_SECTORS = "SUBSCAN_MAP_SECTORS"
SUBSCAN_MAP_VERTEX = "SUBSCAN_MAP_VERTEX"
SUBSCAN_MAP_WALLS = "SUBSCAN_MAP_WALLS"
SUBSCAN_MAP_ACTORS = "SUBSCAN_MAP_ACTORS"
SUBSCAN_MAP_THINGS = "SUBSCAN_MAP_THINGS"

SUBSCAN_FONT_INFO = "SUBSCAN_FONT_INFO"
SUBSCAN_FONT_KEY = "SUBSCAN_FONT_KEY"

SUBMODE_INFO = "(INFO)"
SUBMODE_INFO_END = "(/INFO)"
SUBMODE_SECTORS = "(SECTORS)"
SUBMODE_SECTORS_END = "(/SECTORS)"
SUBMODE_VERTEX = "(VERTEX)"
SUBMODE_VERTEX_END = "(/VERTEX)"
SUBMODE_WALLS = "(WALLS)"
SUBMODE_WALLS_END = "(/WALLS)"
SUBMODE_ACTORS = "(ACTORS)"
SUBMODE_ACTORS_END = "(/ACTORS)"
SUBMODE_THINGS = "(THINGS)"
SUBMODE_THINGS_END = "(/THINGS)"
SUBMODE_KEYS = "(KEYS)"
SUBMODE_KEYS_END = "(/KEYS)"

COMMENT_SIGN = "//"

EMPTY = "EMPTY"

def load_file(path:str)->str:
    try:
        with open(path, 'r') as _f:
            file = _f.read()
            print(f"INFO: LOADING {path} WAS SUCCESFUL")

            return file

    except FileNotFoundError:
        print(f"ERROR 001: LOADING {path} WAS UNSUCCESFUL")
        print(f"INFO: PLEASE CHECK PATH")

        return ""
    
def save_file(path: str, data: str) -> bool:
    try:
        with open(path, 'w') as _f:
            _f.write(data)
            print(f"INFO: SAVING TO {path} WAS SUCCESSFUL")
            return True

    except Exception as e:
        print(f"ERROR 002: SAVING TO {path} WAS UNSUCCESSFUL")
        print(f"INFO: ERROR MESSAGE - {str(e)}")
        return False

def load_project(path:str, map_id:int=1)->tuple:
    file = load_file(path)
    file_lines = file.splitlines()

    ss = EMPTY
    sm = EMPTY
    sw = EMPTY
    sr = EMPTY

    current_map_id = -1
    current_font_id = -1

    sectors_id = full(SECTOR_MAX, 0, dtype=int32)
    sectors_light_factor = full(SECTOR_MAX, 0, dtype=float32)
    sectors_z_floor = full(SECTOR_MAX, 0, dtype=float32)
    sectors_z_ceil = full(SECTOR_MAX, 0, dtype=float32)
    sectors_ceil_texture_id = full(SECTOR_MAX, 0, dtype=int32)
    sectors_ceil_animation_id = full(SECTOR_MAX, 0, dtype=int32)
    sectors_floor_texture_id = full(SECTOR_MAX, 0, dtype=int32)
    sectors_floor_animation_id = full(SECTOR_MAX, 0, dtype=int32)
    sectors_slope_floor_z = full(SECTOR_MAX, 0, dtype=float32)
    sectors_slope_floor_wall_id = full(SECTOR_MAX, 0, dtype=int32)
    sectors_slope_ceil_z = full(SECTOR_MAX, 0, dtype=float32)
    sectors_slope_ceil_wall_id = full(SECTOR_MAX, 0, dtype=int32)
    sectors_slope_floor_friction = full(SECTOR_MAX, 0, dtype=float32)
    sectors_skybox_id = full(SECTOR_MAX, 0, dtype=int32)
    sectors_n = 0

    vertices_id = full(WALL_MAX*2, 0, dtype=int32)
    vertices_x = full(WALL_MAX*2, 0, dtype=float32)
    vertices_y = full(WALL_MAX*2, 0, dtype=float32)
    vertices_sector = full(WALL_MAX*2, 0, dtype=int32)
    vertices_n = 0

    walls_id = full(WALL_MAX, 0, dtype=int32)
    walls_a_id = full(WALL_MAX, 0, dtype=int32)
    walls_b_id = full(WALL_MAX, 0, dtype=int32)
    walls_portal = full(WALL_MAX, 0, dtype=int32)
    walls_portal_wall_id = full(WALL_MAX, 0, dtype=int32)
    walls_sector_id = full(WALL_MAX, 0, dtype=int32)
    walls_animation_id = full(WALL_MAX, 0, dtype=int32)
    walls_texture_id = full(WALL_MAX, 2, dtype=int32)
    walls_texture_id_up = full(WALL_MAX, 2, dtype=int32)
    walls_texture_id_down = full(WALL_MAX, 2, dtype=int32)
    walls_texture_offset_x = full(WALL_MAX, 0, dtype=float32)
    walls_texture_offset_y = full(WALL_MAX, 0, dtype=float32)
    walls_texture_offset_up_x = full(WALL_MAX, 0, dtype=float32)
    walls_texture_offset_up_y = full(WALL_MAX, 0, dtype=float32)
    walls_texture_offset_down_x = full(WALL_MAX, 0, dtype=float32)
    walls_texture_offset_down_y = full(WALL_MAX, 0, dtype=float32)
    walls_n = 0

    textures_sheet_width = 0
    textures_sheet_height = 0
    texture_pool = [None] * TEXTURE_MAX
    textures_width = full(TEXTURE_MAX, 0, dtype=int32)
    textures_height = full(TEXTURE_MAX, 0, dtype=int32)
    textures_x_coordinates = full(TEXTURE_MAX, 0, dtype=int32)
    textures_path = full(TEXTURE_MAX, " " * PATH_MAX)
    textures_n = 0

    sprites_sheet_width = 0
    sprites_sheet_height = 0
    sprite_pool = [None] * SPRITE_MAX
    sprites_width = full(SPRITE_MAX, 0, dtype=int32)
    sprites_height = full(SPRITE_MAX, 0, dtype=int32)
    sprites_x_coordinates = full(SPRITE_MAX, 0, dtype=int32)
    sprites_path = full(SPRITE_MAX, " " * PATH_MAX)
    sprites_n = 0

    skyboxes_sheet_width = 0
    skyboxes_sheet_height = 0
    skyboxes_pool = [None] * SKYBOX_MAX
    skyboxes_width = full(SKYBOX_MAX, 0, dtype=int32)
    skyboxes_height = full(SKYBOX_MAX, 0, dtype=int32)
    skyboxes_x_coordinates = full(SKYBOX_MAX, 0, dtype=int32)
    skyboxes_path = full(SKYBOX_MAX, " " * PATH_MAX)
    skyboxes_n = 0

    environmental_animations_frames = full((ANIMATION_FRAMES_MAX, ENVIRONMENT_ANIMATION_MAX), 0, dtype=int32)
    environmental_animations_ms = full(ENVIRONMENT_ANIMATION_MAX, 0, dtype=float32)
    environmental_animations_frames_count = full(ENVIRONMENT_ANIMATION_MAX, 0, dtype=int32)

    for line in file_lines:
        line = line.strip()

        if len(line) == 0:
            continue

        if line.startswith(COMMENT_SIGN):
            continue

        if line.startswith("<") or line.startswith("[") or line.startswith("("):
            if line.startswith(MODE_TEXTURES):
                ss = SCAN_TEXTURES
            if line.startswith(MODE_TEXTURES_END):
                ss = EMPTY
            if line.startswith(MODE_FONTS):
                ss = SCAN_FONTS
            if line.startswith(MODE_FONTS_END):
                ss = EMPTY
            if line.startswith(MODE_ANIMATIONS):
                ss = SCAN_ANIMATIONS
            if line.startswith(MODE_ANIMATIONS_END):
                ss = EMPTY
            if line.startswith(MODE_MAPS):
                ss = SCAN_MAPS
            if line.startswith(MODE_MAPS_END):
                ss = EMPTY

            if line.startswith(SUBMODE_MAP):
                sm = SUBSCAN_MAP
                current_map_id = get_int_value(line)
            if line.startswith(SUBMODE_MAP_END):
                sm = EMPTY
            if line.startswith(SUBMODE_TEXTURES):
                sm = SUBSCAN_TEXTURES
            if line.startswith(SUBMODE_TEXTURES_END):
                sm = EMPTY
            if line.startswith(SUBMODE_SPRITES):
                sm = SUBSCAN_SPRITES
            if line.startswith(SUBMODE_SPRITES_END):
                sm = EMPTY
            if line.startswith(SUBMODE_SKYBOX):
                sm = SUBSCAN_SKYBOX
            if line.startswith(SUBMODE_SKYBOX_END):
                sm = EMPTY

            if line.startswith(SUBMODE_FONT):
                sm = SUBSCAN_FONT
                current_font_id = get_int_value(line)
            if line.startswith(SUBMODE_FONT_END):
                sm = EMPTY

            if ss == SCAN_MAPS:
                if line.startswith(SUBMODE_INFO):
                    sw = SUBSCAN_MAP_INFO
                if line.startswith(SUBMODE_INFO_END):
                    sw = EMPTY
                if line.startswith(SUBMODE_SECTORS):
                    sw = SUBSCAN_MAP_SECTORS
                if line.startswith(SUBMODE_SECTORS_END):
                    sw = EMPTY
                if line.startswith(SUBMODE_VERTEX):
                    sw = SUBSCAN_MAP_VERTEX
                if line.startswith(SUBMODE_VERTEX_END):
                    sw = EMPTY
                if line.startswith(SUBMODE_WALLS):
                    sw = SUBSCAN_MAP_WALLS
                if line.startswith(SUBMODE_WALLS_END):
                    sw = EMPTY
                if line.startswith(SUBMODE_ACTORS):
                    sw = SUBSCAN_MAP_ACTORS
                if line.startswith(SUBMODE_ACTORS_END):
                    sw = EMPTY
                if line.startswith(SUBMODE_THINGS):
                    sw = SUBSCAN_MAP_ACTORS
                if line.startswith(SUBMODE_THINGS_END):
                    sw = EMPTY
            elif ss == SCAN_FONTS:
                if line.startswith(SUBMODE_INFO):
                    sw = SUBSCAN_FONT_INFO
                if line.startswith(SUBMODE_INFO_END):
                    sw = EMPTY
                if line.startswith(SUBMODE_KEYS):
                    sw = SUBSCAN_FONT_KEY
                if line.startswith(SUBMODE_KEYS_END):
                    sw = EMPTY

        else:
            if ss == SCAN_TEXTURES:
                if sm == SUBSCAN_TEXTURES:
                    texture_id, texture_path = get_values(line)
                    texture_data, texture_width, texture_height = load_texture(texture_path)

                    textures_sheet_width += texture_width

                    if texture_height > textures_sheet_height:
                        textures_sheet_height = texture_height

                    texture_pool[texture_id] = [texture_id, texture_data, texture_width, texture_height]
                    textures_height[texture_id] = texture_height
                    textures_width[texture_id] = texture_width
                    textures_path[texture_id] = texture_path
                    textures_n += 1

                if sm == SUBSCAN_SPRITES:
                    sprite_id, sprite_path = get_values(line)
                    sprite_data, sprite_width, sprite_height = load_texture(sprite_path)

                    sprites_sheet_width += sprite_width

                    if sprite_height > sprites_sheet_height:
                        sprites_sheet_height = sprite_height

                    sprite_pool[sprite_id] = [sprite_id, sprite_data, sprite_width, sprite_height, sprite_path]
                    sprites_height[sprite_id] = sprite_height
                    sprites_width[sprite_id] = sprite_width
                    sprites_path[sprite_id] = sprite_path
                    sprites_n += 1

                if sm == SUBSCAN_SKYBOX:
                    skybox_id, skybox_path = get_values(line)
                    skybox_data, skybox_width, skybox_height = load_texture(skybox_path)

                    skyboxes_sheet_width += skybox_width

                    if skybox_height > skyboxes_sheet_height:
                        skyboxes_sheet_height = skybox_height

                    skyboxes_pool[skybox_id] = [skybox_id, skybox_data, skybox_width, skybox_height, skybox_path]
                    skyboxes_height[skybox_id] = skybox_height
                    skyboxes_width[skybox_id] = skybox_width
                    skyboxes_path[skybox_id] = skybox_path
                    skyboxes_n += 1

            if ss == SCAN_FONTS:
                if sm == SUBSCAN_FONT:
                    if sw == SUBSCAN_FONT_INFO:
                        print(line)

                    if sw == SUBSCAN_FONT_KEY:
                        key_id, key_name, key_data = get_values(line)

            if ss == SCAN_ANIMATIONS:
                animation_id, animation_ms, animation_frames = get_values(line)
                
                environmental_animations_ms[animation_id] = animation_ms

                animation_frames_count = 0

                for index, animation_frame in enumerate(animation_frames.split(":")):
                    environmental_animations_frames[animation_id][index] = int(animation_frame)
                    animation_frames_count += 1

                environmental_animations_frames_count[animation_id] = animation_frames_count

            if ss == SCAN_MAPS and current_map_id == map_id:
                if sm == SUBSCAN_MAP:
                    if sw == SUBSCAN_MAP_INFO:
                        print(line)
                    if sw == SUBSCAN_MAP_SECTORS:
                        sector_id, sector_z_floor, sector_z_ceil, sector_light_factor, sector_floor_texture_id, sector_ceil_texture_id, sector_floor_animation_id, sector_ceil_animation_id, sector_slope_floor_z, sector_slope_floor_wall_id, sector_slope_ceil_z, sector_slope_ceil_wall_id, sector_slope_floor_friction, sector_skybox_id = get_values(line)
                        
                        sectors_id[sector_id] = sector_id
                        sectors_light_factor[sector_id] = sector_light_factor
                        sectors_z_floor[sector_id] = sector_z_floor
                        sectors_z_ceil[sector_id] = sector_z_ceil
                        sectors_ceil_texture_id[sector_id] = sector_ceil_texture_id
                        sectors_ceil_animation_id[sector_id] = sector_ceil_animation_id
                        sectors_floor_texture_id[sector_id] = sector_floor_texture_id
                        sectors_floor_animation_id[sector_id] = sector_floor_animation_id
                        sectors_slope_floor_z[sector_id] = sector_slope_floor_z
                        sectors_slope_floor_wall_id[sector_id] = sector_slope_floor_wall_id
                        sectors_slope_ceil_z[sector_id] = sector_slope_ceil_z
                        sectors_slope_ceil_wall_id[sector_id] = sector_slope_ceil_wall_id
                        sectors_slope_floor_friction[sector_id] = sector_slope_floor_friction
                        sectors_skybox_id[sector_id] = sector_skybox_id
                        sectors_n += 1
                    
                    if sw == SUBSCAN_MAP_WALLS:
                        wall_id, wall_a_id, wall_b_id, wall_portal, wall_sector_id, wall_animation_id, wall_texture_id, wall_texture_offset_x, wall_texture_offset_y, wall_texture_id_down, wall_texture_offset_down_x, wall_texture_offset_down_y, wall_texture_id_up, wall_texture_offset_up_x, wall_texture_offset_up_y, wall_portal_wall_id = get_values(line)
                        
                        walls_id[wall_id] = wall_id
                        walls_a_id[wall_id] = wall_a_id
                        walls_b_id[wall_id] = wall_b_id
                        walls_portal[wall_id] = wall_portal
                        walls_portal_wall_id[wall_id] = wall_portal_wall_id
                        walls_sector_id[wall_id] = wall_sector_id
                        walls_animation_id[wall_id] = wall_animation_id
                        walls_texture_id[wall_id] = wall_texture_id
                        walls_texture_id_up[wall_id] = wall_texture_id_up
                        walls_texture_id_down[wall_id] = wall_texture_id_down
                        walls_texture_offset_x[wall_id] = wall_texture_offset_x
                        walls_texture_offset_y[wall_id] = wall_texture_offset_y
                        walls_texture_offset_up_x[wall_id] = wall_texture_offset_up_x
                        walls_texture_offset_up_y[wall_id] = wall_texture_offset_up_y
                        walls_texture_offset_down_x[wall_id] = wall_texture_offset_down_x
                        walls_texture_offset_down_y[wall_id] = wall_texture_offset_down_y
                        walls_n += 1

                    if sw == SUBSCAN_MAP_VERTEX:
                        vertex_id, vertex_x, vertex_y, vertex_sector = get_values(line)

                        vertices_id[vertex_id] = vertex_id
                        vertices_x[vertex_id] = vertex_x
                        vertices_y[vertex_id] = vertex_y
                        vertices_sector[vertex_id] = vertex_sector
                        vertices_n += 1
                    
                    if sw == SUBSCAN_MAP_ACTORS:
                        print(line)

                    if sw == SUBSCAN_MAP_THINGS:
                        print(line)

    textures_sheet = zeros((textures_sheet_width, textures_sheet_height, 3), dtype=uint8)

    xx = 0

    for texture in texture_pool:
        if texture:
            texture_id = texture[0]
            texture_data = texture[1]
            texture_width = texture[2]
            texture_height = texture[3]

            for x in range(texture_width):
                for y in range(texture_height):
                    textures_sheet[xx+x][y] = texture_data[x][y]

            textures_x_coordinates[texture_id] = xx

            xx += texture_width

    sprites_sheet = zeros((sprites_sheet_width, sprites_sheet_height, 3), dtype=uint8)

    xx = 0

    for sprite in sprite_pool:
        if sprite:
            sprite_id = sprite[0]
            sprite_data = sprite[1]
            sprite_width = sprite[2]
            sprite_height = sprite[3]

            for x in range(sprite_width):
                for y in range(sprite_height):
                    sprites_sheet[xx+x][y] = sprite_data[x][y]

            sprites_x_coordinates[sprite_id] = xx

            xx += sprite_width

    skyboxes_sheet = zeros((skyboxes_sheet_width, skyboxes_sheet_height, 3), dtype=uint8)

    xx = 0

    for skybox in skyboxes_pool:
        if skybox:
            skybox_id = skybox[0]
            skybox_data = skybox[1]
            skybox_width = skybox[2]
            skybox_height = skybox[3]

            for x in range(skybox_width):
                for y in range(skybox_height):
                    skyboxes_sheet[xx+x][y] = skybox_data[x][y]

            skyboxes_x_coordinates[skybox_id] = xx

            xx += skybox_width

    save_texture(skyboxes_sheet, "sky_sheet.png")

    return textures_sheet, sprites_sheet, sectors_id, sectors_light_factor, sectors_z_floor, sectors_z_ceil, sectors_ceil_texture_id, sectors_ceil_animation_id, sectors_floor_texture_id, sectors_floor_animation_id, sectors_slope_floor_z, sectors_slope_floor_wall_id, sectors_slope_ceil_z, sectors_slope_ceil_wall_id, sectors_slope_floor_friction, sectors_n, walls_id, walls_a_id, walls_b_id, walls_portal, walls_portal_wall_id, walls_sector_id, walls_animation_id, walls_texture_id, walls_texture_id_up, walls_texture_id_down, walls_texture_offset_x, walls_texture_offset_y, walls_texture_offset_up_x, walls_texture_offset_up_y, walls_texture_offset_down_x, walls_texture_offset_down_y, walls_n, textures_sheet_width, textures_sheet_height, textures_width, textures_height, textures_x_coordinates, textures_path, sprites_sheet_width, sprites_sheet_height, sprites_width, sprites_height, sprites_x_coordinates, sprites_path, environmental_animations_frames, environmental_animations_ms, environmental_animations_frames_count, vertices_id, vertices_x, vertices_y, vertices_sector, vertices_n, textures_n, sprites_n, sectors_skybox_id, skyboxes_sheet, skyboxes_height, skyboxes_width, skyboxes_x_coordinates, skyboxes_path, skyboxes_n

def get_texture(_i:int, _s:array, _c:array, _w:array, _h:array)->array:
    texture_x_coordinate = _c[_i]
    texture_height = _h[_i]
    texture_width = _w[_i]

    texture_data = zeros((texture_width, texture_height, 3), dtype=uint8)

    for x in range(texture_width):
        for y in range(texture_height):
            texture_data[x][y] = _s[texture_x_coordinate+x][y]
    
    return texture_data

def get_int_value(_s:str)->int:
    return int(_s.split('=')[1][0])

def get_string_value(_s:str)->str:
    return _s.split('=')[1][0]

def get_values(_s: str) -> tuple:
    values = []
    items = _s.strip('{}').split(';')
    
    for item in items:
        value = item.split('=')[1]

        try:
            cast_value = int(value)
        except ValueError:
            try:
                cast_value = float(value)
            except ValueError:
                cast_value = value
        
        values.append(cast_value)
    
    return tuple(values)

def get_float_value(_s:str)->float:
    return float(_s.split('=')[1][0])

def load_texture(path):
    texture_data = array(surfarray.array3d(image.load(path)), dtype=uint8)
    texture_width = int(texture_data.shape[0])
    texture_height = int(texture_data.shape[1])

    return texture_data, texture_width, texture_height

def save_texture(texture_data:array, save_path:str):

    texture_height, texture_width = texture_data.shape[:2]

    texture_surface = surfarray.make_surface(texture_data)

    image.save(texture_surface, save_path)

def save_project(path):
    pass