from numba import njit
from numpy import array, zeros, int32, full, float32
from renderer import pixel, extend_v, closest_point_on_line_segment, distance, get_angle, point_side
from settings import HFOV_2, WALL_MAX
from keys import ENTER, LSHIFT, UP, DOWN, LEFT, RIGHT
from colors import EDITOR_BG_COLOR, EDITOR_GRID_COLOR, EDITOR_LINE_COLOR, EDITOR_LINE_HOVERED, EDITOR_VERTEX_COLOR, EDITOR_VERTEX_HOVERED, EDITOR_VERTEX_SELECTED, EDITOR_LINE_SELECTED

@njit(fastmath=True)
def render_editor(screen:array, editor_width:int, editor_height:int, editor_x:int, editor_y:int, editor_scale:float, editor_origin_x:int, editor_origin_y:int, editor_offset_x:int, editor_offset_y:int, walls_id:array, walls_a_id:array, walls_b_id:array, walls_portal:array, walls_portal_wall_id:array, walls_sector_id:array, walls_texture_id:array, walls_animation_id:array, walls_n:int, camera_pos_x:float, camera_pos_y:float, camera_angle:float, mouse_position_x:int, mouse_position_y:int, vertices_id:array, vertices_x:array, vertices_y:array, vertices_sector:array, vertices_n:int, active_window:str, mouse_l_pressed:bool, prev_mouse_l_pressed:bool, new_vertices_id:array, new_vertices_x:array, new_vertices_y:array, new_vertices_n:int, keys:array, sectors_id:array, sectors_light_factor:array, sectors_z_floor:array, sectors_z_ceil:array, sectors_ceil_texture_id:array, sectors_ceil_animation_id:array, sectors_floor_texture_id:array, sectors_floor_animation_id:array, sectors_slope_floor_z:array, sectors_slope_floor_wall_id:array, sectors_slope_ceil_z:array, sectors_slope_ceil_wall_id:array, sectors_slope_floor_friction:array, sectors_n:int, new_walls_id:array, new_walls_a_id:array, new_walls_b_id:array, new_walls_n:int, editor_mode:str, mouse_r_pressed:bool, prev_mouse_r_pressed:bool, link_wall_start_id:int, link_wall_end_id:int, selected_vertices_id:array, selected_vertices_id_n:int, vertex_hovered_id:int, wall_hovered_id:int, select_mode:str, sectors_skybox_id:array, axis_lock:int)->tuple:
    mouse_editor_position_x, mouse_editor_position_y = mouse_position_x-editor_x, mouse_position_y-editor_y

    vertices_screen_x = zeros(WALL_MAX*2, dtype=int32)
    vertices_screen_y = zeros(WALL_MAX*2, dtype=int32)
    vertices_screen_id = zeros(WALL_MAX*2, dtype=int32)

    new_vertices_screen_x = zeros(WALL_MAX*2, dtype=int32)
    new_vertices_screen_y = zeros(WALL_MAX*2, dtype=int32)
    new_vertices_screen_id = zeros(WALL_MAX*2, dtype=int32)

    for x in range(editor_width+1):
        for y in range(editor_height+1):
            pixel(screen,x,y,EDITOR_BG_COLOR,editor_x, editor_y,editor_width,editor_height)

    grid_size = int(2*editor_scale)
    screen = grid(screen, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y, editor_width, editor_height, grid_size, editor_x, editor_y)

    if active_window == "EDITOR":
        if editor_mode == "DRAW":
            if mouse_l_pressed == True and prev_mouse_l_pressed == False:
                new_vertices_n += 1
                if vertex_hovered_id == 0:
                    new_vertex_x, new_vertex_y = world_conversion(mouse_editor_position_x, mouse_editor_position_y, editor_scale, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y)
                    new_vertices_x[new_vertices_n], new_vertices_y[new_vertices_n] = new_vertex_x, new_vertex_y
                    new_vertices_id[new_vertices_n] = new_vertices_n
                else:
                    new_vertex_x, new_vertex_y = vertices_x[vertex_hovered_id], vertices_y[vertex_hovered_id]
                    new_vertices_x[new_vertices_n], new_vertices_y[new_vertices_n] = new_vertex_x, new_vertex_y
                    new_vertices_id[new_vertices_n] = new_vertices_n

                if new_vertices_n > 1:
                    new_walls_n += 1
                    new_walls_id[new_walls_n] = new_walls_n
                    new_walls_a_id[new_walls_n] = new_vertices_n - 1
                    new_walls_b_id[new_walls_n] = new_vertices_n

                    new_vertex_a_x, new_vertex_a_y = new_vertices_x[1], new_vertices_y[1]
                    new_vertex_b_x, new_vertex_b_y = new_vertices_x[2], new_vertices_y[2]

            if mouse_r_pressed == True and prev_mouse_r_pressed == False:
                if new_vertices_n > 0:
                    new_vertices_id[new_vertices_n] = 0
                    new_vertices_screen_id[new_vertices_n] = 0
                    new_walls_a_id[new_walls_n] = 0
                    new_walls_b_id[new_walls_n] = 0
                    new_walls_id[new_walls_n] = 0
                    new_walls_n -= 1
                    new_vertices_n -= 1

            if new_vertices_n != 0:
                last_new_vertex_x, last_new_vertex_y = new_vertices_x[new_vertices_n], new_vertices_y[new_vertices_n]
                last_new_screen_vertex_x, last_new_screen_vertex_y = screen_conversion(last_new_vertex_x, last_new_vertex_y, editor_scale, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y)
                first_new_vertex_x, first_new_vertex_y = new_vertices_x[1], new_vertices_y[1]
                first_new_screen_vertex_x, first_new_screen_vertex_y = screen_conversion(first_new_vertex_x, first_new_vertex_y, editor_scale, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y)
                
                screen = line(screen, last_new_screen_vertex_x, last_new_screen_vertex_y, mouse_editor_position_x, mouse_editor_position_y, editor_width, editor_height, editor_x, editor_y, (255,255,255), 1, 3)
                screen = line(screen, first_new_screen_vertex_x, first_new_screen_vertex_y, mouse_editor_position_x, mouse_editor_position_y, editor_width, editor_height, editor_x, editor_y, (255,255,255), 1, 3)

            if keys[ENTER]:
                if new_vertices_n >= 3:
                    
                    direction = (new_vertices_x[2]*new_vertices_y[3] + new_vertices_x[1]*new_vertices_y[2] + new_vertices_y[1]*new_vertices_x[3]) - (new_vertices_y[1]*new_vertices_x[2] + new_vertices_y[2]*new_vertices_x[3] + new_vertices_x[1]*new_vertices_y[3])
                    
                    last_sector_light_factor = sectors_light_factor[sectors_n]
                    last_sector_z_floor = sectors_z_floor[sectors_n]
                    last_sector_z_ceil = sectors_z_ceil[sectors_n]
                    last_sector_ceil_texture_id = sectors_ceil_texture_id[sectors_n]
                    last_sector_ceil_animation_id = sectors_ceil_animation_id[sectors_n]
                    last_sector_floor_texture_id = sectors_floor_texture_id[sectors_n]
                    last_sector_floor_animation_id = sectors_floor_animation_id[sectors_n]
                    last_sector_slope_floor_friction = sectors_slope_floor_friction[sectors_n]
                    last_sector_skybox_id = sectors_skybox_id[sectors_n]

                    last_wall_texture_id = walls_texture_id[walls_n]
                    last_wall_animation_id = walls_animation_id[walls_n]

                    sectors_n += 1
                    sectors_id[sectors_n] = sectors_n
                    sectors_light_factor[sectors_n] = last_sector_light_factor
                    sectors_z_floor[sectors_n] = last_sector_z_floor
                    sectors_z_ceil[sectors_n] = last_sector_z_ceil
                    sectors_ceil_texture_id[sectors_n] = last_sector_ceil_texture_id
                    sectors_ceil_animation_id[sectors_n] = last_sector_ceil_animation_id
                    sectors_floor_texture_id[sectors_n] = last_sector_floor_texture_id
                    sectors_floor_animation_id[sectors_n] = last_sector_floor_animation_id
                    sectors_slope_floor_z[sectors_n] = 0
                    sectors_slope_ceil_z[sectors_n] = 0
                    sectors_slope_floor_friction[sectors_n] = last_sector_slope_floor_friction
                    sectors_skybox_id[sectors_n] = last_sector_skybox_id

                    walls_n += 1
                    walls_id[walls_n] = walls_n
                    walls_sector_id[walls_n] = sectors_n

                    sectors_slope_ceil_wall_id[sectors_n] = walls_n
                    sectors_slope_floor_wall_id[sectors_n] = walls_n

                    vertices_n += 1
                    vertices_id[vertices_n] = vertices_n
                    vertices_x[vertices_n], vertices_y[vertices_n] = new_vertices_x[new_vertices_n], new_vertices_y[new_vertices_n]
                    vertices_sector[vertices_n] = sectors_n

                    walls_a_id[walls_n] = vertices_n

                    vertices_n += 1
                    vertices_id[vertices_n] = vertices_n
                    vertices_x[vertices_n], vertices_y[vertices_n] = new_vertices_x[1], new_vertices_y[1]
                    vertices_sector[vertices_n] = sectors_n

                    walls_b_id[walls_n] = vertices_n

                    if direction > 0:
                        walls_a_id[walls_n], walls_b_id[walls_n] = walls_b_id[walls_n], walls_a_id[walls_n]

                    for new_wall_id in new_walls_id:
                        if new_wall_id != 0:

                            walls_n += 1
                            walls_id[walls_n] = walls_n
                            walls_sector_id[walls_n] = sectors_n
                            walls_portal[walls_n] = 0
                            walls_portal_wall_id[walls_n] = 0
                            walls_texture_id[walls_n] = last_wall_texture_id
                            walls_animation_id[walls_n] = last_wall_animation_id

                            new_wall_a_id, new_wall_b_id = new_walls_a_id[new_wall_id], new_walls_b_id[new_wall_id]
                            new_vertex_a_x, new_vertex_a_y = new_vertices_x[new_wall_a_id], new_vertices_y[new_wall_a_id]
                            new_vertex_b_x, new_vertex_b_y = new_vertices_x[new_wall_b_id], new_vertices_y[new_wall_b_id]

                            vertices_n += 1
                            vertices_id[vertices_n] = vertices_n
                            vertices_x[vertices_n], vertices_y[vertices_n] = new_vertex_a_x, new_vertex_a_y
                            vertices_sector[vertices_n] = sectors_n

                            walls_a_id[walls_n] = vertices_n

                            vertices_n += 1
                            vertices_id[vertices_n] = vertices_n
                            vertices_x[vertices_n], vertices_y[vertices_n] = new_vertex_b_x, new_vertex_b_y
                            vertices_sector[vertices_n] = sectors_n

                            walls_b_id[walls_n] = vertices_n

                            if direction > 0:
                                walls_a_id[walls_n], walls_b_id[walls_n] = walls_b_id[walls_n], walls_a_id[walls_n]

                            new_walls_id[new_wall_id] = 0
                            new_walls_a_id[new_wall_id] = 0
                            new_walls_b_id[new_wall_id] = 0
                            new_vertices_id[new_wall_a_id] = 0
                            new_vertices_id[new_wall_b_id] = 0
                            new_vertices_screen_id[new_wall_a_id] = 0
                            new_vertices_screen_id[new_wall_b_id] = 0

                    new_vertices_n = 0

                    sectors_slope_floor_wall_id[sectors_n] = walls_n
                    sectors_slope_ceil_wall_id[sectors_n] = walls_n

                    editor_mode = "NONE"

    if mouse_l_pressed == True and prev_mouse_l_pressed == False:
        if keys[LSHIFT] == 0:
            if selected_vertices_id_n > 0:
                for index, selected_vertex_id in enumerate(selected_vertices_id):
                    if selected_vertex_id != 0:
                        selected_vertices_id[index] = 0

                selected_vertices_id_n = 0

    vertex_hovered_id = 0

    for sector_id in sectors_id:
        if sector_id != 0:

            vertex_max_y = 0
            vertex_min_y = 0

            for vertex_id in vertices_id:
                if vertex_id != 0:
                    vertex_sector = vertices_sector[vertex_id]

                    if vertex_sector == sector_id:
                        vertices_screen_id[vertex_id] = vertex_id
                        vertex_screen_x, vertex_screen_y = screen_conversion(vertices_x[vertex_id], vertices_y[vertex_id], editor_scale, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y)
                        vertices_screen_x[vertex_id], vertices_screen_y[vertex_id] = vertex_screen_x, vertex_screen_y

                        if vertex_screen_y > vertex_max_y:
                            vertex_max_y = vertex_screen_y

                        if vertex_screen_y < vertex_min_y:
                            vertex_min_y = vertex_screen_y

            for wall_id in walls_id:
                if wall_id != 0:
                    wall_sector_id = walls_sector_id[wall_id]
                    
                    if wall_sector_id == sector_id:
                        wall_a_id, wall_b_id = walls_a_id[wall_id], walls_b_id[wall_id]

                        vertex_a_screen_x, vertex_a_screen_y = vertices_screen_x[wall_a_id], vertices_screen_y[wall_a_id]
                        vertex_b_screen_x, vertex_b_screen_y = vertices_screen_x[wall_b_id], vertices_screen_y[wall_b_id]
                        
                        line_color = EDITOR_LINE_COLOR

                        closest_point_x, closest_point_y = closest_point_on_line_segment(mouse_editor_position_x, mouse_editor_position_y, vertex_a_screen_x, vertex_a_screen_y, vertex_b_screen_x, vertex_b_screen_y)

                        if distance(mouse_editor_position_x, mouse_editor_position_y, closest_point_x, closest_point_y) <= 10:
                            line_color = EDITOR_LINE_HOVERED
                            wall_hovered_id = wall_id
                            
                            if editor_mode == "LINK":
                                if mouse_l_pressed == True and prev_mouse_l_pressed == False:
                                    if link_wall_start_id == 0:
                                        link_wall_start_id = wall_id
                                    
                                    else:
                                        if link_wall_start_id != wall_id:
                                            link_wall_end_id = wall_id

                                            link_sector_start_id = walls_sector_id[link_wall_start_id]
                                            link_sector_end_id = walls_sector_id[link_wall_end_id]

                                            walls_portal[link_wall_start_id] = link_sector_end_id
                                            walls_portal_wall_id[link_wall_start_id] = link_wall_end_id
                                            walls_texture_id[link_wall_start_id] = 0
                                            walls_portal[link_wall_end_id] = link_sector_start_id
                                            walls_portal_wall_id[link_wall_end_id] = link_wall_start_id
                                            walls_texture_id[link_wall_end_id] = 0

                                            link_wall_start_id = 0
                                            link_wall_end_id = 0
                                            editor_mode = "NONE"

                            elif editor_mode == "NONE":
                                if select_mode == "WALL":
                                    if mouse_l_pressed == True and prev_mouse_l_pressed == False:

                                        selected_vertices_id[wall_a_id] = wall_a_id
                                        selected_vertices_id_n += 1

                                        selected_vertices_id[wall_b_id] = wall_b_id
                                        selected_vertices_id_n += 1

                        if selected_vertices_id[wall_a_id] != 0 and selected_vertices_id[wall_b_id] != 0:
                            line_color = EDITOR_LINE_SELECTED

                        if wall_id == link_wall_start_id:
                            line_color = (255,255,255)

                        screen = line(screen, vertex_a_screen_x, vertex_a_screen_y, vertex_b_screen_x, vertex_b_screen_y, editor_width, editor_height, editor_x, editor_y, line_color, 0.5, 2)

                        wall_portal = walls_portal[wall_id]
                        if wall_portal != 0:
                            wall_middle_x, wall_middle_y = line_middle(vertex_a_screen_x, vertex_a_screen_y, vertex_b_screen_x, vertex_b_screen_y)
                            wall_angle = get_angle(vertex_a_screen_x, vertex_a_screen_y, vertex_b_screen_x, vertex_b_screen_y)
                            wall_portal_a_x, wall_portal_a_y = extend_v(wall_middle_x, wall_middle_y, 8, wall_angle)
                            wall_portal_b_x, wall_portal_b_y = extend_v(wall_middle_x, wall_middle_y, -8, wall_angle)
                            screen = line(screen, int(wall_portal_a_x), int(wall_portal_a_y), int(wall_portal_b_x), int(wall_portal_b_y), editor_width, editor_height, editor_x, editor_y, (255,0,0), 1, 2)

                        vertices_screen_x[wall_a_id], vertices_screen_y[wall_a_id], vertices_screen_id[wall_a_id] = vertex_a_screen_x, vertex_a_screen_y, wall_a_id
                        vertices_screen_x[wall_b_id], vertices_screen_y[wall_b_id], vertices_screen_id[wall_b_id]  = vertex_b_screen_x, vertex_b_screen_y, wall_b_id

            for vertex_screen_id in vertices_screen_id:
                if vertex_screen_id != 0:
                    vertex_sector_id = vertices_sector[vertex_screen_id]

                    if vertex_sector_id == sector_id:
                        vertex_screen_x, vertex_screen_y = vertices_screen_x[vertex_screen_id], vertices_screen_y[vertex_screen_id]

                        vertex_color = EDITOR_VERTEX_COLOR
                        
                        if distance(mouse_editor_position_x, mouse_editor_position_y, vertex_screen_x, vertex_screen_y) <= 10:
                            vertex_color = EDITOR_VERTEX_HOVERED
                            vertex_hovered_id = vertex_screen_id

                            if editor_mode == "NONE":
                                if select_mode == "VERTEX":
                                    if mouse_l_pressed == True and prev_mouse_l_pressed == False:

                                        selected_vertices_id[vertex_screen_id] = vertex_screen_id
                                        selected_vertices_id_n += 1
                                
                        if selected_vertices_id[vertex_screen_id] != 0 and active_window == "EDITOR":
                            vertex_x, vertex_y = vertices_x[vertex_screen_id], vertices_y[vertex_screen_id]
                            if keys[UP] == 1:
                                vertex_y -= 0.1
                            if keys[DOWN] == 1:
                                vertex_y += 0.1
                            if keys[LEFT] == 1:
                                vertex_x -= 0.1
                            if keys[RIGHT] == 1:
                                vertex_x += 0.1
                            
                            vertices_x[vertex_screen_id], vertices_y[vertex_screen_id] = vertex_x, vertex_y

                        if selected_vertices_id[vertex_screen_id] != 0:
                            vertex_color = EDITOR_VERTEX_SELECTED

                        circle(screen, vertex_screen_x, vertex_screen_y, 5, editor_width, editor_height, editor_x, editor_y, vertex_color)

    for new_wall_id in new_walls_id:
        if new_wall_id != 0:
            new_wall_a_id, new_wall_b_id = new_walls_a_id[new_wall_id], new_walls_b_id[new_wall_id]
            new_vertex_a_x, new_vertex_a_y = new_vertices_x[new_wall_a_id], new_vertices_y[new_wall_a_id]
            new_vertex_b_x, new_vertex_b_y = new_vertices_x[new_wall_b_id], new_vertices_y[new_wall_b_id]

            new_vertex_a_screen_x, new_vertex_a_screen_y = screen_conversion(new_vertex_a_x, new_vertex_a_y, editor_scale, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y)
            new_vertex_b_screen_x, new_vertex_b_screen_y = screen_conversion(new_vertex_b_x, new_vertex_b_y, editor_scale, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y)

            screen = line(screen, new_vertex_a_screen_x, new_vertex_a_screen_y, new_vertex_b_screen_x, new_vertex_b_screen_y, editor_width, editor_height, editor_x, editor_y, EDITOR_LINE_COLOR, 0.5, 5)

            new_vertices_screen_id[new_wall_a_id] = new_wall_a_id
            new_vertices_screen_x[new_wall_a_id], new_vertices_screen_y[new_wall_a_id] = new_vertex_a_screen_x, new_vertex_a_screen_y
            new_vertices_screen_id[new_wall_b_id] = new_wall_b_id
            new_vertices_screen_x[new_wall_b_id], new_vertices_screen_y[new_wall_b_id] = new_vertex_b_screen_x, new_vertex_b_screen_y

    for new_vertex_screen_id in new_vertices_screen_id:
        if new_vertex_screen_id != 0:
            new_vertex_screen_x, new_vertex_screen_y = new_vertices_screen_x[new_vertex_screen_id], new_vertices_screen_y[new_vertex_screen_id]
            
            circle(screen, new_vertex_screen_x, new_vertex_screen_y, 5, editor_width, editor_height, editor_x, editor_y, (255,255,255))

    camera_screen_pos_x, camera_screen_pos_y = screen_conversion(camera_pos_x, camera_pos_y, editor_scale, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y)

    view_frustrum_l_x, view_frustrum_l_y = extend_v(camera_pos_x, camera_pos_y, 2, camera_angle-HFOV_2)
    view_frustrum_r_x, view_frustrum_r_y = extend_v(camera_pos_x, camera_pos_y, 2, camera_angle+HFOV_2)
    view_frustrum_l_screen_x, view_frustrum_l_screen_y = screen_conversion(view_frustrum_l_x, view_frustrum_l_y, editor_scale, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y)
    view_frustrum_r_screen_x, view_frustrum_r_screen_y = screen_conversion(view_frustrum_r_x, view_frustrum_r_y, editor_scale, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y)

    screen = line(screen, camera_screen_pos_x, camera_screen_pos_y, view_frustrum_l_screen_x, view_frustrum_l_screen_y, editor_width, editor_height, editor_x, editor_y, (0,0,0))
    screen = line(screen, camera_screen_pos_x, camera_screen_pos_y, view_frustrum_r_screen_x, view_frustrum_r_screen_y, editor_width, editor_height, editor_x, editor_y, (0,0,0))
    screen = line(screen, view_frustrum_l_screen_x, view_frustrum_l_screen_y, view_frustrum_r_screen_x, view_frustrum_r_screen_y, editor_width, editor_height, editor_x, editor_y, (0,0,0))
    circle(screen, camera_screen_pos_x, camera_screen_pos_y, 5, editor_width, editor_height, editor_x, editor_y, (255,0,255))

    return screen, new_vertices_id, new_vertices_x, new_vertices_y, new_vertices_n, vertices_id, vertices_x, vertices_y, vertices_sector, vertices_n, sectors_id, sectors_light_factor, sectors_z_floor, sectors_z_ceil, sectors_ceil_texture_id, sectors_ceil_animation_id, sectors_floor_texture_id, sectors_floor_animation_id, sectors_slope_floor_z, sectors_slope_floor_wall_id, sectors_slope_ceil_z, sectors_slope_ceil_wall_id, sectors_slope_floor_friction, sectors_n, new_walls_id, new_walls_a_id, new_walls_b_id, new_walls_n, walls_n, editor_mode, link_wall_start_id, link_wall_end_id, selected_vertices_id, selected_vertices_id_n, vertex_hovered_id, wall_hovered_id

@njit(fastmath=True)
def screen_conversion(_v_x: float, _v_y: float, _s: float, _o_x: float, _o_y: float, _or_x: float, _or_y: float) -> tuple:

    screen_x = int(_v_x * _s + _or_x + _o_x)
    screen_y = int(_v_y * _s + _or_y + _o_y)
    
    return screen_x, screen_y

@njit(fastmath=True)
def world_conversion(screen_x: int, screen_y: int, _s: float, _o_x: float, _o_y: float, _or_x: float, _or_y: float) -> tuple:

    translated_x = (screen_x - _o_x - _or_x)
    translated_y = (screen_y - _o_y - _or_y)
    
    _v_x = translated_x / _s
    _v_y = translated_y / _s
    
    return _v_x, _v_y

@njit(fastmath=True)
def line_middle(_a_x, _a_y, _b_x, _b_y)->tuple:
    return (_a_x + _b_x)/2, (_a_y + _b_y) /2

@njit(fastmath=True)
def color_blend(_oc, _c, _a):
    return ((1 - _a) * _oc[0] + _a * _c[0], (1 - _a) * _oc[1] + _a * _c[1], (1 - _a) * _oc[2] + _a * _c[2])

@njit(fastmath=True)
def circle(screen:array, center_x:int, center_y:int, radius:int, dimensions_x:int, dimensions_y:int, offset_x:int, offset_y:int, color:tuple):
    for y in range(dimensions_y):
        for x in range(dimensions_x):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                pixel(screen,x,y,color,offset_x,offset_y,dimensions_x,dimensions_y)

@njit(fastmath=True)
def line(screen:array, _v_x:int, _v_y:int, _v1_x:int, _v1_y:int, dimensions_x:int, dimensions_y:int, offset_x:int, offset_y:int, color:tuple, _a=1, thickness=1)->array:
    if _a != 0:
        dx = abs(_v1_x - _v_x)
        dy = abs(_v1_y - _v_y)
        sx = 1 if _v_x < _v1_x else -1
        sy = 1 if _v_y < _v1_y else -1
        err = dx - dy
        thickness_2 = thickness // 2

        while True:
            for i in range(-thickness_2, thickness_2 + 1):
                for j in range(-thickness_2, thickness_2 + 1):
                    x, y = _v_x + i, _v_y + j

                    if _a == 1:
                        pixel(screen, x, y, color, offset_x, offset_y, dimensions_x, dimensions_y)
                    else:
                        current_color = screen[x + offset_x, y + offset_y]
                        blended_color = color_blend(current_color, color, _a)
                        pixel(screen, x, y, blended_color, offset_x, offset_y, dimensions_x, dimensions_y)

            if _v_x == _v1_x and _v_y == _v1_y:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                _v_x += sx
            if e2 < dx:
                err += dx
                _v_y += sy

    return screen

@njit(fastmath=True)
def grid(screen:array, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y, editor_width, editor_height, grid_size, editor_x, editor_y):

    for i in range(editor_width):
        grid_pos_x, grid_pos_y = world_conversion(i, 0, 1, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y)

        if grid_pos_x % grid_size == 0:
            screen = line(screen, i, 0, i, editor_height, editor_width, editor_height, editor_x, editor_y, EDITOR_GRID_COLOR, 1, 1)

    for i in range(editor_height):
        grid_pos_x, grid_pos_y = world_conversion(0, i, 1, editor_offset_x, editor_offset_y, editor_origin_x, editor_origin_y)

        if grid_pos_y % grid_size == 0:
            screen = line(screen, 0, i, editor_width, i, editor_width, editor_height, editor_x, editor_y, EDITOR_GRID_COLOR, 1, 1)

    return screen