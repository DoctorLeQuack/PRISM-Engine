from numba import njit
from numpy import array
from pygame import key, K_LSHIFT, K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_a, K_b, K_c, K_d, K_e, K_f, K_g, K_h, K_i, K_j, K_k, K_l, K_m, K_n, K_o, K_p, K_q, K_r, K_s, K_t, K_u, K_v, K_w, K_x, K_y, K_z, K_RETURN, K_RSHIFT, K_TAB, K_0, K_1, K_2, K_3
from keys import UP, DOWN, LEFT, RIGHT, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, LSHIFT, SPACE, ENTER, RSHIFT, TAB, ZERO, ONE, TWO, THREE
from renderer import clamp, normalize_angle, intersect_segs, point_side, get_portal_camera_transforms, get_gradient
from settings import WALL_MAX
from math import sin, cos, isnan

@njit
def input(keys:array, prev_keys:array, mouse_delta_x: float, mouse_delta_y: float, mouse_pos_x:int, mouse_pos_y:int, camera_angle: float, camera_angle_sin:float, camera_angle_cos:float, camera_rotation_speed: float, camera_pos_x: float, camera_pos_y: float, camera_pos_z: float, camera_move_speed: float, camera_sector:int, editor_scale:float, editor_offset_x:int, editor_offset_y:int, viewport_width:int, viewport_height:int, viewport_x:int, viewport_y:int, editor_width:int, editor_height:int, editor_x:int, editor_y:int, mouse_r_pressed:bool, mouse_l_pressed:bool, mouse_m_pressed:bool, active_window:str, editor_mode:str, camera_clip:bool, actors_id:array, actors_pos_x:array, actors_pos_y:array, actors_pos_z:array, actors_billboard_id:array, possessed_actor_id:int, sectors_walls:array, vertices_x:array, vertices_y:array, walls_a_id:array, walls_b_id:array, walls_portal:array, walls_portal_wall_id:array, sectors_z_floor:array, sectors_z_ceil:array, sectors_slope_floor_z:array, sectors_slope_ceil_z:array, selected_vertices_id_n:int, sectors_slope_floor_end_x:array, sectors_slope_floor_end_y:array, sectors_slope_ceil_end_x:array, sectors_slope_ceil_end_y:array, sectors_slope_floor_wall_id:array, sectors_slope_ceil_wall_id:array, texture_select_mode:str, engine_state:str, texture_slot:int, axis_lock:int) -> tuple:

    sector_slope_floor_end_x, sector_slope_floor_end_y = sectors_slope_floor_end_x[camera_sector], sectors_slope_floor_end_y[camera_sector]
    sector_slope_floor_wall_id = sectors_slope_floor_wall_id[camera_sector]
    sector_slope_floor_wall_a_id = walls_a_id[sector_slope_floor_wall_id]
    sector_slope_floor_wall_a_x, sector_slope_floor_wall_a_y = vertices_x[sector_slope_floor_wall_a_id], vertices_y[sector_slope_floor_wall_a_id]
    sector_slope_floor_start_x, sector_slope_floor_start_y = sector_slope_floor_wall_a_x, sector_slope_floor_wall_a_y
    
    sector_slope_ceil_end_x, sector_slope_ceil_end_y = sectors_slope_ceil_end_x[camera_sector], sectors_slope_ceil_end_y[camera_sector]
    sector_slope_ceil_wall_id = sectors_slope_ceil_wall_id[camera_sector]
    sector_slope_ceil_wall_a_id = walls_a_id[sector_slope_ceil_wall_id]
    sector_slope_ceil_wall_a_x, sector_slope_ceil_wall_a_y = vertices_x[sector_slope_ceil_wall_a_id], vertices_y[sector_slope_ceil_wall_a_id]
    sector_slope_ceil_start_x, sector_slope_ceil_start_y = sector_slope_ceil_wall_a_x, sector_slope_ceil_wall_a_y

    sector_slope_floor_z = sectors_slope_floor_z[camera_sector]
    sector_slope_ceil_z = sectors_slope_ceil_z[camera_sector]
    sector_floor_z = sectors_z_floor[camera_sector]
    sector_ceil_z = sectors_z_ceil[camera_sector]

    slope_floor_z = sector_floor_z
    slope_ceil_z = sector_ceil_z

    if sector_slope_floor_z != 0:
        sector_slope_floor_z_gradient = get_gradient(sector_slope_floor_start_x, sector_slope_floor_start_y, sector_slope_floor_end_x, sector_slope_floor_end_y, camera_pos_x, camera_pos_y)
        slope_floor_z = sector_floor_z + (sector_slope_floor_z_gradient * sector_slope_floor_z)

    if sector_slope_ceil_z != 0:
        sector_slope_ceil_z_gradient = get_gradient(sector_slope_ceil_start_x, sector_slope_ceil_start_y, sector_slope_ceil_end_x, sector_slope_ceil_end_y, camera_pos_x, camera_pos_y)
        slope_ceil_z = sector_ceil_z + (sector_slope_ceil_z_gradient * sector_slope_ceil_z)

    if in_rectangle(mouse_pos_x, mouse_pos_y, editor_x, editor_y, editor_x+editor_width, editor_y+editor_height):
        if keys[UP] == 1 and selected_vertices_id_n == 0:
            editor_scale += 0.1
            editor_scale = clamp(editor_scale, 1, 200)

        if keys[DOWN] == 1 and editor_scale >= 0.1 and selected_vertices_id_n == 0:
            editor_scale -= 0.1
            editor_scale = clamp(editor_scale, 1, 200)
        
        if keys[W] == 1:
            editor_offset_y += 1

        if keys[A] == 1:
            editor_offset_x += 1

        if keys[S] == 1:
            editor_offset_y -= 1

        if keys[D] == 1:
            editor_offset_x -= 1

        if keys[P] == 1:
            editor_mode = "DRAW"

        if keys[L] == 1:
            editor_mode = "LINK"

        active_window = "EDITOR"

    elif in_rectangle(mouse_pos_x, mouse_pos_y, viewport_x, viewport_y, viewport_x+viewport_width, viewport_y+viewport_height):
        
        if camera_clip == True:
            if keys[RSHIFT] == 1:
                new_camera_pos_z = camera_pos_z
                new_camera_pos_z -= 0.1

                if new_camera_pos_z > slope_floor_z:
                    camera_pos_z = new_camera_pos_z

            if keys[SPACE] == 1:
                new_camera_pos_z = camera_pos_z
                new_camera_pos_z += 0.1

                if new_camera_pos_z < slope_ceil_z:
                    camera_pos_z = new_camera_pos_z

            if keys[LEFT] == 1:
                camera_angle += 0.01
            
            if keys[RIGHT] == 1:
                camera_angle -= 0.01
            
            camera_angle_sin, camera_angle_cos = sin(camera_angle), cos(camera_angle)

            if keys[W] == 1:
                camera_pos_x, camera_pos_y, camera_angle, camera_angle_sin, camera_angle_cos, camera_sector = move_camera(camera_pos_x, camera_pos_y, camera_pos_z, camera_angle, camera_angle_sin, camera_angle_cos, camera_move_speed, camera_sector, camera_angle_cos, camera_angle_sin, sectors_walls, walls_a_id, walls_b_id, vertices_x, vertices_y, walls_portal, walls_portal_wall_id, sectors_z_floor, sectors_z_ceil, sector_slope_floor_z, sector_slope_ceil_z, sector_slope_floor_start_x, sector_slope_floor_start_y, sector_slope_floor_end_x, sector_slope_floor_end_y, sector_slope_ceil_start_x, sector_slope_ceil_start_y, sector_slope_ceil_end_x, sector_slope_ceil_end_y, sectors_slope_floor_z, sectors_slope_ceil_z, sector_floor_z, sector_ceil_z)

            if keys[A] == 1:
                camera_pos_x, camera_pos_y, camera_angle, camera_angle_sin, camera_angle_cos, camera_sector = move_camera(camera_pos_x, camera_pos_y, camera_pos_z, camera_angle, camera_angle_sin, camera_angle_cos, camera_move_speed, camera_sector, -camera_angle_sin, camera_angle_cos, sectors_walls, walls_a_id, walls_b_id, vertices_x, vertices_y, walls_portal, walls_portal_wall_id, sectors_z_floor, sectors_z_ceil, sector_slope_floor_z, sector_slope_ceil_z, sector_slope_floor_start_x, sector_slope_floor_start_y, sector_slope_floor_end_x, sector_slope_floor_end_y, sector_slope_ceil_start_x, sector_slope_ceil_start_y, sector_slope_ceil_end_x, sector_slope_ceil_end_y, sectors_slope_floor_z, sectors_slope_ceil_z, sector_floor_z, sector_ceil_z)

            if keys[S] == 1:
                camera_pos_x, camera_pos_y, camera_angle, camera_angle_sin, camera_angle_cos, camera_sector = move_camera(camera_pos_x, camera_pos_y, camera_pos_z, camera_angle, camera_angle_sin, camera_angle_cos, -camera_move_speed, camera_sector, camera_angle_cos, camera_angle_sin, sectors_walls, walls_a_id, walls_b_id, vertices_x, vertices_y, walls_portal, walls_portal_wall_id, sectors_z_floor, sectors_z_ceil, sector_slope_floor_z, sector_slope_ceil_z, sector_slope_floor_start_x, sector_slope_floor_start_y, sector_slope_floor_end_x, sector_slope_floor_end_y, sector_slope_ceil_start_x, sector_slope_ceil_start_y, sector_slope_ceil_end_x, sector_slope_ceil_end_y, sectors_slope_floor_z, sectors_slope_ceil_z, sector_floor_z, sector_ceil_z)

            if keys[D] == 1:
                camera_pos_x, camera_pos_y, camera_angle, camera_angle_sin, camera_angle_cos, camera_sector = move_camera(camera_pos_x, camera_pos_y, camera_pos_z, camera_angle, camera_angle_sin, camera_angle_cos, camera_move_speed, camera_sector, camera_angle_sin, -camera_angle_cos, sectors_walls, walls_a_id, walls_b_id, vertices_x, vertices_y, walls_portal, walls_portal_wall_id, sectors_z_floor, sectors_z_ceil, sector_slope_floor_z, sector_slope_ceil_z, sector_slope_floor_start_x, sector_slope_floor_start_y, sector_slope_floor_end_x, sector_slope_floor_end_y, sector_slope_ceil_start_x, sector_slope_ceil_start_y, sector_slope_ceil_end_x, sector_slope_ceil_end_y, sectors_slope_floor_z, sectors_slope_ceil_z, sector_floor_z, sector_ceil_z)

            if keys[I] == 1:
                if keys[LSHIFT] == 1:
                    sector_z_ceil = sectors_z_ceil[camera_sector]
                    sector_z_ceil += 0.01
                    sectors_z_ceil[camera_sector] = sector_z_ceil
                else:
                    sector_z_floor = sectors_z_floor[camera_sector]
                    sector_z_floor += 0.01
                    sectors_z_floor[camera_sector] = sector_z_floor
            
            if keys[K] == 1:
                if keys[LSHIFT] == 1:
                    sector_z_ceil = sectors_z_ceil[camera_sector]
                    sector_z_ceil -= 0.01
                    sectors_z_ceil[camera_sector] = sector_z_ceil
                else:
                    sector_z_floor = sectors_z_floor[camera_sector]
                    sector_z_floor -= 0.01
                    sectors_z_floor[camera_sector] = sector_z_floor

            if keys[U] == 1:
                if keys[LSHIFT] == 1:
                    sector_slope_ceil_z = sectors_slope_ceil_z[camera_sector]
                    sector_slope_ceil_z += 0.01
                    sectors_slope_ceil_z[camera_sector] = sector_slope_ceil_z
                else:
                    sector_slope_floor_z = sectors_slope_floor_z[camera_sector]
                    sector_slope_floor_z += 0.01
                    sectors_slope_floor_z[camera_sector] = sector_slope_floor_z
            
            if keys[J] == 1:
                if keys[LSHIFT] == 1:
                    sector_slope_ceil_z = sectors_slope_ceil_z[camera_sector]
                    sector_slope_ceil_z -= 0.01
                    sectors_slope_ceil_z[camera_sector] = sector_slope_ceil_z
                else:
                    sector_slope_floor_z = sectors_slope_floor_z[camera_sector]
                    sector_slope_floor_z -= 0.01
                    sectors_slope_floor_z[camera_sector] = sector_slope_floor_z

            if keys[TAB] == 1 and prev_keys[TAB] == 0:
                if texture_select_mode == "WALL":
                    texture_select_mode = "SECTOR"
                elif texture_select_mode == "SECTOR":
                    texture_select_mode = "SKYBOX"
                elif texture_select_mode == "SKYBOX":
                    texture_select_mode = "WALL"

            if keys[Y] == 1 and prev_keys[Y] == 0:
                if engine_state == "DEBUG":
                    engine_state = "NONE"
                else:
                    engine_state = "DEBUG"
            
            if keys[ONE]:
                texture_slot = 0

            if keys[TWO]:
                texture_slot = 1

            if keys[THREE]:
                texture_slot = 2

                if texture_select_mode == "SECTOR":
                    texture_slot = 0
        else:
            pass

        active_window = "VIEWPORT"
    
    else:
        active_window = "NONE"

    return camera_angle, camera_angle_sin, camera_angle_cos, camera_pos_x, camera_pos_y, camera_pos_z, camera_sector, editor_scale, editor_offset_x, editor_offset_y, active_window, editor_mode, camera_clip, actors_id, actors_pos_x, actors_pos_y, actors_pos_z, actors_billboard_id, possessed_actor_id, sectors_slope_floor_z, sectors_slope_ceil_z, sectors_z_floor, sectors_z_ceil, texture_select_mode, engine_state, texture_slot, axis_lock

@njit(fastmath=True)
def rotate_camera(camera_angle:float, mouse_delta_x:float, camera_rotation_speed:float) -> tuple:
    camera_angle += (mouse_delta_x * -camera_rotation_speed) / 2
    camera_angle = normalize_angle(camera_angle)

    camera_angle_sin = sin(camera_angle)
    camera_angle_cos = cos(camera_angle)

    return camera_angle, camera_angle_sin, camera_angle_cos

@njit 
def move_camera(camera_pos_x:float, camera_pos_y:float, camera_pos_z:float, camera_angle:float, camera_angle_sin:float, camera_angle_cos:float, camera_move_speed:float, camera_sector:int, delta_x:float, delta_y:float, sectors_walls:array, walls_a_id:array, walls_b_id:array, vertices_x:array, vertices_y:array, walls_portal:array, walls_portal_wall_id:array, sectors_z_floor:array, sectors_z_ceil:array, sector_slope_floor_z:float, sector_slope_ceil_z:float, sector_slope_floor_start_x:float, sector_slope_floor_start_y:float, sector_slope_floor_end_x:float, sector_slope_floor_end_y:float, sector_slope_ceil_start_x:float, sector_slope_ceil_start_y:float, sector_slope_ceil_end_x:float, sector_slope_ceil_end_y:float, sectors_slope_floor_z:array, sectors_slope_ceil_z:array, sector_floor_z:float, sector_ceil_z:float) -> tuple:
    new_camera_pos_x = camera_pos_x + camera_move_speed * delta_x
    new_camera_pos_y = camera_pos_y + camera_move_speed * delta_y
    new_camera_angle = camera_angle
    new_camera_angle_sin = camera_angle_sin
    new_camera_angle_cos = camera_angle_cos

    collision = False
    new_camera_sector = camera_sector

    for wall_n in range(WALL_MAX):
        wall_id = sectors_walls[camera_sector][wall_n]

        if wall_id != 0:
            wall_a_id = walls_a_id[wall_id]
            wall_b_id = walls_b_id[wall_id]

            wall_a_x, wall_a_y = vertices_x[wall_a_id], vertices_y[wall_a_id]
            wall_b_x, wall_b_y = vertices_x[wall_b_id], vertices_y[wall_b_id]

            wall_portal = walls_portal[wall_id]
            wall_portal_wall_id = walls_portal_wall_id[wall_id]

            collision_x, collision_y = intersect_segs(camera_pos_x, camera_pos_y, new_camera_pos_x, new_camera_pos_y, wall_a_x, wall_a_y, wall_b_x, wall_b_y)

            if not isnan(collision_x) and point_side(camera_pos_x, camera_pos_y, wall_a_x, wall_a_y, wall_b_x, wall_b_y) < 0:
                if wall_portal == 0:
                    collision = True
                else:
                    portal_sector_z_floor = sectors_z_floor[wall_portal]
                    portal_sector_z_ceil = sectors_z_ceil[wall_portal]
                    portal_sector_slope_floor_z = sectors_slope_floor_z[wall_portal]
                    portal_sector_slope_ceil_z = sectors_slope_ceil_z[wall_portal]

                    if portal_sector_z_floor > camera_pos_z or camera_pos_z > portal_sector_z_ceil:
                        collision = True
                        break

                    new_camera_sector = wall_portal
                    
                    portal_wall_a_id = walls_a_id[wall_portal_wall_id]
                    portal_wall_b_id = walls_b_id[wall_portal_wall_id]
                    portal_wall_a_x, portal_wall_a_y = vertices_x[portal_wall_a_id], vertices_y[portal_wall_a_id]
                    portal_wall_b_x, portal_wall_b_y = vertices_x[portal_wall_b_id], vertices_y[portal_wall_b_id]
                    
                    if wall_a_x != portal_wall_a_x and wall_a_y != portal_wall_a_y and wall_b_x != portal_wall_b_x and wall_b_y != portal_wall_b_y:
                        new_camera_pos_x, new_camera_pos_y, new_camera_angle, new_camera_angle_sin, new_camera_angle_cos = get_portal_camera_transforms(new_camera_pos_x, new_camera_pos_y, new_camera_angle, wall_a_x, wall_a_y, wall_b_x, wall_b_y, portal_wall_a_x, portal_wall_a_y, portal_wall_b_x, portal_wall_b_y)
                break

    if sector_slope_floor_z != 0:
        sector_slope_floor_z_gradient = get_gradient(sector_slope_floor_start_x, sector_slope_floor_start_y, sector_slope_floor_end_x, sector_slope_floor_end_y, new_camera_pos_x, new_camera_pos_y)
        slope_floor_z = sector_floor_z + (sector_slope_floor_z_gradient * sector_slope_floor_z)

        if slope_floor_z >= camera_pos_z: 
            collision = True

    if sector_slope_ceil_z != 0:
        sector_slope_ceil_z_gradient = get_gradient(sector_slope_ceil_start_x, sector_slope_ceil_start_y, sector_slope_ceil_end_x, sector_slope_ceil_end_y, new_camera_pos_x, new_camera_pos_y)
        slope_ceil_z = sector_ceil_z + (sector_slope_ceil_z_gradient * sector_slope_ceil_z)

        if camera_pos_z >= slope_ceil_z:
            collision = True

    if collision == True:
        return camera_pos_x, camera_pos_y, camera_angle, camera_angle_sin, camera_angle_cos, camera_sector
    else:
        return new_camera_pos_x, new_camera_pos_y, new_camera_angle, new_camera_angle_sin, new_camera_angle_cos, new_camera_sector

@njit(fastmath=True)
def check_collision():
    pass

@njit(fastmath=True)
def in_rectangle(_p_x, _p_y, _p1_x, _p1_y, _p2_x, _p2_y):
    min_x = min(_p1_x, _p2_x)
    max_x = max(_p1_x, _p2_x)
    min_y = min(_p1_y, _p2_y)
    max_y = max(_p1_y, _p2_y)

    return min_x <= _p_x <= max_x and min_y <= _p_y <= max_y

@njit(fastmath=True)        
def flip_flop(_m:bool, _m1:bool, _m2:bool)->bool:
    m = _m
    if _m == _m1:
        m = _m2
    else:
        m = _m1

    return m

def input_assignment(keys:array):
    pygame_keys = key.get_pressed()

    keys = press_key(pygame_keys, K_UP, keys, UP)
    keys = press_key(pygame_keys, K_DOWN, keys, DOWN)
    keys = press_key(pygame_keys, K_LEFT, keys, LEFT)
    keys = press_key(pygame_keys, K_RIGHT, keys, RIGHT)

    keys = press_key(pygame_keys, K_a, keys, A)
    keys = press_key(pygame_keys, K_b, keys, B)
    keys = press_key(pygame_keys, K_c, keys, C)
    keys = press_key(pygame_keys, K_d, keys, D)
    keys = press_key(pygame_keys, K_e, keys, E)
    keys = press_key(pygame_keys, K_f, keys, F)
    keys = press_key(pygame_keys, K_g, keys, G)
    keys = press_key(pygame_keys, K_h, keys, H)
    keys = press_key(pygame_keys, K_i, keys, I)
    keys = press_key(pygame_keys, K_j, keys, J)
    keys = press_key(pygame_keys, K_k, keys, K)
    keys = press_key(pygame_keys, K_l, keys, L)
    keys = press_key(pygame_keys, K_m, keys, M)
    keys = press_key(pygame_keys, K_n, keys, N)
    keys = press_key(pygame_keys, K_o, keys, O)
    keys = press_key(pygame_keys, K_p, keys, P)
    keys = press_key(pygame_keys, K_q, keys, Q)
    keys = press_key(pygame_keys, K_r, keys, R)
    keys = press_key(pygame_keys, K_s, keys, S)
    keys = press_key(pygame_keys, K_t, keys, T)
    keys = press_key(pygame_keys, K_u, keys, U)
    keys = press_key(pygame_keys, K_v, keys, V)
    keys = press_key(pygame_keys, K_w, keys, W)
    keys = press_key(pygame_keys, K_x, keys, X)
    keys = press_key(pygame_keys, K_y, keys, Y)
    keys = press_key(pygame_keys, K_z, keys, Z)

    keys = press_key(pygame_keys, K_LSHIFT, keys, LSHIFT)
    keys = press_key(pygame_keys, K_SPACE, keys, SPACE)
    keys = press_key(pygame_keys, K_TAB, keys, TAB)
    keys = press_key(pygame_keys, K_RETURN, keys, ENTER)
    keys = press_key(pygame_keys, K_RSHIFT, keys, RSHIFT)

    keys = press_key(pygame_keys, K_0, keys, ZERO)
    keys = press_key(pygame_keys, K_1, keys, ONE)
    keys = press_key(pygame_keys, K_2, keys, TWO)
    keys = press_key(pygame_keys, K_3, keys, THREE)

    return keys
    
def press_key(pygame_keys:array, pygame_key:any, keys:array, key:int)->array:
    if pygame_keys[pygame_key]:
        keys[key] = 1
    else:
        keys[key] = 0
    
    return keys