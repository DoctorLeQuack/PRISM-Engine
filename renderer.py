from numba import njit, prange
from numpy import full, int32, float32, array, zeros
from math import cos, sin, atan2, floor, nan, fabs, isnan, tan, sqrt, atan, inf
from settings import HFOV_2, ZNEAR, ZFAR, QUEUE_MAX, SECTOR_MAX, WALL_MAX, MASKED_MAX, PI, TAU, PI_2, PI_4, HFOV
from keys import B, N, M
from colors import DEBUG_FLOOR_COLOR, DEBUG_CEILING_COLOR, DEBUG_WALL_COLOR, DEBUG_WALLPORTALBOTTOM_COLOR, DEBUG_WALLPORTALTOP_COLOR, KEY_COLOR

@njit
def render_viewport(screen:array, viewport_width:int, viewport_height:int, viewport_x:int, viewport_y:int, znl_x:float, znl_y:float, znr_x:float, znr_y:float, zfl_x:float, zfl_y:float, zfr_x:float, zfr_y:float, camera_sector:int, camera_pos_x:float, camera_pos_y:float, camera_pos_z:float, camera_angle:float, camera_fog_distance:float, camera_angle_sin:float, camera_angle_cos:float, sectors_id:array, sectors_light_factor:array, sectors_z_floor:array, sectors_z_ceil:array, sectors_ceil_texture_id:array, sectors_ceil_animation_id:array, sectors_floor_texture_id:array, sectors_floor_animation_id:array, sectors_slope_floor_z:array, sectors_slope_floor_wall_id:array, sectors_slope_ceil_z:array, sectors_slope_ceil_wall_id:array, sectors_slope_floor_friction:array, walls_id:array, walls_a_id:array, walls_b_id:array, walls_portal:array, walls_portal_wall_id:array, walls_sector_id:array, walls_texture_id:array, walls_texture_id_up:array, walls_texture_id_down:array, walls_animation_id:array, walls_texture_offset_x:array, walls_texture_offset_y:array, walls_texture_offset_up_x:array, walls_texture_offset_up_y:array, walls_texture_offset_down_x:array, walls_texture_offset_down_y:array, textures_sheet:array, textures_width:array, textures_height:array, environmental_animations_frames:array, environmental_animations_ms:array, ticks:int, textures_x_coordinate:array, engine_state:str, sectors_slope_floor_end_x:array, sectors_slope_floor_end_y:array, sectors_slope_ceil_end_x:array, sectors_slope_ceil_end_y:array, environmental_animations_frames_count:array, billboards_id:array, billboards_sector_id:array, billboards_sprite_id:array, billboards_position_x:array, billboards_position_y:array, billboards_position_z:array, sprites_sheet:array, sprites_x_coordinate:array, sprites_width:array, sprites_height:array, vertices_id:array, vertices_x:array, vertices_y:array, sectors_walls:array, keys:array, prev_keys:array, textures_n:int, skyboxes_sheet:array, skyboxes_width:array, skyboxes_height:array, skyboxes_x_coordinates:array, sectors_skybox_id:array, texture_select_mode:str, texture_slot_id:int, skyboxes_n:int)->tuple:
    
    viewport_width_2 = viewport_width // 2
    viewport_height_2 = viewport_height // 2
    viewport_ratio_height, viewport_ratio_width_i, viewport_projection_plane_distance = viewport_width/viewport_height, (viewport_height/viewport_width)-1, viewport_width_2 / tan(HFOV_2)

    y_lo = full(viewport_width+1, 0, dtype=int32)
    y_hi = full(viewport_width+1, viewport_height, dtype=int32)

    sector_draw = 0

    faced_wall_id = 0
    detection_point_x, detection_point_y = extend_v(camera_pos_x, camera_pos_y, ZFAR, camera_angle)

    sector_queue_n, sector_queue_id, sector_queue_wall_id, sector_queue_x0, sector_queue_x1, sector_queue_camera_pos_x, sector_queue_camera_pos_y, sector_queue_camera_angle, sector_queue_camera_angle_sin, sector_queue_camera_angle_cos = create_sector_queue()
    
    sector_queue_n = 1
    sector_queue_id[0] = camera_sector
    sector_queue_wall_id[0] = -1
    sector_queue_x0[0] = 0
    sector_queue_x1[0] = viewport_width
    sector_queue_camera_pos_x[0] = camera_pos_x
    sector_queue_camera_pos_y[0] = camera_pos_y
    sector_queue_camera_angle[0] = camera_angle
    sector_queue_camera_angle_sin[0] = camera_angle_sin
    sector_queue_camera_angle_cos[0] = camera_angle_cos
    
    masked_queue_n = 0
    masked_queue_type = zeros(MASKED_MAX, dtype=int32)
    masked_queue_y_lo = zeros((viewport_width, MASKED_MAX), dtype=int32)
    masked_queue_y_hi = zeros((viewport_width, MASKED_MAX), dtype=int32)
    masked_queue_length = zeros(MASKED_MAX, dtype=float32)
    masked_queue_x0 = zeros(MASKED_MAX, dtype=int32)
    masked_queue_x1 = zeros(MASKED_MAX, dtype=int32)
    masked_queue_txd = zeros(MASKED_MAX, dtype=int32)
    masked_queue_tx0 = zeros(MASKED_MAX, dtype=int32)
    masked_queue_iz0 = zeros(MASKED_MAX, dtype=float32)
    masked_queue_iz1 = zeros(MASKED_MAX, dtype=float32)
    masked_queue_u0_z0 = zeros(MASKED_MAX, dtype=float32)
    masked_queue_u1_z1 = zeros(MASKED_MAX, dtype=float32)
    masked_queue_yfd = zeros(MASKED_MAX, dtype=int32)
    masked_queue_yf0 = zeros(MASKED_MAX, dtype=int32)
    masked_queue_ycd = zeros(MASKED_MAX, dtype=int32)
    masked_queue_yc0 = zeros(MASKED_MAX, dtype=int32)
    masked_queue_yyfd = zeros(MASKED_MAX, dtype=int32)
    masked_queue_yyf0 = zeros(MASKED_MAX, dtype=int32)
    masked_queue_yycd = zeros(MASKED_MAX, dtype=int32)
    masked_queue_yyc0 = zeros(MASKED_MAX, dtype=int32)
    masked_queue_sector_z_floor_total = zeros(MASKED_MAX, dtype=float32)
    masked_queue_sector_z_ceil_total = zeros(MASKED_MAX, dtype=float32)
    masked_queue_texture_offset_x = zeros(MASKED_MAX, dtype=int32)
    masked_queue_texture_offset_y = zeros(MASKED_MAX, dtype=int32)
    masked_queue_texture_width = zeros(MASKED_MAX, dtype=int32)
    masked_queue_texture_height = zeros(MASKED_MAX, dtype=int32)
    masked_queue_texture_x_coordinate = zeros(MASKED_MAX, dtype=int32)
    masked_queue_sector_light_factor = zeros(MASKED_MAX, dtype=float32)
    masked_queue_billboard_a_x = zeros(MASKED_MAX, dtype=float32)
    masked_queue_billboard_a_y = zeros(MASKED_MAX, dtype=float32)
    masked_queue_billboard_b_x = zeros(MASKED_MAX, dtype=float32)
    masked_queue_billboard_b_y = zeros(MASKED_MAX, dtype=float32)
    masked_queue_billboard_z = zeros(MASKED_MAX, dtype=float32)
    masked_queue_billboard_size_x = zeros(MASKED_MAX, dtype=float32)
    masked_queue_billboard_size_y = zeros(MASKED_MAX, dtype=float32)
    masked_queue_camera_position_x = zeros(MASKED_MAX, dtype=float32)
    masked_queue_camera_position_y = zeros(MASKED_MAX, dtype=float32)
    masked_queue_camera_angle = zeros(MASKED_MAX, dtype=float32)
    masked_queue_camera_angle_cos = zeros(MASKED_MAX, dtype=float32)
    masked_queue_camera_angle_sin = zeros(MASKED_MAX, dtype=float32)
    masked_queue_entry_x0 = zeros(MASKED_MAX, dtype=int32)
    masked_queue_entry_x1 = zeros(MASKED_MAX, dtype=int32)

    while (sector_queue_n != 0 and sector_draw < SECTOR_MAX):
        entry_id = sector_queue_id[sector_queue_n-1]
        entry_wall_id = sector_queue_wall_id[sector_queue_n-1]
        entry_x0 = sector_queue_x0[sector_queue_n-1]
        entry_x1 = sector_queue_x1[sector_queue_n-1]
        entry_camera_pos_x = sector_queue_camera_pos_x[sector_queue_n-1]
        entry_camera_pos_y = sector_queue_camera_pos_y[sector_queue_n-1]
        entry_camera_angle = sector_queue_camera_angle[sector_queue_n-1]
        entry_camera_angle_sin = sector_queue_camera_angle_sin[sector_queue_n-1]
        entry_camera_angle_cos = sector_queue_camera_angle_cos[sector_queue_n-1]

        skybox_texture_id = sectors_skybox_id[entry_id]
        skybox_texture_x_coordinate = skyboxes_x_coordinates[skybox_texture_id]
        skybox_texture_width = skyboxes_width[skybox_texture_id]
        skybox_texture_height = skyboxes_height[skybox_texture_id]

        sector_queue_n -= 1
        sector_draw += 1

        sector_walls = full(WALL_MAX, 0, dtype=int32)

        for wall_id in walls_id:
            if wall_id != 0:
                wall_sector_id = walls_sector_id[wall_id]

                if wall_sector_id == entry_id:
                    sector_walls[wall_id] = wall_id
                    sectors_walls[entry_id][wall_id] = wall_id

        sector_light_factor = sectors_light_factor[entry_id]

        sector_z_floor = sectors_z_floor[entry_id]
        sector_slope_floor_wall_id = sectors_slope_floor_wall_id[entry_id]
        sector_slope_floor_wall_a_x, sector_slope_floor_wall_a_y, sector_slope_floor_wall_a_id, sector_slope_floor_wall_b_x, sector_slope_floor_wall_b_y, sector_slope_floor_wall_b_id, sector_slope_floor_wall_portal, sector_slope_floor_wall_portal_wall_id, sector_slope_floor_wall_sector_id, sector_slope_floor_wall_texture_id, sector_slope_floor_wall_texture_id_up, sector_slope_floor_wall_texture_id_down, sector_slope_floor_wall_animation_id, sector_slope_floor_wall_texture_offset_x, sector_slope_floor_wall_texture_offset_y, sector_slope_floor_wall_texture_offset_up_x, sector_slope_floor_wall_texture_offset_up_y, sector_slope_floor_wall_texture_offset_down_x, sector_slope_floor_wall_texture_offset_down_y = get_wall(sector_slope_floor_wall_id, walls_a_id, walls_b_id, walls_portal, walls_portal_wall_id, walls_sector_id, walls_texture_id, walls_texture_id_up, walls_texture_id_down, walls_animation_id, walls_texture_offset_x, walls_texture_offset_y, walls_texture_offset_up_x, walls_texture_offset_up_y, walls_texture_offset_down_x, walls_texture_offset_down_y, vertices_id, vertices_x, vertices_y)
        sector_slope_floor_z = sectors_slope_floor_z[entry_id]
        sector_slope_floor_start_x = sector_slope_floor_wall_a_x
        sector_slope_floor_start_y = sector_slope_floor_wall_a_y
        sector_slope_floor_end_x = sectors_slope_floor_end_x[entry_id]
        sector_slope_floor_end_y = sectors_slope_floor_end_y[entry_id]

        sector_z_ceil = sectors_z_ceil[entry_id]
        sector_slope_ceil_wall_id = sectors_slope_ceil_wall_id[entry_id]
        sector_slope_ceil_wall_a_x, sector_slope_ceil_wall_a_y, sector_slope_ceil_wall_a_id, sector_slope_ceil_wall_b_x, sector_slope_ceil_wall_b_y, sector_slope_ceil_wall_b_id, sector_slope_ceil_wall_portal, sector_slope_ceil_wall_portal_wall_id, sector_slope_ceil_wall_sector_id, sector_slope_ceil_wall_texture_id, sector_slope_ceil_wall_texture_id_up, sector_slope_ceil_wall_texture_id_down, sector_slope_ceil_wall_animation_id, sector_slope_ceil_wall_texture_offset_x, sector_slope_ceil_wall_texture_offset_y, sector_slope_ceil_wall_texture_offset_up_x, sector_slope_ceil_wall_texture_offset_up_y, sector_slope_ceil_wall_texture_offset_down_x, sector_slope_ceil_wall_texture_offset_down_y = get_wall(sector_slope_ceil_wall_id, walls_a_id, walls_b_id, walls_portal, walls_portal_wall_id, walls_sector_id, walls_texture_id, walls_texture_id_up, walls_texture_id_down, walls_animation_id, walls_texture_offset_x, walls_texture_offset_y, walls_texture_offset_up_x, walls_texture_offset_up_y, walls_texture_offset_down_x, walls_texture_offset_down_y, vertices_id, vertices_x, vertices_y)
        sector_slope_ceil_z = sectors_slope_ceil_z[entry_id]
        sector_slope_ceil_start_x = sector_slope_ceil_wall_a_x
        sector_slope_ceil_start_y = sector_slope_ceil_wall_a_y
        sector_slope_ceil_end_x = sectors_slope_ceil_end_x[entry_id]
        sector_slope_ceil_end_y = sectors_slope_ceil_end_y[entry_id]

        if sector_slope_floor_z != 0:
            sector_slope_floor_end_x, sector_slope_floor_end_y = get_slope_point(sector_slope_floor_wall_a_x, sector_slope_floor_wall_a_y, sector_slope_floor_wall_b_x, sector_slope_floor_wall_b_y, sector_walls, walls_a_id, walls_b_id, vertices_x, vertices_y)

            sectors_slope_floor_end_x[entry_id] = sector_slope_floor_end_x
            sectors_slope_floor_end_y[entry_id] = sector_slope_floor_end_y

            local_sector_slope_floor_start_x, local_sector_slope_floor_start_y = world_pos_to_local(entry_camera_pos_x, entry_camera_pos_y, entry_camera_angle_sin, entry_camera_angle_cos, sector_slope_floor_start_x, sector_slope_floor_start_y)
            local_sector_slope_floor_end_x, local_sector_slope_floor_end_y = world_pos_to_local(entry_camera_pos_x, entry_camera_pos_y, entry_camera_angle_sin, entry_camera_angle_cos, sector_slope_floor_end_x, sector_slope_floor_end_y)

        if sector_slope_ceil_z != 0:
            sector_slope_ceil_end_x, sector_slope_ceil_end_y = get_slope_point(sector_slope_ceil_wall_a_x, sector_slope_ceil_wall_a_y, sector_slope_ceil_wall_b_x, sector_slope_ceil_wall_b_y, sector_walls, walls_a_id, walls_b_id, vertices_x, vertices_y)
            
            sectors_slope_ceil_end_x[entry_id] = sector_slope_ceil_end_x
            sectors_slope_ceil_end_y[entry_id] = sector_slope_ceil_end_y

            local_sector_slope_ceil_start_x, local_sector_slope_ceil_start_y = world_pos_to_local(entry_camera_pos_x, entry_camera_pos_y, entry_camera_angle_sin, entry_camera_angle_cos, sector_slope_ceil_start_x, sector_slope_ceil_start_y)
            local_sector_slope_ceil_end_x, local_sector_slope_ceil_end_y = world_pos_to_local(entry_camera_pos_x, entry_camera_pos_y, entry_camera_angle_sin, entry_camera_angle_cos, sector_slope_ceil_end_x, sector_slope_ceil_end_y)

        sector_z_floor_total = total_z(sector_z_floor, sector_slope_floor_z, True)
        sector_z_ceil_total = total_z(sector_z_ceil, sector_slope_ceil_z, False)

        sector_animation_floor_id = sectors_floor_animation_id[entry_id]
        sector_animation_ceil_id = sectors_ceil_animation_id[entry_id]

        sector_texture_floor_id = sectors_floor_texture_id[entry_id]
        sector_texture_ceil_id = sectors_ceil_texture_id[entry_id]

        if sector_animation_floor_id != 0:

            environmental_animation_ms = environmental_animations_ms[sector_animation_floor_id]
            environmental_animation_frames_count = environmental_animations_frames_count[sector_animation_floor_id]
            sector_texture_floor_id = environmental_animations_frames[sector_animation_floor_id][get_frame(ticks, environmental_animation_ms, environmental_animation_frames_count)]

        if sector_animation_ceil_id != 0:

            environmental_animation_ms = environmental_animations_ms[sector_animation_ceil_id]
            environmental_animation_frames_count = environmental_animations_frames_count[sector_animation_ceil_id]
            sector_texture_ceil_id = environmental_animations_frames[sector_animation_ceil_id][get_frame(ticks, environmental_animation_ms, environmental_animation_frames_count)]

        sector_floor_texture_x_coordinate = textures_x_coordinate[sector_texture_floor_id]
        sector_floor_texture_width = textures_width[sector_texture_floor_id]
        sector_floor_texture_height = textures_height[sector_texture_floor_id]

        sector_ceil_texture_x_coordinate = textures_x_coordinate[sector_texture_ceil_id]
        sector_ceil_texture_width = textures_width[sector_texture_ceil_id]
        sector_ceil_texture_height = textures_height[sector_texture_ceil_id]

        for billboard_id in billboards_id:
            if billboard_id != 0:
                billboard_sector = billboards_sector_id[billboard_id]

                if billboard_sector == entry_id:

                    for x in range(viewport_width):
                        masked_queue_y_lo[x][masked_queue_n] = y_lo[x]
                        masked_queue_y_hi[x][masked_queue_n] = y_hi[x]

                    masked_queue_sector_light_factor[masked_queue_n] = sector_light_factor

                    masked_queue_camera_position_x[masked_queue_n] = entry_camera_pos_x
                    masked_queue_camera_position_y[masked_queue_n] = entry_camera_pos_y
                    masked_queue_camera_angle[masked_queue_n] = entry_camera_angle
                    masked_queue_camera_angle_cos[masked_queue_n] = entry_camera_angle_cos
                    masked_queue_camera_angle_sin[masked_queue_n] = entry_camera_angle_sin

                    billboard_sprite_id = billboards_sprite_id[billboard_id]
                    sprite_width, sprite_height = sprites_width[billboard_sprite_id], sprites_height[billboard_sprite_id]
                    masked_queue_texture_x_coordinate[masked_queue_n] = sprites_x_coordinate[billboard_sprite_id]
                    masked_queue_texture_height[masked_queue_n] = sprite_height
                    masked_queue_texture_width[masked_queue_n] = sprite_width

                    billboard_position_x, billboard_position_y = billboards_position_x[billboard_id], billboards_position_y[billboard_id]
                    billboard_size_x, billboard_size_y = sprite_width*0.01, sprite_height*0.01
                    billboard_size_x_2 = billboard_size_x * 0.5
                    billboard_angle = get_angle(camera_pos_x, camera_pos_y, billboard_position_x, billboard_position_y)
                    masked_queue_length[masked_queue_n] = billboard_size_x
                    masked_queue_billboard_size_x[masked_queue_n], masked_queue_billboard_size_y[masked_queue_n] = billboard_size_x, billboard_size_y
                    masked_queue_billboard_a_x[masked_queue_n], masked_queue_billboard_a_y[masked_queue_n] = extend_v(billboard_position_x, billboard_position_y, -billboard_size_x_2, billboard_angle)
                    masked_queue_billboard_b_x[masked_queue_n], masked_queue_billboard_b_y[masked_queue_n] = extend_v(billboard_position_x, billboard_position_y, billboard_size_x_2, billboard_angle)
                    masked_queue_billboard_z[masked_queue_n] = billboards_position_z[masked_queue_n]
                    masked_queue_entry_x0[masked_queue_n] = entry_x0
                    masked_queue_entry_x1[masked_queue_n] = entry_x1

                    masked_queue_type[masked_queue_n] = 2

                    masked_queue_n += 1

        for wall_id in sector_walls:
            if wall_id != 0:
                wall_sector_id = walls_sector_id[wall_id]

                if wall_sector_id == entry_id:

                    wall_a_x, wall_a_y, wall_a_id, wall_b_x, wall_b_y, wall_b_id, wall_portal, wall_portal_wall_id, wall_sector_id, wall_texture_id, wall_texture_id_up, wall_texture_id_down, wall_animation_id, wall_texture_offset_x, wall_texture_offset_y, wall_texture_up_offset_x, wall_texture_up_offset_y, wall_texture_down_offset_x, wall_texture_down_offset_y = get_wall(wall_id, walls_a_id, walls_b_id, walls_portal, walls_portal_wall_id, walls_sector_id, walls_texture_id, walls_texture_id_up, walls_texture_id_down, walls_animation_id, walls_texture_offset_x, walls_texture_offset_y, walls_texture_offset_up_x, walls_texture_offset_up_y, walls_texture_offset_down_x, walls_texture_offset_down_y, vertices_id, vertices_x, vertices_y)

                    if wall_sector_id == camera_sector:
                        intersection_x, intersection_y = intersect_segs(camera_pos_x, camera_pos_y, detection_point_x, detection_point_y, wall_a_x, wall_a_y, wall_b_x, wall_b_y)
                        if not isnan(intersection_x):
                            faced_wall_id = wall_id

                    if point_side(entry_camera_pos_x, entry_camera_pos_y, wall_a_x, wall_a_y, wall_b_x, wall_b_y) < 0:
                        op0_x, op0_y = world_pos_to_local(entry_camera_pos_x, entry_camera_pos_y, entry_camera_angle_sin, entry_camera_angle_cos, wall_a_x, wall_a_y)
                        op1_x, op1_y = world_pos_to_local(entry_camera_pos_x, entry_camera_pos_y, entry_camera_angle_sin, entry_camera_angle_cos, wall_b_x, wall_b_y)

                        cp0_x, cp0_y = op0_x, op0_y
                        cp1_x, cp1_y = op1_x, op1_y

                        if (cp0_y <= 0 and cp1_y <= 0): continue

                        ap0 = get_angle(0, 0, cp0_x, cp0_y)
                        ap1 = get_angle(0, 0, cp1_x, cp1_y)

                        cp0_x, cp0_y, cp1_x, cp1_y, ap0, ap1 = near_clip(cp0_x, cp0_y, cp1_x, cp1_y, ap0, ap1, znl_x, znl_y, zfl_x, zfl_y, znr_x, znr_y, zfr_x, zfr_y)

                        if ap0 < ap1: continue
                        if (ap0 < -(HFOV_2) and ap1 < -(HFOV_2)) or (ap0 > +(HFOV_2) and ap1 > +(HFOV_2)): continue

                        tx0 = screen_angle_to_x(ap0, viewport_width_2)
                        tx1 = screen_angle_to_x(ap1, viewport_width_2)

                        if tx0 > entry_x1: continue
                        if tx1 < entry_x0: continue

                        x0 = clamp(tx0, entry_x0, entry_x1)
                        x1 = clamp(tx1, entry_x0, entry_x1)

                        wall_texture_x_coordinate = textures_x_coordinate[wall_texture_id]
                        wall_texture_height = textures_height[wall_texture_id]
                        wall_texture_width = textures_width[wall_texture_id]

                        wall_texture_down_x_coordinate = textures_x_coordinate[wall_texture_id_down]
                        wall_texture_down_height = textures_height[wall_texture_id_down]
                        wall_texture_down_width = textures_width[wall_texture_id_down]

                        wall_texture_up_x_coordinate = textures_x_coordinate[wall_texture_id_up]
                        wall_texture_up_height = textures_height[wall_texture_id_up]
                        wall_texture_up_width = textures_width[wall_texture_id_up]

                        wall_u0 = 0
                        wall_u1 = 1

                        if cp0_x != op0_x:
                            wall_u0 = extension_factor(op0_x, op0_y, op1_x, op1_y, cp0_x, cp0_y, 0)

                        if cp1_x != op1_x:
                            wall_u1 = extension_factor(op0_x, op0_y, op1_x, op1_y, cp1_x, cp1_y, 1)

                        wall_z_floor0 = sector_z_floor
                        wall_z_floor1 = sector_z_floor

                        wall_z_ceil0 = sector_z_ceil
                        wall_z_ceil1 = sector_z_ceil

                        wall_length = distance(wall_a_x, wall_a_y, wall_b_x, wall_b_y)
                        sector_height = sector_z_ceil_total - sector_z_floor_total

                        if sector_slope_floor_z != 0:
                            wall_z_floor0, wall_z_floor1 = get_z_on_slope(local_sector_slope_floor_start_x, local_sector_slope_floor_start_y, local_sector_slope_floor_end_x, local_sector_slope_floor_end_y, cp0_x, cp0_y, cp1_x, cp1_y, sector_z_floor, sector_slope_floor_z)

                        if sector_slope_ceil_z != 0:
                            wall_z_ceil0, wall_z_ceil1 = get_z_on_slope(local_sector_slope_ceil_start_x, local_sector_slope_ceil_start_y, local_sector_slope_ceil_end_x, local_sector_slope_ceil_end_y, cp0_x, cp0_y, cp1_x, cp1_y, sector_z_ceil, sector_slope_ceil_z)

                        portal_sector_z_floor = 0
                        portal_sector_z_ceil = 0

                        portal_wall_z_floor0 = portal_sector_z_floor
                        portal_wall_z_floor1 = portal_sector_z_floor

                        portal_wall_z_ceil0 = portal_sector_z_ceil
                        portal_wall_z_ceil1 = portal_sector_z_ceil

                        if wall_portal != 0:
                            portal_camera_pos_x, portal_camera_pos_y, portal_camera_angle, portal_camera_angle_sin, portal_camera_angle_cos = entry_camera_pos_x, entry_camera_pos_y, entry_camera_angle, entry_camera_angle_sin, entry_camera_angle_cos
                
                            portal_sector_z_floor = sectors_z_floor[wall_portal]
                            portal_sector_z_ceil = sectors_z_ceil[wall_portal]

                            portal_wall_z_floor0 = portal_sector_z_floor
                            portal_wall_z_floor1 = portal_sector_z_floor

                            portal_wall_z_ceil0 = portal_sector_z_ceil
                            portal_wall_z_ceil1 = portal_sector_z_ceil

                            portal_wall_a_id = walls_a_id[wall_portal_wall_id]
                            portal_wall_b_id = walls_b_id[wall_portal_wall_id]

                            portal_wall_a_x = vertices_x[portal_wall_a_id]
                            portal_wall_a_y = vertices_y[portal_wall_a_id]

                            portal_wall_b_x = vertices_x[portal_wall_b_id]
                            portal_wall_b_y = vertices_y[portal_wall_b_id]

                            portal_sector_slope_floor_wall_id = sectors_slope_floor_wall_id[wall_portal]
                            portal_sector_slope_floor_wall_a_x, portal_sector_slope_floor_wall_a_y, portal_sector_slope_floor_wall_a_id, portal_sector_slope_floor_wall_b_x, portal_sector_slope_floor_wall_b_y, portal_sector_slope_floor_wall_b_id, portal_sector_slope_floor_wall_portal, portal_sector_slope_floor_wall_portal_wall_id, portal_sector_slope_floor_wall_sector_id, portal_sector_slope_floor_wall_texture_id, portal_sector_slope_floor_wall_texture_id_up, portal_sector_slope_floor_wall_texture_id_down, portal_sector_slope_floor_wall_animation_id, portal_sector_slope_floor_wall_texture_offset_x, portal_sector_slope_floor_wall_texture_offset_y, portal_sector_slope_floor_wall_texture_offset_up_x, portal_sector_slope_floor_wall_texture_offset_up_y, portal_sector_slope_floor_wall_texture_offset_down_x, portal_sector_slope_floor_wall_texture_offset_down_y = get_wall(portal_sector_slope_floor_wall_id, walls_a_id, walls_b_id, walls_portal, walls_portal_wall_id, walls_sector_id, walls_texture_id, walls_texture_id_up, walls_texture_id_down, walls_animation_id, walls_texture_offset_x, walls_texture_offset_y, walls_texture_offset_up_x, walls_texture_offset_up_y, walls_texture_offset_down_x, walls_texture_offset_down_y, vertices_id, vertices_x, vertices_y)
                            portal_sector_slope_floor_start_x = portal_sector_slope_floor_wall_a_x
                            portal_sector_slope_floor_start_y = portal_sector_slope_floor_wall_a_y
                            portal_sector_slope_floor_end_x = sectors_slope_floor_end_x[wall_portal]
                            portal_sector_slope_floor_end_y = sectors_slope_floor_end_y[wall_portal]

                            portal_sector_slope_ceil_wall_id = sectors_slope_ceil_wall_id[entry_id]
                            portal_sector_slope_ceil_wall_a_x, portal_sector_slope_ceil_wall_a_y, portal_sector_slope_ceil_wall_a_id, portal_sector_slope_ceil_wall_b_x, portal_sector_slope_ceil_wall_b_y, portal_sector_slope_ceil_wall_b_id, portal_sector_slope_ceil_wall_portal, portal_sector_slope_ceil_wall_portal_wall_id, portal_sector_slope_ceil_wall_sector_id, portal_sector_slope_ceil_wall_texture_id, portal_sector_slope_ceil_wall_texture_id_up, portal_sector_slope_ceil_wall_texture_id_down, portal_sector_slope_ceil_wall_animation_id, portal_sector_slope_ceil_wall_texture_offset_x, portal_sector_slope_ceil_wall_texture_offset_y, portal_sector_slope_ceil_wall_texture_offset_up_x, portal_sector_slope_ceil_wall_texture_offset_up_y, portal_sector_slope_ceil_wall_texture_offset_down_x, portal_sector_slope_ceil_wall_texture_offset_down_y = get_wall(portal_sector_slope_ceil_wall_id, walls_a_id, walls_b_id, walls_portal, walls_portal_wall_id, walls_sector_id, walls_texture_id, walls_texture_id_up, walls_texture_id_down, walls_animation_id, walls_texture_offset_x, walls_texture_offset_y, walls_texture_offset_up_x, walls_texture_offset_up_y, walls_texture_offset_down_x, walls_texture_offset_down_y, vertices_id, vertices_x, vertices_y)
                            portal_sector_slope_ceil_start_x = portal_sector_slope_ceil_wall_a_x
                            portal_sector_slope_ceil_start_y = portal_sector_slope_ceil_wall_a_y
                            portal_sector_slope_ceil_end_x = sectors_slope_ceil_end_x[wall_portal]
                            portal_sector_slope_ceil_end_y = sectors_slope_ceil_end_y[wall_portal]

                            if wall_a_x != portal_wall_a_x and wall_a_y != portal_wall_a_y and wall_b_x != portal_wall_b_x and wall_b_y != portal_wall_b_y:
                                portal_camera_pos_x, portal_camera_pos_y, portal_camera_angle, portal_camera_angle_sin, portal_camera_angle_cos = get_portal_camera_transforms(entry_camera_pos_x, entry_camera_pos_y, entry_camera_angle, wall_a_x, wall_a_y, wall_b_x, wall_b_y, portal_wall_a_x, portal_wall_a_y, portal_wall_b_x, portal_wall_b_y)

                            oop0_x, oop0_y = world_pos_to_local(portal_camera_pos_x, portal_camera_pos_y, portal_camera_angle_sin, portal_camera_angle_cos, portal_wall_b_x, portal_wall_b_y)
                            oop1_x, oop1_y = world_pos_to_local(portal_camera_pos_x, portal_camera_pos_y, portal_camera_angle_sin, portal_camera_angle_cos, portal_wall_a_x, portal_wall_a_y)

                            ccp0_x, ccp0_y = oop0_x, oop0_y
                            ccp1_x, ccp1_y = oop1_x, oop1_y

                            if ccp0_y <= 0 and ccp1_y <= 0: continue

                            aap0 = get_angle(0, 0, ccp0_x, ccp0_y)
                            aap1 = get_angle(0, 0, ccp1_x, ccp1_y)

                            ccp0_x, ccp0_y, ccp1_x, ccp1_y, aap0, aap1 = near_clip(ccp0_x, ccp0_y, ccp1_x, ccp1_y, aap0, aap1, znl_x, znl_y, zfl_x, zfl_y, znr_x, znr_y, zfr_x, zfr_y)

                            portal_sector_walls = full(WALL_MAX, 0, dtype=int32)

                            for portal_wall_id in walls_id:
                                portal_wall_sector_id = walls_sector_id[portal_wall_id]

                                if portal_wall_sector_id == wall_portal:
                                    portal_sector_walls[portal_wall_id] = portal_wall_id
                                    sectors_walls[wall_portal][portal_wall_id] = portal_wall_id

                            portal_sector_slope_floor_z = sectors_slope_floor_z[wall_portal]
                            portal_sector_slope_ceil_z = sectors_slope_ceil_z[wall_portal]

                            local_portal_sector_slope_floor_start_x, local_portal_sector_slope_floor_start_y = world_pos_to_local(portal_camera_pos_x, portal_camera_pos_y, portal_camera_angle_sin, portal_camera_angle_cos, portal_sector_slope_floor_start_x, portal_sector_slope_floor_start_y)
                            local_portal_sector_slope_floor_end_x, local_portal_sector_slope_floor_end_y = world_pos_to_local(portal_camera_pos_x, portal_camera_pos_y, portal_camera_angle_sin, portal_camera_angle_cos, portal_sector_slope_floor_end_x, portal_sector_slope_floor_end_y)

                            if portal_sector_slope_floor_z != 0:
                                portal_wall_z_floor0, portal_wall_z_floor1 = get_z_on_slope(local_portal_sector_slope_floor_start_x, local_portal_sector_slope_floor_start_y, local_portal_sector_slope_floor_end_x, local_portal_sector_slope_floor_end_y, ccp0_x, ccp0_y, ccp1_x, ccp1_y, portal_sector_z_floor, portal_sector_slope_floor_z)
 
                            local_portal_sector_slope_ceil_start_x, local_portal_sector_slope_ceil_start_y = world_pos_to_local(portal_camera_pos_x, portal_camera_pos_y, portal_camera_angle_sin, portal_camera_angle_cos, portal_sector_slope_ceil_start_x, portal_sector_slope_ceil_start_y)
                            local_portal_sector_slope_ceil_end_x, local_portal_sector_slope_ceil_end_y = world_pos_to_local(portal_camera_pos_x, portal_camera_pos_y, portal_camera_angle_sin, portal_camera_angle_cos, portal_sector_slope_ceil_end_x, portal_sector_slope_ceil_end_y)

                            if portal_sector_slope_ceil_z != 0:
                                portal_wall_z_ceil0, portal_wall_z_ceil1 = get_z_on_slope(local_portal_sector_slope_ceil_start_x, local_portal_sector_slope_ceil_start_y, local_portal_sector_slope_ceil_end_x, local_portal_sector_slope_ceil_end_y, ccp0_x, ccp0_y, ccp1_x, ccp1_y, portal_sector_z_ceil, portal_sector_slope_ceil_z)

                        yf0, yc0, txd, yfd, ycd, iz0, iz1, u0_z0, u1_z1, yyf0, yyfd, yyc0, yycd, nyyf0, nyyfd, nyyc0, nyycd = screen_space_transform(cp0_y, cp1_y, sector_z_floor_total, sector_z_ceil_total, camera_pos_z, tx1, tx0, wall_u0, wall_u1, wall_z_floor0, wall_z_floor1, wall_z_ceil0, wall_z_ceil1, portal_wall_z_floor0, portal_wall_z_floor1, portal_wall_z_ceil0, portal_wall_z_ceil1, viewport_height_2)

                        if engine_state != "DEBUG":
                            if sector_slope_floor_z == 0:
                                    screen = render_unsloped_visplane(screen, True, entry_camera_pos_x, entry_camera_pos_y, camera_pos_z, entry_camera_angle, camera_fog_distance, sector_z_floor_total, txd, tx0, yyfd, yyf0, y_lo, y_hi, x0, x1, textures_sheet, sector_floor_texture_x_coordinate, sector_floor_texture_width, sector_floor_texture_height, sector_light_factor, viewport_height, viewport_height_2, viewport_width, viewport_ratio_width_i, viewport_projection_plane_distance, viewport_x, viewport_y)

                            if sector_slope_ceil_z == 0:
                                    screen = render_unsloped_visplane(screen, False, entry_camera_pos_x, entry_camera_pos_y, camera_pos_z, entry_camera_angle, camera_fog_distance, sector_z_ceil_total, txd, tx0, yycd, yyc0, y_lo, y_hi, x0, x1, textures_sheet, sector_ceil_texture_x_coordinate, sector_ceil_texture_width, sector_ceil_texture_height, sector_light_factor, viewport_height, viewport_height_2, viewport_width, viewport_ratio_width_i, viewport_projection_plane_distance, viewport_x, viewport_y)
                
                        y_lo, y_hi, screen = process_wall_rendering(screen, textures_sheet, camera_fog_distance, wall_portal, wall_length, sector_height, wall_texture_id, sector_slope_floor_z, sector_slope_ceil_z, sector_z_floor, sector_z_ceil, sector_slope_floor_wall_a_x, sector_slope_floor_wall_a_y, sector_slope_floor_end_x, sector_slope_floor_end_y, sector_slope_ceil_wall_a_x, sector_slope_ceil_wall_a_y, sector_slope_ceil_end_x, sector_slope_ceil_end_y, x0, x1, y_hi, y_lo, engine_state, sector_z_floor_total, camera_pos_z, camera_angle, sector_light_factor, entry_camera_pos_x, entry_camera_pos_y, nyyfd, nyyf0, nyycd, nyyc0, txd, tx0, yfd, yf0, ycd, yc0, iz0, iz1, yyfd, yyf0, yycd, yyc0, sector_texture_floor_id, sector_texture_ceil_id, u0_z0, u1_z1, wall_texture_width, wall_texture_height, wall_texture_offset_x, wall_texture_offset_y, wall_texture_x_coordinate, wall_texture_up_width, wall_texture_up_height, wall_texture_up_offset_x, wall_texture_up_offset_y, wall_texture_up_x_coordinate, wall_texture_down_width, wall_texture_down_height, wall_texture_down_offset_x, wall_texture_down_offset_y, wall_texture_down_x_coordinate, skybox_texture_x_coordinate, skybox_texture_width, skybox_texture_height, entry_camera_angle_sin, entry_camera_angle_cos, sector_ceil_texture_x_coordinate, sector_ceil_texture_width, sector_ceil_texture_height, sector_floor_texture_x_coordinate, sector_floor_texture_width, sector_floor_texture_height, viewport_height, viewport_width, viewport_ratio_width_i, viewport_ratio_height, viewport_x, viewport_y, op0_x, op0_y, op1_x, op1_y, skyboxes_sheet, wall_texture_id_down, wall_texture_id_up)
                        
                        if (sector_draw >= QUEUE_MAX): print("WARNING 001: Sector-Queue ist überlastet.")
                        elif wall_portal != 0:
                                
                                if wall_texture_id != 0:

                                    for x in range(viewport_width):
                                        masked_queue_y_lo[x][masked_queue_n] = y_lo[x]
                                        masked_queue_y_hi[x][masked_queue_n] = y_hi[x]

                                    masked_queue_length[masked_queue_n] = wall_length
                                    masked_queue_type[masked_queue_n] = 1
                                    masked_queue_x0[masked_queue_n] = x0
                                    masked_queue_x1[masked_queue_n] = x1
                                    masked_queue_txd[masked_queue_n] = txd
                                    masked_queue_tx0[masked_queue_n] = tx0
                                    masked_queue_iz0[masked_queue_n] = iz0
                                    masked_queue_iz1[masked_queue_n] = iz1
                                    masked_queue_u0_z0[masked_queue_n] = u0_z0
                                    masked_queue_u1_z1[masked_queue_n] = u1_z1
                                    masked_queue_yfd[masked_queue_n] = yfd
                                    masked_queue_yf0[masked_queue_n] = yf0
                                    masked_queue_ycd[masked_queue_n] = ycd
                                    masked_queue_yc0[masked_queue_n] = yc0
                                    masked_queue_yyfd[masked_queue_n] = yyfd
                                    masked_queue_yyf0[masked_queue_n] = yyf0
                                    masked_queue_yycd[masked_queue_n] = yycd
                                    masked_queue_yyc0[masked_queue_n] = yyc0
                                    masked_queue_sector_z_floor_total[masked_queue_n] = sector_z_floor_total
                                    masked_queue_sector_z_ceil_total[masked_queue_n] = sector_z_ceil_total
                                    masked_queue_texture_offset_x[masked_queue_n] = wall_texture_offset_x
                                    masked_queue_texture_offset_y[masked_queue_n] = wall_texture_offset_y
                                    masked_queue_texture_width[masked_queue_n] = wall_texture_width
                                    masked_queue_texture_height[masked_queue_n] = wall_texture_height
                                    masked_queue_texture_x_coordinate[masked_queue_n] = wall_texture_x_coordinate
                                    masked_queue_sector_light_factor[masked_queue_n] = sector_light_factor

                                    masked_queue_n += 1
                                
                                sector_queue_id[sector_queue_n] = wall_portal
                                sector_queue_wall_id[sector_queue_n] = wall_portal_wall_id
                                sector_queue_x0[sector_queue_n] = x0
                                sector_queue_x1[sector_queue_n] = x1
                                sector_queue_camera_pos_x[sector_queue_n] = portal_camera_pos_x
                                sector_queue_camera_pos_y[sector_queue_n] = portal_camera_pos_y
                                sector_queue_camera_angle[sector_queue_n] = portal_camera_angle
                                sector_queue_camera_angle_sin[sector_queue_n] = portal_camera_angle_sin
                                sector_queue_camera_angle_cos[sector_queue_n] = portal_camera_angle_cos
                                sector_queue_n += 1

                else: continue
    
    for mask_n in range(masked_queue_n):
        mask_n = masked_queue_n-mask_n-1

        mask_y_lo = full(viewport_width, 0, dtype=int32)
        mask_y_hi = full(viewport_width, viewport_height, dtype=int32)

        for x in range(viewport_width):
            mask_y_lo[x] = masked_queue_y_lo[x][mask_n]
            mask_y_hi[x] = masked_queue_y_hi[x][mask_n]

        mask_length = masked_queue_length[mask_n]
        mask_sector_light_factor = masked_queue_sector_light_factor[mask_n]
        mask_sector_z_floor_total = masked_queue_sector_z_floor_total[mask_n]
        mask_sector_z_ceil_total = masked_queue_sector_z_ceil_total[mask_n]
        mask_type = masked_queue_type[mask_n]
        mask_texture_x_coordinate = masked_queue_texture_x_coordinate[mask_n]
        mask_texture_height = masked_queue_texture_height[mask_n]
        mask_texture_width = masked_queue_texture_width[mask_n]
        sector_height = mask_sector_z_ceil_total - mask_sector_z_floor_total

        if mask_type == 1:
            mask_texture_offset_x = masked_queue_texture_offset_x[mask_n]
            mask_texture_offset_y = masked_queue_texture_offset_x[mask_n]
            mask_x0 = masked_queue_x0[mask_n]
            mask_x1 = masked_queue_x1[mask_n]
            mask_txd = masked_queue_txd[mask_n]
            mask_tx0 = masked_queue_tx0[mask_n]
            mask_iz0 = masked_queue_iz0[mask_n]
            mask_iz1 = masked_queue_iz1[mask_n]
            mask_u0_z0 = masked_queue_u0_z0[mask_n]
            mask_u1_z1 = masked_queue_u1_z1[mask_n]
            mask_yfd = masked_queue_yfd[mask_n]
            mask_yf0 = masked_queue_yf0[mask_n]
            mask_ycd = masked_queue_ycd[mask_n]
            mask_yc0 = masked_queue_yc0[mask_n]
            mask_yyfd = masked_queue_yyfd[mask_n]
            mask_yyf0 = masked_queue_yyf0[mask_n]
            mask_yycd = masked_queue_yycd[mask_n]
            mask_yyc0 = masked_queue_yyc0[mask_n]
        
            screen = process_mask_rendering(screen, mask_x0, mask_x1, mask_length, mask_txd, mask_tx0, mask_u0_z0, mask_u1_z1, mask_iz0, mask_iz1, mask_yfd, mask_yf0, mask_ycd, mask_yc0, mask_yyfd, mask_yyf0, mask_yycd, mask_yyc0, mask_y_lo, mask_y_hi, mask_sector_light_factor, mask_sector_z_floor_total, mask_sector_z_ceil_total, engine_state, viewport_width, viewport_height, sector_height, camera_pos_z, entry_camera_pos_x, entry_camera_pos_y, camera_fog_distance, viewport_ratio_width_i, viewport_ratio_height, viewport_x, viewport_y, textures_sheet, mask_texture_width, mask_texture_height, mask_texture_offset_x, mask_texture_offset_y, mask_texture_x_coordinate)
        
        elif mask_type == 2:
            mask_billboard_a_x = masked_queue_billboard_a_x[mask_n]
            mask_billboard_a_y = masked_queue_billboard_a_y[mask_n]
            mask_billboard_b_x = masked_queue_billboard_b_x[mask_n]
            mask_billboard_b_y = masked_queue_billboard_b_y[mask_n]
            mask_billboard_z = masked_queue_billboard_z[mask_n]
            mask_billboard_size_x = masked_queue_billboard_size_x[mask_n]
            mask_billboard_size_y = masked_queue_billboard_size_y[mask_n]
            mask_camera_position_x = masked_queue_camera_position_x[mask_n]
            mask_camera_position_y = masked_queue_camera_position_y[mask_n]
            mask_camera_angle = masked_queue_camera_angle[mask_n]
            mask_camera_angle_cos = masked_queue_camera_angle_cos[mask_n]
            mask_camera_angle_sin = masked_queue_camera_angle_sin[mask_n]
            mask_entry_x0 = masked_queue_entry_x0[mask_n]
            mask_entry_x1 = masked_queue_entry_x1[mask_n]
            mask_texture_offset_x = 0
            mask_texture_offset_y = 0

            if point_side(mask_camera_position_x, mask_camera_position_y, mask_billboard_a_x, mask_billboard_a_y, mask_billboard_b_x, mask_billboard_b_y) < 0:
                op0_x, op0_y = world_pos_to_local(mask_camera_position_x, mask_camera_position_y, mask_camera_angle_sin, mask_camera_angle_cos, mask_billboard_a_x, mask_billboard_a_y)
                op1_x, op1_y = world_pos_to_local(mask_camera_position_x, mask_camera_position_y, mask_camera_angle_sin, mask_camera_angle_cos, mask_billboard_b_x, mask_billboard_b_y)

                cp0_x, cp0_y = op0_x, op0_y
                cp1_x, cp1_y = op1_x, op1_y

                if (cp0_y <= 0 and cp1_y <= 0): continue

                ap0 = get_angle(0, 0, cp0_x, cp0_y)
                ap1 = get_angle(0, 0, cp1_x, cp1_y)

                cp0_x, cp0_y, cp1_x, cp1_y, ap0, ap1 = near_clip(cp0_x, cp0_y, cp1_x, cp1_y, ap0, ap1, znl_x, znl_y, zfl_x, zfl_y, znr_x, znr_y, zfr_x, zfr_y)

                if ap0 < ap1: continue
                if (ap0 < -(HFOV_2) and ap1 < -(HFOV_2)) or (ap0 > +(HFOV_2) and ap1 > +(HFOV_2)): continue

                mask_tx0 = screen_angle_to_x(ap0, viewport_width_2)
                mask_tx1 = screen_angle_to_x(ap1, viewport_width_2)

                if mask_tx0 > mask_entry_x1: continue
                if mask_tx1 < mask_entry_x0: continue

                mask_x0 = clamp(mask_tx0, mask_entry_x0, mask_entry_x1)
                mask_x1 = clamp(mask_tx1, mask_entry_x0, mask_entry_x1)

                mask_u0 = 0
                mask_u1 = 1

                if cp0_x != op0_x:
                    mask_u0 = extension_factor(op0_x, op0_y, op1_x, op1_y, cp0_x, cp0_y, 0)

                if cp1_x != op1_x:
                    mask_u1 = extension_factor(op0_x, op0_y, op1_x, op1_y, cp1_x, cp1_y, 1)

                mask_z_floor = mask_billboard_z
                mask_z_ceil = mask_billboard_z + mask_billboard_size_y

                mask_yf0, mask_yc0, mask_txd, mask_yfd, mask_ycd, mask_iz0, mask_iz1, mask_u0_z0, mask_u1_z1, mask_yyf0, mask_yyfd, mask_yyc0, mask_yycd, mask_nyyf0, mask_nyyfd, mask_nyyc0, mask_nyycd = screen_space_transform(cp0_y, cp1_y, mask_z_floor, mask_z_ceil, camera_pos_z, mask_tx1, mask_tx0, mask_u0, mask_u1, mask_z_floor, mask_z_floor, mask_z_ceil, mask_z_ceil, 0, 0, 0, 0, viewport_height_2)
                
                screen = process_mask_rendering(screen, mask_x0, mask_x1, mask_length, mask_txd, mask_tx0, mask_u0_z0, mask_u1_z1, mask_iz0, mask_iz1, mask_yfd, mask_yf0, mask_ycd, mask_yc0, mask_yyfd, mask_yyf0, mask_yycd, mask_yyc0, mask_y_lo, mask_y_hi, mask_sector_light_factor, mask_z_floor, mask_z_ceil, engine_state, viewport_width, viewport_height, mask_billboard_size_y, camera_pos_z, mask_camera_position_x, mask_camera_position_y, camera_fog_distance, viewport_ratio_width_i, viewport_ratio_height, viewport_x, viewport_y, sprites_sheet, mask_texture_width, mask_texture_height, mask_texture_offset_x, mask_texture_offset_y, mask_texture_x_coordinate)

    if faced_wall_id != 0:
        if texture_select_mode == "WALL":
            wall_texture_id = walls_texture_id[faced_wall_id]
            wall_texture_selected = False
            wall_texture_id_up = walls_texture_id_up[faced_wall_id]
            wall_texture_selected_up = False
            wall_texture_id_down = walls_texture_id_down[faced_wall_id]
            wall_texture_selected_down = False
            wall_portal = walls_portal[faced_wall_id]

            if wall_portal != 0:
                if texture_slot_id == 0:
                    walls_texture_id[faced_wall_id] = change_texture(keys, prev_keys, wall_texture_id, textures_n)
                    wall_texture_selected = True
                elif texture_slot_id == 1:
                    walls_texture_id_down[faced_wall_id] = change_texture(keys, prev_keys, wall_texture_id_down, textures_n)
                    wall_texture_selected_down = True
                elif texture_slot_id == 2:
                    walls_texture_id_up[faced_wall_id] = change_texture(keys, prev_keys, wall_texture_id_up, textures_n)
                    wall_texture_selected_up = True
            else:
                walls_texture_id[faced_wall_id] = change_texture(keys, prev_keys, wall_texture_id, textures_n)
                wall_texture_selected = True

            if wall_texture_id != 0:

                wall_texture_width = textures_width[wall_texture_id]
                wall_texture_height = textures_height[wall_texture_id]
                wall_texture_x_coordinate = textures_x_coordinate[wall_texture_id]

                screen = texture_slot(screen, 64, wall_texture_width, wall_texture_height, wall_texture_x_coordinate, textures_sheet, viewport_x, viewport_y, viewport_width, viewport_height, 10, 10, wall_texture_selected)
            
            if wall_portal != 0:

                if wall_texture_id_up != 0:

                    wall_texture_up_width = textures_width[wall_texture_id_up]
                    wall_texture_up_height = textures_height[wall_texture_id_up]
                    wall_texture_up_x_coordinate = textures_x_coordinate[wall_texture_id_up]

                    screen = texture_slot(screen, 64, wall_texture_up_width, wall_texture_up_height, wall_texture_up_x_coordinate, textures_sheet, viewport_x, viewport_y, viewport_width, viewport_height, 80, 10, wall_texture_selected_up)

                if wall_texture_id_down != 0:

                    wall_texture_down_width = textures_width[wall_texture_id_down]
                    wall_texture_down_height = textures_height[wall_texture_id_down]
                    wall_texture_down_x_coordinate = textures_x_coordinate[wall_texture_id_down]

                    screen = texture_slot(screen, 64, wall_texture_down_width, wall_texture_down_height, wall_texture_down_x_coordinate, textures_sheet, viewport_x, viewport_y, viewport_width, viewport_height, 150, 10, wall_texture_selected_down)

        elif texture_select_mode == "SECTOR":
            sector_floor_texture_id = sectors_floor_texture_id[camera_sector]
            sector_ceil_texture_id = sectors_ceil_texture_id[camera_sector]

            sector_floor_texture_selected = False
            sector_ceil_texture_selected = False

            if texture_slot_id == 0:
                sectors_floor_texture_id[camera_sector] = change_texture(keys, prev_keys, sector_floor_texture_id, textures_n)
                sector_floor_texture_selected = True
            elif texture_slot_id == 1:
                sectors_ceil_texture_id[camera_sector] = change_texture(keys, prev_keys, sector_ceil_texture_id, textures_n)
                sector_ceil_texture_selected = True
            elif texture_slot_id == 2:
                texture_slot_id = 1
            
            if sector_floor_texture_id != 0:
                sector_floor_texture_width = textures_width[sector_floor_texture_id]
                sector_floor_texture_height = textures_height[sector_floor_texture_id]
                sector_floor_texture_x_coordinate = textures_x_coordinate[sector_floor_texture_id]
                
                screen = texture_slot(screen, 64, sector_floor_texture_width, sector_floor_texture_height, sector_floor_texture_x_coordinate, textures_sheet, viewport_x, viewport_y, viewport_width, viewport_height, 80, 10, sector_floor_texture_selected)

            if sector_ceil_texture_id != 0:
                sector_ceil_texture_width = textures_width[sector_ceil_texture_id]
                sector_ceil_texture_height = textures_height[sector_ceil_texture_id]
                sector_ceil_texture_x_coordinate = textures_x_coordinate[sector_ceil_texture_id]

                screen = texture_slot(screen, 64, sector_ceil_texture_width, sector_ceil_texture_height, sector_ceil_texture_x_coordinate, textures_sheet, viewport_x, viewport_y, viewport_width, viewport_height, 10, 10, sector_ceil_texture_selected)

        elif texture_select_mode == "SKYBOX":
            sector_skybox_id = sectors_skybox_id[camera_sector]
            sector_skybox_id = change_skybox(keys, prev_keys, sector_skybox_id, skyboxes_n)

            if sector_skybox_id == 0:
                sector_skybox_id = 1
            
            sectors_skybox_id[camera_sector] = sector_skybox_id

            sector_skybox_width = skyboxes_width[sector_skybox_id]
            sector_skybox_height = skyboxes_height[sector_skybox_id]
            sector_skybox_x_coordinate = skyboxes_x_coordinates[sector_skybox_id]

            screen = texture_slot(screen, 64, sector_skybox_width, sector_skybox_height, sector_skybox_x_coordinate, skyboxes_sheet, viewport_x, viewport_y, viewport_width, viewport_height, 10, 10, True)

    return screen, sectors_slope_floor_end_x, sectors_slope_floor_end_y, sectors_slope_ceil_end_x, sectors_slope_ceil_end_y, sectors_walls, walls_texture_id, texture_slot_id

@njit(fastmath=True)
def change_texture(keys:array, prev_keys:array, texture_id:int, textures_n:int):
    if keys[N] == 1 and prev_keys[N] == 0:
        if texture_id != 0:
            texture_id -= 1
        else:
            texture_id = textures_n
    
    if keys[M] == 1 and prev_keys[M] == 0:
        if texture_id != textures_n:
            texture_id += 1
        else:
            texture_id = 0
    
    return texture_id


@njit(fastmath=True)
def change_skybox(keys:array, prev_keys:array, skybox_id:int, skyboxes_n:int):
    if keys[N] == 1 and prev_keys[N] == 0:
        if skybox_id != 1:
            skybox_id -= 1
        else:
            skybox_id = skyboxes_n
    
    if keys[M] == 1 and prev_keys[M] == 0:
        if skybox_id != skyboxes_n:
            skybox_id += 1
        else:
            skybox_id = 1
    
    return skybox_id

@njit(fastmath=True)
def texture_slot(screen:array, tile_size:int, texture_width:int, texture_height:int, texture_x_coordinate:int, textures_sheet:array, viewport_x:int, viewport_y:int, viewport_width:int, viewport_height:int, x:int, y:int, selected=False)->array:

    scale = tile_size / texture_width

    scaled_wall_texture_height_down = texture_height * scale

    pos_x, pos_y = viewport_width - tile_size - x, viewport_height - scaled_wall_texture_height_down - y

    if selected == True:
        screen = rectangle(screen, pos_x-1, pos_y-1, pos_x+1+tile_size, pos_y+1+scaled_wall_texture_height_down, (255,255,255), viewport_x, viewport_y, viewport_width, viewport_height)
    
    screen = texture(screen, textures_sheet, texture_width, texture_height, texture_x_coordinate, pos_x, pos_y, viewport_x, viewport_y, viewport_width, viewport_height, scale)

    return screen

@njit(fastmath=True)
def rectangle(screen:array, _v1_x:int, _v1_y:int, _v2_x:int, _v2_y:int, color:tuple, viewport_x:int, viewport_y:int, viewport_width:int, viewport_height:int)->array:
    x1, y1 = min(_v1_x, _v2_x), min(_v1_y, _v2_y)
    x2, y2 = max(_v1_x, _v2_x), max(_v1_y, _v2_y)
    
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            pixel(screen, x, y, color, viewport_x, viewport_y, viewport_width, viewport_height)
    
    return screen

@njit(fastmath=False)
def process_mask_rendering(screen, x0, x1, mask_length, txd, tx0, u0_z0, u1_z1, iz0, iz1, yfd, yf0, ycd, yc0, yyfd, yyf0, yycd, yyc0, y_lo, y_hi, light_factor, z_floor_total, z_ceil_total, engine_state, viewport_width, viewport_height, sector_height, camera_pos_z, camera_pos_x, camera_pos_y, camera_fog_distance, viewport_ratio_width_i, viewport_ratio_height, viewport_x, viewport_y, textures_sheet, wall_texture_width, wall_texture_height, wall_texture_offset_x, wall_texture_offset_y, wall_texture_coordinate)->array:
    
    for x in range(x0, x1 + 1):

        if txd != 0:
            xp = relative_change(x, tx0, txd)
        else:
            xp = 0
        
        u = tiled_u_coordinate(mask_length, xp, u0_z0, u1_z1, iz0, iz1)

        tyf = int(linear_function(xp, yfd, yf0))
        tyc = int(linear_function(xp, ycd, yc0))

        tyyf = int(linear_function(xp, yyfd, yyf0))
        tyyc = int(linear_function(xp, yycd, yyc0))

        yyf = clamp(tyyf, y_lo[x], y_hi[x])
        yyc = clamp(tyyc, y_lo[x], y_hi[x])

        if engine_state == "DEBUG":
            screen = debug_verline(
                screen, x, yyf, yyc,
                color_mult(DEBUG_WALL_COLOR, light_factor), viewport_height, viewport_width, viewport_x, viewport_y
            )
            
        else:
            screen = texture_verline(screen, textures_sheet, wall_texture_width, wall_texture_height, wall_texture_offset_x, wall_texture_offset_y, wall_texture_coordinate, x, yyf, yyc, u, tyf, tyc, light_factor, z_floor_total, sector_height, camera_pos_z, tyyf, tyyc, camera_pos_x, camera_pos_y, camera_fog_distance, viewport_height, viewport_width, viewport_ratio_width_i, viewport_ratio_height, viewport_x, viewport_y) 

    return screen

@njit(fastmath=False)
def get_frame(ticks:int, animation_ms:float, animation_frames:int)->int:
    return int((ticks / animation_ms) % animation_frames) if animation_ms != 0 else 1

@njit(fastmath=True)
def get_z_on_slope(slope_start_x:float, slope_start_y:float, slope_end_x:float, slope_end_y:float, p0_x:float, p0_y:float, p1_x:float, p1_y:float, z:float, slope_z:float) -> tuple:
    sg0 = get_gradient(slope_start_x, slope_start_y, slope_end_x, slope_end_y, p0_x, p0_y)
    sg1 = get_gradient(slope_start_x, slope_start_y, slope_end_x, slope_end_y, p1_x, p1_y)

    z0 = z + (sg0 * slope_z)
    z1 = z + (sg1 * slope_z)

    return z0, z1

@njit(fastmath=True)
def get_gradient(_v1_x:float, _v1_y:float, _v2_x:float, _v2_y:float, _vt_x:float, _vt_y:float) -> float:
    t_x, t_y = closest_point_on_line_segment(_vt_x, _vt_y, _v1_x, _v1_y, _v2_x, _v2_y)

    d1 = distance(_v1_x, _v1_y, _v2_x, _v2_y)
    d2 = distance(t_x, t_y, _v2_x, _v2_y)
    
    return d2 / d1 if d1 != 0 else 0

@njit(fastmath=True)
def closest_point_on_line_segment(p_x:float, p_y:float, x1_x:float, x1_y:float, x2_x:float, x2_y:float) -> tuple:
    dx, dy = x2_x - x1_x, x2_y - x1_y
    denominator = dx**2 + dy**2

    if denominator != 0:
        t = max(0, min(1, dot(p_x - x1_x, p_y - x1_y, dx, dy) / denominator))
        return x1_x + t * dx, x1_y + t * dy
    else:
        return x1_x, x1_y
    
@njit(fastmath=True, parallel=True)
def render_unsloped_visplane(screen:array, isFloor:bool, camera_pos_x:float, camera_pos_y:float, camera_pos_z:float, camera_angle:float, camera_fog_distance:float, z:float, txd:int, tx0:int, yyfd:int, yyf0:int, y_lo:array, y_hi:array, x_0:int, x_1:int, textures_sheet:array, texture_x_coordinate:int, texture_width:int, texture_height:int, light_factor:float, viewport_height:int, viewport_height_2:int, viewport_width:int, viewport_ratio_width_i:float, viewport_projection_plane_distance:float, viewport_x:int, viewport_y:int):

    y_start = 0 if isFloor else viewport_height_2
    y_end = viewport_height_2 if isFloor else viewport_height

    cos_HFOV_2 = cos(HFOV_2)
    cos_camera_angle_HFOV_2 = cos(camera_angle + HFOV_2)
    sin_camera_angle_HFOV_2 = sin(camera_angle + HFOV_2)
    cos_camera_angle_HFOV = cos(camera_angle - HFOV)
    sin_camera_angle_HFOV = sin(camera_angle - HFOV)

    for y in prange(y_start, y_end):
        yy = viewport_height - y
                    
        if not (y - viewport_height_2) == 0:
            dist = (camera_pos_z - z + (viewport_ratio_width_i*(camera_pos_z-z))) * viewport_projection_plane_distance / (y - viewport_height_2) if not (y - viewport_height_2) == 0 else 1
        
        else: dist = inf

        length = dist / cos_HFOV_2
        i_x = -camera_pos_x + length * cos_camera_angle_HFOV_2
        i_y = camera_pos_y - length * sin_camera_angle_HFOV_2
        step = dist / viewport_projection_plane_distance
        stepX = step * cos_camera_angle_HFOV
        stepY = -step * sin_camera_angle_HFOV

        fog_factor = round(1 - (distance3D(camera_pos_x, camera_pos_y, camera_pos_z, -i_x, i_y, z) * camera_fog_distance), 2)

        for x in range(0, viewport_width+1):
            if txd != 0:
                xp = relative_change(x, tx0, txd)
            else:
                xp = 0

            tyyf = int(linear_function(xp, yyfd, yyf0))
            yyf = clamp(tyyf, y_lo[x], y_hi[x])

            if x_0 <= x and x <= x_1:
                if (y_lo[x] <= y and y <= yyf and isFloor) or (y <= y_hi[x] and yyf <= y and not isFloor):

                    tx = int(((100) * -i_x) % texture_width) if texture_width != 0 else 0
                    ty = int(((100) * -i_y) % texture_height) if texture_height != 0 else 0

                    if (tx < 0): tx = 0

                    if (ty < 0): ty = 0

                    color = textures_sheet[texture_x_coordinate+tx][ty]

                    if check_keyColor(color):
                        pixel(screen, x, yy, color_mult(color, clamp(light_factor*fog_factor, 0,1)), viewport_x, viewport_y, viewport_width, viewport_height)

            i_x += stepX
            i_y += stepY

    return screen

@njit(fastmath=True)
def pixel(screen:array, x:int, y:int, color:tuple, offset_x:int, offset_y:int, dimensions_x:int, dimesions_y:int):
    if 0 <= x <= dimensions_x and 0 <= y <= dimesions_y:
        screen[x+offset_x][y+offset_y] = color

@njit(fastmath=True)
def render_sloped_visplane(screen: array, camera_pos_x: float, camera_pos_y: float, camera_pos_z: float, plane_point_x: float, plane_point_y: float, plane_point_z: float, plane_normal_x: float, plane_normal_y: float, plane_normal_z: float, x: int, y0: int, y1: int, textures_sheet: array, texture_width: int, texture_height: int, texture_x_coordinate: int, light_factor: float, camera_angle_sin: float, camera_angle_cos: float, camera_fog_distance: float, viewport_height:int, viewport_width:int, viewport_x:int, viewport_y:int):

    xcam = (2 * (x / viewport_width)) - 1

    ray_direction_x = camera_angle_cos + xcam * camera_angle_sin
    ray_direction_y = camera_angle_sin + xcam * -camera_angle_cos

    for y in range(y0, y1 + 1):
        yy = viewport_height - y

        ycam = (2 * (y / viewport_height)) - 1

        ray_direction_z = ycam
        i_x, i_y, i_z = ray_plane_intersection(camera_pos_x, camera_pos_y, camera_pos_z, ray_direction_x, ray_direction_y, ray_direction_z, plane_point_x, plane_point_y, plane_point_z, plane_normal_x, plane_normal_y, plane_normal_z)

        if isnan(i_x):
            continue

        fog_factor = 1 - (distance3D(camera_pos_x, camera_pos_y, camera_pos_z, i_x, i_y, i_z) * camera_fog_distance)

        plane_tangent_x = -plane_normal_y
        plane_tangent_y = plane_normal_x
        plane_tangent_z = 0
        
        plane_bitangent_x = plane_normal_z * plane_normal_x
        plane_bitangent_y = plane_normal_z * plane_normal_y
        plane_bitangent_z = -(plane_normal_x**2 + plane_normal_y**2)

        tangent_length = 1 / sqrt(plane_tangent_x**2 + plane_tangent_y**2 + plane_tangent_z**2) if sqrt(plane_tangent_x**2 + plane_tangent_y**2 + plane_tangent_z**2) != 0 else 0
        bitangent_length = 1 / sqrt(plane_bitangent_x**2 + plane_bitangent_y**2 + plane_bitangent_z**2) if sqrt(plane_bitangent_x**2 + plane_bitangent_y**2 + plane_bitangent_z**2) != 0 else 0

        plane_tangent_x = plane_tangent_x * tangent_length if tangent_length != 0 else 0
        plane_tangent_y = plane_tangent_y * tangent_length if tangent_length != 0 else 0
        plane_tangent_z = plane_tangent_z * tangent_length if tangent_length != 0 else 0
        
        plane_bitangent_x = plane_bitangent_x * bitangent_length if bitangent_length != 0 else 0
        plane_bitangent_y = plane_bitangent_y * bitangent_length if bitangent_length != 0 else 0
        plane_bitangent_z = plane_bitangent_z * bitangent_length if bitangent_length != 0 else 0

        u = (i_x - plane_point_x) * plane_tangent_x + (i_y - plane_point_y) * plane_tangent_y + (i_z - plane_point_z) * plane_tangent_z
        v = (i_x - plane_point_x) * plane_bitangent_x + (i_y - plane_point_y) * plane_bitangent_y + (i_z - plane_point_z) * plane_bitangent_z

        tx = int((u * 100) % texture_width) if texture_width != 0 else 0
        ty = int((v * 100) % texture_height) if texture_height != 0 else 0

        if tx < 0: tx = 0
        if ty < 0: ty = 0

        color = textures_sheet[texture_x_coordinate + tx][ty]

        if check_keyColor(color):
            pixel(screen, x, yy, color_mult(color, clamp(light_factor * fog_factor, 0, 1)), viewport_x, viewport_y, viewport_width, viewport_height)

    return screen

@njit(fastmath=True)
def ray_plane_intersection(ray_origin_x:float, ray_origin_y:float, ray_origin_z:float, ray_direction_x:float, ray_direction_y:float, ray_direction_z:float, plane_point_x:float, plane_point_y:float, plane_point_z:float, plane_normal_x:float, plane_normal_y:float, plane_normal_z:float):
    denominator = round(dot3D(plane_normal_x, plane_normal_y, plane_normal_z, ray_direction_x, ray_direction_y, ray_direction_z),4)
    if abs(denominator) < 1e-3:
        return nan,nan,nan
    
    p_x, p_y, p_z = plane_point_x - ray_origin_x, plane_point_y - ray_origin_y, plane_point_z - ray_origin_z
    t = dot3D(plane_normal_x, plane_normal_y, plane_normal_z, p_x, p_y, p_z) / denominator if denominator != 0 else 1
    if t < 0:
        return nan,nan,nan
    
    return ray_origin_x + ray_direction_x * t, ray_origin_y + ray_direction_y * t, ray_origin_z + ray_direction_z * t

@njit(fastmath=True)
def dot3D(_v_x:float, _v_y:float, _v_z:float, _v1_x:float, _v1_y:float, _v1_z:float):
        return _v_x * _v1_x + _v_y * _v1_y + _v_z * _v1_z

@njit(fastmath=True)
def distance3D(_v_x, _v_y, _v_z, _v1_x, _v1_y, _v1_z):
        return sqrt((_v1_x - _v_x)**2 + (_v1_y - _v_y)**2 + (_v1_z - _v_z)**2)

@njit(fastmath=True)
def screen_space_conversion(h:float, pos_z:float, _sy:float, viewport_height_2:int)->int:
    return (viewport_height_2) + int((h - pos_z) * _sy)

@njit(fastmath=True)
def ifnan(_x:float, _a:float)->float:                            
    return _x if not isnan(_x) else _a

@njit(fastmath=True)
def screen_space_transform(cp0_y:float, cp1_y:float, sector_z_floor:float, sector_z_ceil:float, camera_pos_z:float, tx1:int, tx0:int, u0:float, u1:float, wall_z_floor0:float, wall_z_floor1:float, wall_z_ceil0:float, wall_z_ceil1:float, portal_wall_z_floor0:float, portal_wall_z_floor1:float, portal_wall_z_ceil0:float, portal_wall_z_ceil1:float, viewport_height_2:int)->tuple:

    sy0 = ifnan((viewport_height_2) / cp0_y, 1e10) if cp0_y != 0 else 0
    sy1 = ifnan((viewport_height_2) / cp1_y, 1e10) if cp1_y != 0 else 0

    yf0 = screen_space_conversion(sector_z_floor, camera_pos_z, sy0, viewport_height_2)
    yc0 = screen_space_conversion(sector_z_ceil, camera_pos_z, sy0, viewport_height_2)
    yf1 = screen_space_conversion(sector_z_floor, camera_pos_z, sy1, viewport_height_2)
    yc1 = screen_space_conversion(sector_z_ceil, camera_pos_z, sy1, viewport_height_2)

    yyf0 = screen_space_conversion(wall_z_floor0, camera_pos_z, sy0, viewport_height_2)
    yyc0 = screen_space_conversion(wall_z_ceil0, camera_pos_z, sy0, viewport_height_2)
    yyf1 = screen_space_conversion(wall_z_floor1, camera_pos_z, sy1, viewport_height_2)
    yyc1 = screen_space_conversion(wall_z_ceil1, camera_pos_z, sy1, viewport_height_2)      

    nyyf0 = screen_space_conversion(portal_wall_z_floor0, camera_pos_z, sy0, viewport_height_2)
    nyyc0 = screen_space_conversion(portal_wall_z_ceil0, camera_pos_z, sy0, viewport_height_2)
    nyyf1 = screen_space_conversion(portal_wall_z_floor1, camera_pos_z, sy1, viewport_height_2)
    nyyc1 = screen_space_conversion(portal_wall_z_ceil1, camera_pos_z, sy1, viewport_height_2)                

    txd = tx1 - tx0
    yfd = yf1 - yf0
    ycd = yc1 - yc0

    yyfd = yyf1 - yyf0
    yycd = yyc1 - yyc0

    nyyfd = nyyf1 - nyyf0
    nyycd = nyyc1 - nyyc0

    z0 = cp0_y
    z1 = cp1_y

    iz0 = 1 / z0 if z0 != 0 else 1
    iz1 = 1 / z1 if z1 != 0 else 1

    u0_z0 = u0 * iz0 if iz0 != 0 else 1
    u1_z1 = u1 * iz1 if iz1 != 0 else 1

    return yf0, yc0, txd, yfd, ycd, iz0, iz1, u0_z0, u1_z1, yyf0, yyfd, yyc0, yycd, nyyf0, nyyfd, nyyc0, nyycd

@njit(fastmath=True)
def length(_vl_x:float, _vl_y:float)->float:                                
    return sqrt(dot(_vl_x, _vl_y, _vl_x, _vl_y))

@njit(fastmath=True)
def dot(_v0_x:float, _v0_y:float, _v1_x:float, _v1_y:float)->float:                              
    return (_v0_x * _v1_x) + (_v0_y * _v1_y)

@njit(fastmath=True)
def normalize3D(_v_x:float, _v_y:float, _v_z:float)->tuple:
        mag = magnitude3D(_v_x, _v_y, _v_z)
        if mag != 0:
            return  _v_x / mag, _v_y / mag, _v_z / mag
        else:
            return 0,0,0

@njit(fastmath=True)
def magnitude3D(_v_x:float, _v_y:float, _v_z:float)->float:
        return sqrt(_v_x**2 + _v_y**2 + _v_z**2)

@njit(fastmath=True)
def linear_function(_x:float, _k:float, _d:float)->float:
    return (_x * _k) + _d

@njit(fastmath=True)
def relative_change(_x:float, _x0:float, _dx:float)->float:
    return (_x - _x0) / _dx if _dx != 0 else 0

@njit(fastmath=True)
def u_coordinate(xp:float, u0_z0:float, u1_z1:float, iz0:float, iz1:float)->float:
    return (((1 - xp) * u0_z0) + (xp * u1_z1)) / (((1 - xp) * iz0) + (xp * iz1)) if not (((1 - xp) * iz0) + (xp * iz1)) == 0 else 1

@njit(fastmath=True)
def tiled_u_coordinate(wall_length:float, xp:float, u0_z0:float, u1_z1:float, iz0:float, iz1:float)->float:
    return wall_length * u_coordinate(xp, u0_z0, u1_z1, iz0, iz1) * 0.5

@njit(fastmath=True)
def color_mult(_c:tuple, _a:float)->tuple:
    return (_c[0] * _a, _c[1] * _a, _c[2] * _a)

@njit(fastmath=True)
def rad2deg(_d: float)->float:
    return _d * (180 / PI)

@njit(fastmath=True)
def sky_verline(screen:array, textures_sheet:array, texture_x_coordinate:int, texture_width:int, texture_height:int, camera_angle:float, x:int, y0:int, y1:int, viewport_height:int, viewport_width:int, viewport_x:int, viewport_y:int)->array:

    texture_offset = int(-camera_angle * 0.1 * texture_width) % texture_width if texture_width != 0 else 0
    tx = int((x + texture_offset) % texture_width) if texture_width != 0 else 0

    for y in range(y0, y1):
        yy = viewport_height - y
        ty = int((yy / viewport_height) * texture_height)

        pixel(screen, x, yy, textures_sheet[texture_x_coordinate + tx][ty], viewport_x, viewport_y, viewport_width, viewport_height)

    return screen

@njit(fastmath=True)
def debug_verline(screen:tuple, x:int, y0:int, y1:int, c:tuple, viewport_height:int, viewport_width:int, viewport_x:int, viewport_y:int)->array:
    for y in range(y0,y1+1):
        yy = viewport_height - y

        pixel(screen, x, yy, c, viewport_x, viewport_y, viewport_width, viewport_height)

    return screen

@njit(fastmath=True)
def v_coordinate(yy:float, y_ceil:float, y_floor:float)->float:
    return (1 - ((yy - y_ceil) / (y_floor - y_ceil))) * 0.5 if (y_floor - y_ceil) != 0 else 0

@njit(fastmath=True)
def tiled_v_coordinate(sector_height:float, cpos_z:float, yy:float, y_ceil:float, y_floor:float)->float:
    return sector_height * v_coordinate(yy, y_ceil, y_floor) - cpos_z

@njit(fastmath=True)
def texture_verline(screen:int, textures_sheet:array, texture_width:int, texture_height:int, texture_offset_x:float, texture_offset_y:float, texture_x_coordinate:int, x:int, y0:int, y1:int, u:float, y_floor:float, y_ceil:float, light_factor:float, z_floor:float, sector_height:float, camera_pos_z:float, tyyf:int, tyyc:int, camera_pos_x:float, camera_pos_y:float, camera_fog_distance:float, viewport_height:int, viewport_width:int, viewport_ratio_width_i:float, viewport_ratio_height:float, viewport_x:int, viewport_y:int):
    
    tx = floor((u * 200) + texture_offset_x)
    tx %= texture_width

    hyd = y_ceil - y_floor

    for y in range(y0, y1+1):

        if hyd != 0:
            yp = relative_change(y, y_floor, hyd)
        else:
            yp = 0

        i_z = z_floor + sector_height * yp

        yy = viewport_height - y
        v = tiled_v_coordinate(sector_height, camera_pos_z, yy, y_ceil, y_floor)

        ty = floor((((v + z_floor) + ((v + z_floor) * viewport_ratio_width_i)) * (200 * (viewport_ratio_height))) + texture_offset_y)
        ty %= texture_height

        if tx < texture_width and ty < texture_height:
            if y >= tyyf and y <= tyyc:
                color = textures_sheet[texture_x_coordinate+tx][ty]

                if check_keyColor(color):
                    pixel(screen, x, yy, color_mult(color, clamp(light_factor, 0, 1)), viewport_x, viewport_y, viewport_width, viewport_height)
                
    return screen

@njit(fastmath=True)
def sky_pixel(screen:array, textures_sheet:array, texture_x_coordinate:int, texture_width:int, texture_height:int, camera_angle:float, x:int, y:int, viewport_height:int, viewport_width:int, viewport_x:int, viewport_y:int):
    pass

@njit(fastmath=True)
def check_keyColor(_c:tuple)->bool:
    if not (_c[0] == KEY_COLOR[0] and _c[1] == KEY_COLOR[1] and _c[2] == KEY_COLOR[2]):
        return True
    else:
        return False
    
@njit(fastmath=True)
def get_slope_point(sector_slope_wall_a_x:float, sector_slope_wall_a_y:float, sector_slope_wall_b_x:float, sector_slope_wall_b_y:float, sector_walls:array, walls_a_id:array, walls_b_id:array, vertices_x:array, vertices_y:array):
    a0 = get_angle(sector_slope_wall_a_x, sector_slope_wall_a_y, sector_slope_wall_b_x, sector_slope_wall_b_y)
    a1 = a0 + PI_2

    max_y = 0
    furthest_point_x, furthest_point_y = nan, nan

    for wall_id in sector_walls:
        if wall_id != 0:
            wall_a_id = walls_a_id[wall_id]
            wall_b_id = walls_b_id[wall_id]
            wall_a_x, wall_a_y = vertices_x[wall_a_id], vertices_y[wall_a_id]
            wall_b_x, wall_b_y = vertices_x[wall_b_id], vertices_y[wall_b_id]

            local_a_x, local_a_y = world_pos_to_local(sector_slope_wall_a_x, sector_slope_wall_a_y, sin(a0), cos(a0), wall_a_x, wall_a_y)
            local_b_x, local_b_y = world_pos_to_local(sector_slope_wall_a_x, sector_slope_wall_a_y, sin(a0), cos(a0), wall_b_x, wall_b_y)

            if abs(local_a_y) > max_y:
                max_y = local_a_y
                furthest_point_x, furthest_point_y = wall_a_x, wall_a_y

            if abs(local_b_y) > max_y:
                max_y = local_b_y
                furthest_point_x, furthest_point_y = wall_b_x, wall_b_y

    slope_end_x, slope_end_y = intersect_lin(a0, sector_slope_wall_a_x, sector_slope_wall_a_y, a1, furthest_point_x, furthest_point_y)

    return slope_end_x, slope_end_y

@njit(fastmath=True)
def intersect_lin(_a1:float, _v1_x:float, _v1_y:float, _a2:float, _v2_x:float, _v2_y:float)->tuple:
    m1 = tan(_a1)
    m2 = tan(_a2)
    
    b1 = _v1_y - m1 * _v1_x
    b2 = _v2_y - m2 * _v2_x
    
    x_intersect = (b2 - b1) / (m1 - m2) if not (m1 - m2) == 0 else 1

    y_intersect = m1 * x_intersect + b1
    
    return x_intersect, y_intersect

@njit(fastmath=True)
def find_y(x1, y1, x2, y2, x):

    m = (y2 - y1) / (x2 - x1)

    b = y1 - m * x1

    y = m * x + b
    
    return y

@njit(fastmath=True)
def process_wall_rendering(screen:array, textures_sheet:array, camera_fog_distance:int, wall_portal:int, wall_length:float, sector_height:float, wall_texture_id:int, sector_slope_floor_z:float, sector_slope_ceil_z:float, sector_z_floor:float, sector_z_ceil:float, sector_slope_floor_wall_a_x:float, sector_slope_floor_wall_a_y:float, sector_slope_floor_end_x:float, sector_slope_floor_end_y:float, sector_slope_ceil_wall_a_x:float, sector_slope_ceil_wall_a_y:float, sector_slope_ceil_end_x:float, sector_slope_ceil_end_y:float, x0:int, x1:int, y_hi:array, y_lo:array, engine_state:str, z_floor_total:float, camera_pos_z:float, camera_angle:float, light_factor:float, camera_pos_x:float, camera_pos_y:float, nyyfd:int, nyyf0:int, nyycd:int, nyyc0:int, txd:int, tx0:int, yfd:int, yf0:int, ycd:int, yc0:int, iz0:int, iz1:float, yyfd:int, yyf0:int, yycd:int, yyc0:int, sector_texture_floor_id:int, sector_texture_ceil_id:int, u0_z0:float, u1_z1:float, wall_texture_width:int, wall_texture_height:int, wall_texture_offset_x:float, wall_texture_offset_y:float, wall_texture_coordinate:int, wall_texture_up_width:int, wall_texture_up_height:int, wall_texture_up_offset_x:float, wall_texture_up_offset_y:float, wall_texture_up_x_coordinate:int, wall_texture_down_width:int, wall_texture_down_height:int, wall_texture_down_offset_x:float, wall_texture_down_offset_y:float, wall_texture_down_x_coordinate:int, skybox_texture_x_coordinate:int, skybox_texture_width:int, skybox_texture_height:int, camera_angle_sin:float, camera_angle_cos:float, sector_ceil_texture_x_coordinate:int, sector_ceil_texture_width:int, sector_ceil_texture_height:int, sector_floor_texture_x_coordinate:int, sector_floor_texture_width:int, sector_floor_texture_height:int, viewport_height:int, viewport_width:int, viewport_ratio_width_i:float, viewport_ratio_height:float, viewport_x:int, viewport_y:int, cp0_x:float, cp0_y:float, cp1_x:float, cp1_y:float, skyboxes_sheet:array, wall_texture_id_down:int, wall_texture_id_up:int)->tuple:

    point_floor_x = 0
    point_floor_y = 0
    point_floor_z = 0

    plane_normal_floor_x = 0
    plane_normal_floor_y = 0
    plane_normal_floor_z = 0

    point_ceil_x = 0
    point_ceil_y = 0
    point_ceil_z = 0

    plane_normal_ceil_x = 0
    plane_normal_ceil_y = 0
    plane_normal_ceil_z = 0

    if sector_slope_floor_z != 0:
        point_floor_x = sector_slope_floor_wall_a_x
        point_floor_y = sector_slope_floor_wall_a_y
        point_floor_z = sector_z_floor + sector_slope_floor_z

        plane_yaw_floor = get_angle(sector_slope_floor_wall_a_x, sector_slope_floor_wall_a_y, sector_slope_floor_end_x, sector_slope_floor_end_y) + PI_2
        plane_pitch_floor = -(atan(sector_slope_floor_z / distance(sector_slope_floor_wall_a_x, sector_slope_floor_wall_a_y, sector_slope_floor_end_x, sector_slope_floor_end_y)) - PI_2) if distance(sector_slope_floor_wall_a_x, sector_slope_floor_wall_a_y, sector_slope_floor_end_x, sector_slope_floor_end_y) != 0 else 0

        normal_floor_x = cos(plane_pitch_floor) * cos(plane_yaw_floor)
        normal_floor_y = cos(plane_pitch_floor) * sin(plane_yaw_floor)
        normal_floor_z = sin(plane_pitch_floor)

        plane_normal_floor_x, plane_normal_floor_y, plane_normal_floor_z = normalize3D(normal_floor_x, normal_floor_y, normal_floor_z)

    if sector_slope_ceil_z != 0:
        point_ceil_x = sector_slope_ceil_wall_a_x
        point_ceil_y = sector_slope_ceil_wall_a_y
        point_ceil_z = sector_z_ceil + sector_slope_ceil_z

        plane_yaw_ceil = get_angle(sector_slope_ceil_wall_a_x, sector_slope_ceil_wall_a_y, sector_slope_ceil_end_x, sector_slope_ceil_end_y) + PI_2
        plane_pitch_ceil = -(atan(sector_slope_ceil_z / distance(sector_slope_ceil_wall_a_x, sector_slope_ceil_wall_a_y, sector_slope_ceil_end_x, sector_slope_ceil_end_y)) - PI_2) if distance(sector_slope_ceil_wall_a_x, sector_slope_ceil_wall_a_y, sector_slope_ceil_end_x, sector_slope_ceil_end_y) != 0 else 0

        normal_ceil_x = cos(plane_pitch_ceil) * cos(plane_yaw_ceil)
        normal_ceil_y = cos(plane_pitch_ceil) * sin(plane_yaw_ceil)
        normal_ceil_z = sin(plane_pitch_ceil)

        plane_normal_ceil_x, plane_normal_ceil_y, plane_normal_ceil_z = normalize3D(normal_ceil_x, normal_ceil_y, normal_ceil_z)

    for x in range(x0, x1 + 1):

        if txd != 0:
            xp = relative_change(x, tx0, txd)
        else:
            xp = 0

        u = tiled_u_coordinate(wall_length, xp, u0_z0, u1_z1, iz0, iz1)

        tyf = int(linear_function(xp, yfd, yf0))
        tyc = int(linear_function(xp, ycd, yc0))

        tyyf = int(linear_function(xp, yyfd, yyf0))
        tyyc = int(linear_function(xp, yycd, yyc0))

        yyf = clamp(tyyf, y_lo[x], y_hi[x])
        yyc = clamp(tyyc, y_lo[x], y_hi[x])

        if yyf > y_lo[x]:
            if engine_state == "DEBUG":
                screen = debug_verline(
                    screen, x, y_lo[x], yyf,
                    color_mult(DEBUG_FLOOR_COLOR, light_factor), viewport_height, viewport_width, viewport_x, viewport_y
                )
            else:
                if sector_texture_floor_id == 0:
                    screen = sky_verline(screen, skyboxes_sheet, skybox_texture_x_coordinate, skybox_texture_width, skybox_texture_height, camera_angle, x, y_lo[x], yyf, viewport_height, viewport_width, viewport_x, viewport_y)
                elif sector_slope_floor_z != 0:
                    screen = render_sloped_visplane(screen, camera_pos_x, camera_pos_y, camera_pos_z, point_floor_x, point_floor_y, point_floor_z, plane_normal_floor_x, plane_normal_floor_y, plane_normal_floor_z, x, y_lo[x], yyf, textures_sheet, sector_floor_texture_width, sector_floor_texture_height, sector_floor_texture_x_coordinate, light_factor, camera_angle_sin, camera_angle_cos, camera_fog_distance, viewport_height, viewport_width, viewport_x, viewport_y)

        if yyc < y_hi[x]:
            if engine_state == "DEBUG":
                screen = debug_verline(
                    screen, x, yyc, y_hi[x],
                    color_mult(DEBUG_CEILING_COLOR, light_factor), viewport_height, viewport_width, viewport_x, viewport_y
                )
            else:
                if sector_texture_ceil_id == 0:
                    screen = sky_verline(screen, skyboxes_sheet, skybox_texture_x_coordinate, skybox_texture_width, skybox_texture_height, camera_angle, x, yyc, y_hi[x], viewport_height, viewport_width, viewport_x, viewport_y)
                elif sector_slope_ceil_z != 0:
                    screen = render_sloped_visplane(screen, camera_pos_x, camera_pos_y, camera_pos_z, point_ceil_x, point_ceil_y, point_ceil_z, plane_normal_ceil_x, plane_normal_ceil_y, plane_normal_ceil_z, x, yyc, y_hi[x], textures_sheet, sector_ceil_texture_width, sector_ceil_texture_height, sector_ceil_texture_x_coordinate, light_factor, camera_angle_sin, camera_angle_cos, camera_fog_distance, viewport_height, viewport_width, viewport_x, viewport_y)

        if wall_portal != 0:
            tnyyf = int(linear_function(xp, nyyfd, nyyf0))
            tnyyc = int(linear_function(xp, nyycd, nyyc0))

            nyyf = clamp(tnyyf, y_lo[x], y_hi[x])
            nyyc = clamp(tnyyc, y_lo[x], y_hi[x])

            if engine_state == "DEBUG":
                if tnyyc <= viewport_height:
                    screen = debug_verline(
                        screen, x, nyyc, yyc,
                        color_mult(DEBUG_WALLPORTALTOP_COLOR, light_factor), viewport_height, viewport_width, viewport_x, viewport_y
                    )
                if tnyyf >= 0:
                    screen = debug_verline(
                        screen, x, yyf, nyyf,
                        color_mult(DEBUG_WALLPORTALBOTTOM_COLOR, light_factor), viewport_height, viewport_width, viewport_x, viewport_y
                    )
            else:
                if tnyyc <= viewport_height:
                    if wall_texture_id_up != 0:
                        screen = texture_verline(screen, textures_sheet, wall_texture_up_width, wall_texture_up_height, wall_texture_up_offset_x, wall_texture_up_offset_y, wall_texture_up_x_coordinate, x, nyyc, yyc, u, tyf, tyc, light_factor,z_floor_total, sector_height, camera_pos_z, tyyf, tyyc, camera_pos_x, camera_pos_y, camera_fog_distance, viewport_height, viewport_width, viewport_ratio_width_i, viewport_ratio_height, viewport_x, viewport_y)
                    else:
                        screen = sky_verline(screen, skyboxes_sheet, skybox_texture_x_coordinate, skybox_texture_width, skybox_texture_height, camera_angle, x, nyyc, yyc, viewport_height, viewport_width, viewport_x, viewport_y)

                if tnyyf >= 0:
                    if wall_texture_id_down != 0:
                        screen = texture_verline(screen, textures_sheet, wall_texture_down_width, wall_texture_down_height, wall_texture_down_offset_x, wall_texture_down_offset_y, wall_texture_down_x_coordinate, x, yyf, nyyf, u, tyf, tyc, light_factor,z_floor_total, sector_height, camera_pos_z, tyyf, tyyc, camera_pos_x, camera_pos_y, camera_fog_distance, viewport_height, viewport_width, viewport_ratio_width_i, viewport_ratio_height, viewport_x, viewport_y)
                    else:
                        screen = sky_verline(screen, skyboxes_sheet, skybox_texture_x_coordinate, skybox_texture_width, skybox_texture_height, camera_angle, x, yyf, nyyf, viewport_height, viewport_width, viewport_x, viewport_y)

            y_hi[x] = clamp(min(min(yyc, nyyc), y_hi[x]), 0, viewport_height - 1)
            y_lo[x] = clamp(max(max(yyf, nyyf), y_lo[x]), 0, viewport_height - 1)
        else:
            if engine_state == "DEBUG":
                screen = debug_verline(
                    screen, x, yyf, yyc,
                    color_mult(DEBUG_WALL_COLOR, light_factor), viewport_height, viewport_width, viewport_x, viewport_y
                )
            else:
                if wall_texture_id != 0:
                    screen = texture_verline(screen, textures_sheet, wall_texture_width, wall_texture_height, wall_texture_offset_x, wall_texture_offset_y, wall_texture_coordinate, x, yyf, yyc, u, tyf, tyc, light_factor, z_floor_total, sector_height, camera_pos_z, tyyf, tyyc, camera_pos_x, camera_pos_y, camera_fog_distance, viewport_height, viewport_width, viewport_ratio_width_i, viewport_ratio_height, viewport_x, viewport_y)
                else:
                    screen = sky_verline(screen, skyboxes_sheet, skybox_texture_x_coordinate, skybox_texture_width, skybox_texture_height, camera_angle, x, yyf, yyc, viewport_height, viewport_width, viewport_x, viewport_y)

    return y_lo, y_hi, screen

@njit(fastmath=True)
def get_portal_camera_transforms(camera_pos_x:float, camera_pos_y:float, camera_angle:float, wall_a_x:float, wall_a_y:float, wall_b_x:float, wall_b_y:float, portal_wall_a_x:float, portal_wall_a_y:float, portal_wall_b_x:float, portal_wall_b_y:float)->tuple:
    a1 = get_angle(wall_a_x, wall_a_y, wall_b_x, wall_b_y)
    a2 = get_angle(portal_wall_b_x, portal_wall_b_y, portal_wall_a_x, portal_wall_a_y)

    a = a2 - a1

    l1 = distance(wall_a_x, wall_a_y, camera_pos_x, camera_pos_y)
    
    a0 = a2 + (get_angle(wall_a_x, wall_a_y, camera_pos_x, camera_pos_y) - a1 + a + (PI_2 * cos(a)))

    portal_camera_pos_x, portal_camera_pos_y = extend_v(portal_wall_b_x, portal_wall_b_y, l1, a0)

    portal_camera_angle = normalize_angle(camera_angle+a)

    portal_camera_angle_sin = sin(portal_camera_angle)
    portal_camera_angle_cos = cos(portal_camera_angle)

    return portal_camera_pos_x, portal_camera_pos_y, portal_camera_angle, portal_camera_angle_sin, portal_camera_angle_cos

@njit(fastmath=True)
def extend_v(_v_x:float, _v_y:float, _s:float, _a:float)->tuple:
    x, y = _v_x, _v_y
    x1 = x + _s * cos(_a)
    y1 = y + _s * sin(_a)

    return x1, y1

@njit(fastmath=True)
def extension_factor(_v1_x:float, _v1_y:float, _v2_x:float, _v2_y:float, _v3_x:float, _v3_y:float, _a:float)->float:
    v1v2_distance = distance(_v1_x, _v1_y, _v2_x, _v2_y)
    
    v1v3_distance = distance(_v1_x, _v1_y, _v3_x, _v3_y)

    ext = _a
    
    if v1v2_distance != 0:
        ext = v1v3_distance / v1v2_distance
    
    return ext

@njit(fastmath=True)
def distance(_v1_x:float, _v1_y:float, _v2_x:float, _v2_y:float)->float:
    return sqrt((_v1_x - _v2_x)**2 + (_v1_y - _v2_y)**2) 

@njit(fastmath=True)
def clamp(_x, _mi, _ma):
    return max(_mi, min(_x, _ma))                       

@njit(fastmath=True)
def screen_angle_to_x(_a:float, viewport_width_2:int)->int:
    return int((viewport_width_2) * (1 - tan(((_a + (HFOV_2)) / HFOV) * PI_2 - PI_4)))

@njit
def near_clip(cp0_x:float, cp0_y:float, cp1_x:float, cp1_y:float, ap0:float, ap1:float, znl_x:float, znl_y:float, zfl_x:float, zfl_y:float, znr_x:float, znr_y:float, zfr_x:float, zfr_y:float):
    if (cp0_y < ZNEAR or cp1_y < ZNEAR or ap0 > +(HFOV_2) or ap1 < -(HFOV_2)):

        il_x, il_y = intersect_segs(cp0_x, cp0_y, cp1_x, cp1_y, znl_x, znl_y, zfl_x, zfl_y)
        ir_x, ir_y = intersect_segs(cp0_x, cp0_y, cp1_x, cp1_y, znr_x, znr_y, zfr_x, zfr_y)

        if (not isnan(il_x)):
            cp0_x, cp0_y = il_x, il_y
            ap0 = get_angle(0, 0, cp0_x, cp0_y)

        if (not isnan(ir_x)):
            cp1_x, cp1_y = ir_x, ir_y
            ap1 = get_angle(0, 0, cp1_x, cp1_y)

    return cp0_x, cp0_y, cp1_x, cp1_y, ap0, ap1

@njit(fastmath=True)
def intersect_segs(_a0_x, _a0_y, _a1_x, _a1_y, _b0_x, _b0_y, _b1_x, _b1_y):
    d = ((_a0_x - _a1_x) * (_b0_y - _b1_y) - ((_a0_y - _a1_y) * (_b0_x - _b1_x)))

    if (fabs(d) < 0.00001):
        return nan, nan
    
    t = (((_a0_x - _b0_x) * (_b0_y - _b1_y)) - ((_a0_y - _b0_y) * (_b0_x - _b1_x))) / d if d != 0 else 0
    u = (((_a0_x - _b0_x) * (_a0_y - _a1_y)) - ((_a0_y - _b0_y) * (_a0_x - _a1_x))) / d if d != 0 else 0

    if t >= 0 and t <= 1 and u >= 0 and u <= 1:
        return (_a0_x + (t * (_a1_x - _a0_x)), _a0_y + (t * (_a1_y - _a0_y)))
    else:
        return nan, nan
    
@njit(fastmath=True)
def get_angle(_v1_x:float, _v1_y:float, _v2_x:float, _v2_y:float)->float:
    return normalize_angle(atan2(_v2_y - _v1_y, _v2_x - _v1_x) - PI_2)

@njit(fastmath=True)
def normalize_angle(_a:float)->float:
    return _a - (TAU * floor((_a + PI) / TAU))

@njit(fastmath=True)
def world_pos_to_local(_p_x:float, _p_y:float, _a0sin:float, _a0cos:float, _v_x:float, _v_y:float)->tuple:
    u_x, u_y = _v_x - _p_x, _v_y - _p_y

    return u_x * _a0sin - u_y * _a0cos, u_x * _a0cos + u_y * _a0sin

@njit(fastmath=True)
def point_side(_p_x:float, _p_y:float, _a_x:float, _a_y:float, _b_x:float, _b_y:float)->float:                     
    return -(((_p_x - _a_x) * (_b_y - _a_y)) - ((_p_y - _a_y) * (_b_x - _a_x)))

@njit(fastmath=False)
def get_frame(ticks:float, animation_ms:float, animation_frames:int)->int:
    return int((ticks / animation_ms) % animation_frames) if animation_ms != 0 else 1

@njit(fastmath=True)
def create_animation(animation_id:int, animations_frames:array, animations_ms:array)->tuple:
    animation_frames = animations_frames[animation_id]
    animation_ms = animations_ms[animation_id]

    return animation_frames, animation_ms

@njit(fastmath=True)
def total_z(_z:float, _s:float, _b:bool)->float:
    if _b:
        if _z + _s < _z:
            _z += + _s

    else:
        if _z + _s > _z:
            _z += _s

    return _z

@njit(fastmath=True)
def get_wall(wall_id:int, walls_a_id:array, walls_b_id:array, walls_portal:array, walls_portal_wall_id:array, walls_sector_id:array, walls_texture_id:array, walls_texture_id_up:array, walls_texture_id_down:array, walls_animation_id:array, walls_texture_offset_x:array, walls_texture_offset_y:array, walls_texture_offset_up_x:array, walls_texture_offset_up_y:array, walls_texture_offset_down_x:array, walls_texture_offset_down_y:array, vertices_id:array, vertices_x:array, vertices_y:array)->tuple:
    wall_a_id = walls_a_id[wall_id]
    wall_a_x = vertices_x[wall_a_id]
    wall_a_y = vertices_y[wall_a_id]
    wall_b_id = walls_b_id[wall_id]
    wall_b_x = vertices_x[wall_b_id]
    wall_b_y = vertices_y[wall_b_id]
    wall_portal = walls_portal[wall_id]
    wall_portal_wall_id = walls_portal_wall_id[wall_id]
    wall_sector_id = walls_sector_id[wall_id]
    wall_texture_id = walls_texture_id[wall_id]
    wall_texture_id_up = walls_texture_id_up[wall_id]
    wall_texture_id_down = walls_texture_id_down[wall_id]
    wall_animation_id = walls_animation_id[wall_id]
    wall_texture_offset_x = walls_texture_offset_x[wall_id]
    wall_texture_offset_y = walls_texture_offset_y[wall_id]
    wall_texture_offset_up_x = walls_texture_offset_up_x[wall_id]
    wall_texture_offset_up_y = walls_texture_offset_up_y[wall_id]
    wall_texture_offset_down_x = walls_texture_offset_down_x[wall_id]
    wall_texture_offset_down_y = walls_texture_offset_down_y[wall_id]

    return wall_a_x, wall_a_y, wall_a_id, wall_b_x, wall_b_y, wall_b_id, wall_portal, wall_portal_wall_id, wall_sector_id, wall_texture_id, wall_texture_id_up, wall_texture_id_down, wall_animation_id, wall_texture_offset_x, wall_texture_offset_y, wall_texture_offset_up_x, wall_texture_offset_up_y, wall_texture_offset_down_x, wall_texture_offset_down_y

@njit(fastmath=True)
def create_sector_queue()->tuple:
    sector_queue_id = full(QUEUE_MAX, -1, dtype=int32)
    sector_queue_wall_id = full(QUEUE_MAX, -1, int32)
    sector_queue_x0 = full(QUEUE_MAX, -1, int32)
    sector_queue_x1 = full(QUEUE_MAX, -1, int32)
    sector_queue_camera_pos_x = full(QUEUE_MAX, 0, float32)
    sector_queue_camera_pos_y = full(QUEUE_MAX, 0, float32)
    sector_queue_camera_angle = full(QUEUE_MAX, 0, float32)
    sector_queue_camera_angle_sin = full(QUEUE_MAX, 0, float32)
    sector_queue_camera_angle_cos = full(QUEUE_MAX, 0, float32)
    sector_queue_n = 0

    return sector_queue_n, sector_queue_id, sector_queue_wall_id, sector_queue_x0, sector_queue_x1, sector_queue_camera_pos_x, sector_queue_camera_pos_y, sector_queue_camera_angle, sector_queue_camera_angle_sin, sector_queue_camera_angle_cos

@njit(fastmath=True)
def view_frustrum()->tuple:
    zdl_x, zdl_y = rotate(0, 1, +(HFOV_2))
    zdr_x, zdr_y = rotate(0, 1, -(HFOV_2))
    znl_x, znl_y = zdl_x * ZNEAR, zdl_y * ZNEAR
    znr_x, znr_y = zdr_x * ZNEAR, zdr_y * ZNEAR
    zfl_x, zfl_y = zdl_x * ZFAR, zdl_y * ZFAR
    zfr_x, zfr_y = zdr_x * ZFAR, zdr_y * ZFAR

    return znl_x, znl_y, znr_x, znr_y, zfl_x, zfl_y, zfr_x, zfr_y
    
@njit(fastmath=True)
def rotate(_v_x, _v_y, _a)->tuple:                               
    rotated_x = (_v_x) * cos(_a) - (_v_y) * sin(_a)
    rotated_y = (_v_x) * sin(_a) + (_v_y) * cos(_a)
        
    return rotated_x, rotated_y

@njit(fastmath=True)
def texture(screen:array, textures_sheet:array, texture_width:int, texture_height:int, texture_x_coordinate:int, pos_x:int, pos_y:int, window_pos_x:int, window_pos_y:int, window_width:int, window_height:int, scale=1.0):
    scaled_texture_width = int(texture_width*scale)
    scaled_texture_height = int(texture_height*scale)
    scale_i = 1 / scale

    for y in range(scaled_texture_height):
        yy = int(y * scale_i)
        yyy = int(window_pos_y + pos_y + y)

        for x in range(scaled_texture_width):
            xx = int(texture_x_coordinate + x * scale_i)
            xxx = int(window_pos_x + pos_x + x)

            if 0 <= x <= window_width and 0 <= y <= window_height:
                color = textures_sheet[xx][yy]

                if check_keyColor(color):
                    screen[xxx][yyy] = color
    
    return screen