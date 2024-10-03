from numpy import full, int32, float32, uint8, string_, zeros, copy
from renderer import render_viewport, view_frustrum, clamp
from input import input, input_assignment
from math import nan
import pygame as pg
from editor import render_editor
from project_loader import load_project
from settings import WINDOW_WIDTH, WINDOW_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT, SCALED_SCREEN_WIDTH, SCALED_SCREEN_HEIGHT, ACTOR_MAX, SECTOR_MAX, WALL_MAX, TEXTURE_MAX, ENVIRONMENT_ANIMATION_MAX, BILLBOARD_MAX, SKYBOX_MAX
from keys import R

class Game:
    def __init__(self) -> None:

        self.window = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.screen = zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=uint8)

        self.znl_x, self.znl_y, self.znr_x, self.znr_y, self.zfl_x, self.zfl_y, self.zfr_x, self.zfr_y = view_frustrum()

        self.camera_sector = 1
        self.camera_pos_x = 2.5
        self.camera_pos_y = 5
        self.camera_pos_z = 0.6
        self.camera_angle = 0
        self.camera_angle_sin = 0
        self.camera_angle_cos = 0
        self.camera_fog_distance = 1/100
        self.camera_rotation_speed = 0.0016
        self.camera_move_speed = 0.016
        
        self.sectors_id = full(SECTOR_MAX, 0, dtype=int32)
        self.sectors_light_factor = full(SECTOR_MAX, 0, dtype=float32)
        self.sectors_z_floor = full(SECTOR_MAX, 0, dtype=float32)
        self.sectors_z_ceil = full(SECTOR_MAX, 0, dtype=float32)
        self.sectors_ceil_texture_id = full(SECTOR_MAX, 0, dtype=int32)
        self.sectors_ceil_animation_id = full(SECTOR_MAX, 0, dtype=int32)
        self.sectors_floor_texture_id = full(SECTOR_MAX, 0, dtype=int32)
        self.sectors_floor_animation_id = full(SECTOR_MAX, 0, dtype=int32)
        self.sectors_slope_floor_z = full(SECTOR_MAX, 0, dtype=float32)
        self.sectors_slope_floor_wall_id = full(SECTOR_MAX, 0, dtype=int32)
        self.sectors_slope_ceil_z = full(SECTOR_MAX, 0, dtype=float32)
        self.sectors_slope_ceil_wall_id = full(SECTOR_MAX, 0, dtype=int32)
        self.sectors_slope_floor_friction = full(SECTOR_MAX, 0, dtype=float32)
        self.sectors_slope_floor_end_x = full(SECTOR_MAX, nan, dtype=float32)
        self.sectors_slope_floor_end_y = full(SECTOR_MAX, nan, dtype=float32)
        self.sectors_slope_ceil_end_x = full(SECTOR_MAX, nan, dtype=float32)
        self.sectors_slope_ceil_end_y = full(SECTOR_MAX, nan, dtype=float32)
        self.sectors_skybox_id = full(SECTOR_MAX, 0, dtype=int32)
        self.sectors_walls = full((SECTOR_MAX, WALL_MAX), 0, dtype=int32)
        self.sectors_n = 0

        self.vertices_id = full(WALL_MAX*2, 0, dtype=int32)
        self.vertices_x = full(WALL_MAX*2, 0, dtype=float32)
        self.vertices_y = full(WALL_MAX*2, 0, dtype=float32)
        self.vertices_sector = full(WALL_MAX*2, 0, dtype=int32)
        self.vertices_n = 0

        self.selected_vertices_id = full(WALL_MAX*2, 0, dtype=int32)
        self.selected_vertices_id_n = 0

        self.new_vertices_id = full(WALL_MAX*2, 0, dtype=int32)
        self.new_vertices_x = full(WALL_MAX*2, 0, dtype=float32)
        self.new_vertices_y = full(WALL_MAX*2, 0, dtype=float32)
        self.new_vertices_type = full(WALL_MAX*2, 0, dtype=int32)
        self.new_vertices_n = 0

        self.new_walls_id = full(WALL_MAX, 0, dtype=int32)
        self.new_walls_a_id = full(WALL_MAX, 0, dtype=int32)
        self.new_walls_b_id = full(WALL_MAX, 0, dtype=int32)
        self.new_walls_n = 0

        self.walls_id = full(WALL_MAX, 0, dtype=int32)
        self.walls_a_id = full(WALL_MAX, 0, dtype=int32)
        self.walls_b_id = full(WALL_MAX, 0, dtype=int32)
        self.walls_portal = full(WALL_MAX, 0, dtype=int32)
        self.walls_portal_wall_id = full(WALL_MAX, 0, dtype=int32)
        self.walls_sector_id = full(WALL_MAX, 0, dtype=int32)
        self.walls_texture_id = full(WALL_MAX, 0, dtype=int32)
        self.walls_texture_id_up = full(WALL_MAX, 0, dtype=int32)
        self.walls_texture_id_down = full(WALL_MAX, 0, dtype=int32)
        self.walls_animation_id = full(WALL_MAX, 0, dtype=int32)
        self.walls_texture_offset_x = full(WALL_MAX, 0, dtype=float32)
        self.walls_texture_offset_y = full(WALL_MAX, 0, dtype=float32)
        self.walls_texture_offset_up_x = full(WALL_MAX, 0, dtype=float32)
        self.walls_texture_offset_up_y = full(WALL_MAX, 0, dtype=float32)
        self.walls_texture_offset_down_x = full(WALL_MAX, 0, dtype=float32)
        self.walls_texture_offset_down_y = full(WALL_MAX, 0, dtype=float32)
        self.walls_n = 0

        self.billboards_id = full(BILLBOARD_MAX, 0, dtype=int32)
        self.billboards_sector_id = full(BILLBOARD_MAX, 0, dtype=int32)
        self.billboards_sprite_id = full(BILLBOARD_MAX, 0, dtype=int32)
        self.billboards_position_x = full(BILLBOARD_MAX, 0, dtype=float32)
        self.billboards_position_y = full(BILLBOARD_MAX, 0, dtype=float32)
        self.billboards_position_z = full(BILLBOARD_MAX, 0, dtype=float32)

        self.textures_sheet = zeros((1, 1, 1), dtype=uint8)
        self.textures_width = full(TEXTURE_MAX, 0, dtype=int32)
        self.textures_height = full(TEXTURE_MAX, 0, dtype=int32)
        self.textures_x_coordinates = full(TEXTURE_MAX, 0, dtype=int32)
        self.texture_paths = full(TEXTURE_MAX, 0, dtype=string_)
        self.environmental_animations_frames = full(ENVIRONMENT_ANIMATION_MAX, 0, dtype=string_)
        self.environmental_animations_ms = full(ENVIRONMENT_ANIMATION_MAX, 0, dtype=float32)
        self.environmental_animations_frames_count = full(ENVIRONMENT_ANIMATION_MAX, 0, dtype=int32)
        self.textures_n = 0

        self.skyboxes_sheet = zeros((1, 1, 1), dtype=uint8)
        self.skyboxes_width = full(SKYBOX_MAX, 0, dtype=int32)
        self.skyboxes_height = full(SKYBOX_MAX, 0, dtype=int32)
        self.skyboxes_x_coordinates = full(SKYBOX_MAX, 0, dtype=int32)
        self.skyboxes_paths = full(SKYBOX_MAX, 0, dtype=string_)
        self.skyboxes_n = 0

        self.sprites_sheet = zeros((1, 1, 1), dtype=uint8)
        self.sprites_width = full(TEXTURE_MAX, 0, dtype=int32)
        self.sprites_height = full(TEXTURE_MAX, 0, dtype=int32)
        self.sprites_x_coordinates = full(TEXTURE_MAX, 0, dtype=int32)
        self.sprites_paths = full(TEXTURE_MAX, 0, dtype=string_)
        self.sprites_n = 0

        self.link_wall_start_id = 0
        self.link_wall_end_id = 0

        self.ticks = 0
        self.state = "RUN"
        self.engine_state = "NONE"
        self.editor_mode = "NONE"
        self.camera_clip = True
        self.axis_lock = 0

        self.viewport_width, self.viewport_height = SCREEN_WIDTH*(10/24), SCREEN_HEIGHT*(10/16)
        self.viewport_x , self.viewport_y = SCREEN_WIDTH*(2/24), SCREEN_HEIGHT*(2/16)

        self.editor_width, self.editor_height = SCREEN_WIDTH*(10/24), SCREEN_HEIGHT*(10/16)
        self.editor_x, self.editor_y = SCREEN_WIDTH*(2/24)+SCREEN_WIDTH*(10/24), SCREEN_HEIGHT*(2/16)
        self.editor_scale = 20
        self.editor_origin_x, self.editor_origin_y = self.editor_width // 2, self.editor_height // 2
        self.editor_offset_x, self.editor_offset_y = 0, 0

        self.viewport_width, self.viewport_height = int(self.viewport_width), int(self.viewport_height)
        self.viewport_x, self.viewport_y = int(self.viewport_x), int(self.viewport_y)
        self.editor_width, self.editor_height = int(self.editor_width), int(self.editor_height)
        self.editor_x, self.editor_y = int(self.editor_x), int(self.editor_y)

        self.mouse_r_pressed = False
        self.prev_mouse_r_pressed = False
        self.mouse_l_pressed = False
        self.prev_mouse_l_pressed = False
        self.mouse_m_pressed = False
        self.prev_mouse_m_pressed = False
        self.mouse_wheel_state = "NONE"
        self.mouse_position_x, self.mouse_position_y = 0, 0
        self.mouse_delta_x, self.mouse_delta_y = 0, 0
        self.drawing_angle = 0
        self.vertex_hovered_id = 0
        self.wall_hovered_id = 0

        self.active_window = "NONE"
        self.select_mode = "VERTEX"
        self.texture_select_mode = "WALL"
        self.texture_slot_id = 0

        self.possessed_actor_id = 1
        self.actors_id = full(ACTOR_MAX, 0, dtype=int32)
        self.actors_pos_x = full(ACTOR_MAX, 0, dtype=float32)
        self.actors_pos_y = full(ACTOR_MAX, 0, dtype=float32)
        self.actors_pos_z = full(ACTOR_MAX, 0, dtype=float32)
        self.actors_billboard_id = full(ACTOR_MAX, 0, dtype=int32)

        self.project_path = "project.fizz"

        self.clock = pg.time.Clock()
        self.ticks = pg.time.get_ticks()
        self.fps = 0
        self.timestep = 1/60
        self.keys = zeros(128, dtype=int32)
        self.prev_keys = zeros(128, dtype=int32)

    def main(self) -> None:
        self.textures_sheet, self.sprites_sheet, self.sectors_id, self.sectors_light_factor, self.sectors_z_floor, self.sectors_z_ceil, self.sectors_ceil_texture_id, self.sectors_ceil_animation_id, self.sectors_floor_texture_id, self.sectors_floor_animation_id, self.sectors_slope_floor_z, self.sectors_slope_floor_wall_id, self.sectors_slope_ceil_z, self.sectors_slope_ceil_wall_id, self.sectors_slope_floor_friction, self.sectors_n, self.walls_id, self.walls_a_id, self.walls_b_id, self.walls_portal, self.walls_portal_wall_id, self.walls_sector_id, self.walls_animation_id, self.walls_texture_id, self.walls_texture_id_up, self.walls_texture_id_down, self.walls_texture_offset_x, self.walls_texture_offset_y, self.walls_texture_offset_up_x, self.walls_texture_offset_up_y, self.walls_texture_offset_down_x, self.walls_texture_offset_down_y, self.walls_n, self.textures_sheet_width, self.textures_sheet_height, self.textures_width, self.textures_height, self.textures_x_coordinates, self.textures_path, self.sprites_sheet_width, self.sprites_sheet_height, self.sprites_width, self.sprites_height, self.sprites_x_coordinate, self.sprites_path, self.environmental_animations_frames, self.environmental_animations_ms, self.environmental_animations_frames_count, self.vertices_id, self.vertices_x, self.vertices_y, self.vertices_sector, self.vertices_n, self.textures_n, self.sprites_n, self.sectors_skybox_id, self.skyboxes_sheet, self.skyboxes_height, self.skyboxes_width, self.skyboxes_x_coordinates, self.skyboxes_paths, self.skyboxes_n = load_project(self.project_path)

        while self.state == "RUN":
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.state = "EXIT"
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        self.mouse_wheel_state = "UP"
                    elif event.button == 5:
                        self.mouse_wheel_state = "DOWN"
                    else:
                        self.mouse_wheel_state = "NONE"

                    if event.button == 3:
                        self.mouse_r_pressed = True
                    elif event.button == 2:
                        self.mouse_m_pressed = True
                    elif event.button == 1:
                        self.mouse_l_pressed = True

                elif event.type == pg.MOUSEBUTTONUP:
                    if event.button == 3:
                        self.mouse_r_pressed = False
                    elif event.button == 2:
                        self.mouse_m_pressed = False
                    elif event.button == 1:
                        self.mouse_l_pressed = False
                
                elif event.type == pg.DROPFILE:
                    print(event.file)
            
            self.fps = round(self.clock.get_fps())
            self.ticks = pg.time.get_ticks()

            print(self.fps)

            self.keys = input_assignment(self.keys)

            if self.keys[R] == 1 and self.prev_keys[R] == 0:
                self.textures_sheet, self.sprites_sheet, self.sectors_id, self.sectors_light_factor, self.sectors_z_floor, self.sectors_z_ceil, self.sectors_ceil_texture_id, self.sectors_ceil_animation_id, self.sectors_floor_texture_id, self.sectors_floor_animation_id, self.sectors_slope_floor_z, self.sectors_slope_floor_wall_id, self.sectors_slope_ceil_z, self.sectors_slope_ceil_wall_id, self.sectors_slope_floor_friction, self.sectors_n, self.walls_id, self.walls_a_id, self.walls_b_id, self.walls_portal, self.walls_portal_wall_id, self.walls_sector_id, self.walls_animation_id, self.walls_texture_id, self.walls_texture_id_up, self.walls_texture_id_down, self.walls_texture_offset_x, self.walls_texture_offset_y, self.walls_texture_offset_up_x, self.walls_texture_offset_up_y, self.walls_texture_offset_down_x, self.walls_texture_offset_down_y, self.walls_n, self.textures_sheet_width, self.textures_sheet_height, self.textures_width, self.textures_height, self.textures_x_coordinates, self.textures_path, self.sprites_sheet_width, self.sprites_sheet_height, self.sprites_width, self.sprites_height, self.sprites_x_coordinate, self.sprites_path, self.environmental_animations_frames, self.environmental_animations_ms, self.environmental_animations_frames_count, self.vertices_id, self.vertices_x, self.vertices_y, self.vertices_sector, self.vertices_n, self.textures_n, self.sprites_n, self.sectors_skybox_id, self.skyboxes_sheet, self.skyboxes_height, self.skyboxes_width, self.skyboxes_x_coordinates, self.skyboxes_paths, self.skyboxes_n = load_project(self.project_path)

            self.screen, self.sectors_slope_floor_end_x, self.sectors_slope_floor_end_y, self.sectors_slope_ceil_end_x, self.sectors_slope_ceil_end_y, self.sectors_walls, self.walls_texture_id, self.texture_slot_id = render_viewport(
                self.screen, self.viewport_width, self.viewport_height, self.viewport_x, self.viewport_y, self.znl_x, self.znl_y, self.znr_x, self.znr_y, self.zfl_x, self.zfl_y, self.zfr_x, self.zfr_y, self.camera_sector, self.camera_pos_x, self.camera_pos_y, self.camera_pos_z, self.camera_angle, self.camera_fog_distance, self.camera_angle_sin, self.camera_angle_cos, self.sectors_id, self.sectors_light_factor, self.sectors_z_floor, self.sectors_z_ceil, self.sectors_ceil_texture_id, self.sectors_ceil_animation_id, self.sectors_floor_texture_id, self.sectors_floor_animation_id, self.sectors_slope_floor_z, self.sectors_slope_floor_wall_id, self.sectors_slope_ceil_z, self.sectors_slope_ceil_wall_id, self.sectors_slope_floor_friction, self.walls_id, self.walls_a_id, self.walls_b_id, self.walls_portal, self.walls_portal_wall_id, self.walls_sector_id, self.walls_texture_id, self.walls_texture_id_up, self.walls_texture_id_down, self.walls_animation_id, self.walls_texture_offset_x, self.walls_texture_offset_y, self.walls_texture_offset_up_x, self.walls_texture_offset_up_y, self.walls_texture_offset_down_x, self.walls_texture_offset_down_y, self.textures_sheet, self.textures_width, self.textures_height, self.environmental_animations_frames, self.environmental_animations_ms, self.ticks, self.textures_x_coordinates, self.engine_state, self.sectors_slope_floor_end_x, self.sectors_slope_floor_end_y, self.sectors_slope_ceil_end_x, self.sectors_slope_ceil_end_y, self.environmental_animations_frames_count, self.billboards_id, self.billboards_sector_id, self.billboards_sprite_id, self.billboards_position_x, self.billboards_position_y, self.billboards_position_z, self.sprites_sheet, self.sprites_x_coordinate, self.sprites_width, self.sprites_height, self.vertices_id, self.vertices_x, self.vertices_y, self.sectors_walls, self.keys, self.prev_keys, self.textures_n, self.skyboxes_sheet, self.skyboxes_width, self.skyboxes_height, self.skyboxes_x_coordinates, self.sectors_skybox_id, self.texture_select_mode, self.texture_slot_id, self.skyboxes_n
            )

            self.screen, self.new_vertices_id, self.new_vertices_x, self.new_vertices_y, self.new_vertices_n, self.vertices_id, self.vertices_x, self.vertices_y, self.vertices_sector, self.vertices_n, self.sectors_id, self.sectors_light_factor, self.sectors_z_floor, self.sectors_z_ceil, self.sectors_ceil_texture_id, self.sectors_ceil_animation_id, self.sectors_floor_texture_id, self.sectors_floor_animation_id, self.sectors_slope_floor_z, self.sectors_slope_floor_wall_id, self.sectors_slope_ceil_z, self.sectors_slope_ceil_wall_id, self.sectors_slope_floor_friction, self.sectors_n, self.new_walls_id, self.new_walls_a_id, self.new_walls_b_id, self.new_walls_n, self.walls_n, self.editor_mode, self.link_wall_start_id, self.link_wall_end_id, self.selected_vertices_id, self.selected_vertices_id_n, self.vertex_hovered_id, self.wall_hovered_id = render_editor(
                self.screen, self.editor_width, self.editor_height, self.editor_x, self.editor_y, self.editor_scale, self.editor_origin_x, self.editor_origin_y, self.editor_offset_x, self.editor_offset_y, self.walls_id, self.walls_a_id, self.walls_b_id, self.walls_portal, self.walls_portal_wall_id, self.walls_sector_id, self.walls_texture_id, self.walls_animation_id, self.walls_n, self.camera_pos_x, self.camera_pos_y, self.camera_angle, self.mouse_position_x, self.mouse_position_y, self.vertices_id, self.vertices_x, self.vertices_y, self.vertices_sector, self.vertices_n, self.active_window, self.mouse_l_pressed, self.prev_mouse_l_pressed, self.new_vertices_id, self.new_vertices_x, self.new_vertices_y, self.new_vertices_n, self.keys, self.sectors_id, self.sectors_light_factor, self.sectors_z_floor, self.sectors_z_ceil, self.sectors_ceil_texture_id, self.sectors_ceil_animation_id, self.sectors_floor_texture_id, self.sectors_floor_animation_id, self.sectors_slope_floor_z, self.sectors_slope_floor_wall_id, self.sectors_slope_ceil_z, self.sectors_slope_ceil_wall_id, self.sectors_slope_floor_friction, self.sectors_n, self.new_walls_id, self.new_walls_a_id, self.new_walls_b_id, self.new_walls_n, self.editor_mode, self.mouse_r_pressed, self.prev_mouse_r_pressed, self.link_wall_start_id, self.link_wall_end_id, self.selected_vertices_id, self.selected_vertices_id_n, self.vertex_hovered_id, self.wall_hovered_id, self.select_mode, self.sectors_skybox_id, self.axis_lock
            )

            self.mouse_delta_x, self.mouse_delta_y = pg.mouse.get_rel()
            self.mouse_position_x, self.mouse_position_y = clamp(pg.mouse.get_pos()[0], 0, SCALED_SCREEN_WIDTH), clamp(pg.mouse.get_pos()[1], 0, SCALED_SCREEN_HEIGHT)

            self.camera_angle, self.camera_angle_sin, self.camera_angle_cos, self.camera_pos_x, self.camera_pos_y, self.camera_pos_z, self.camera_sector, self.editor_scale, self.editor_offset_x, self.editor_offset_y, self.active_window, self.editor_mode, self.camera_clip, self.actors_id, self.actors_pos_x, self.actors_pos_y, self.actors_pos_z, self.actors_billboard_id, self.possessed_actor_id, self.sectors_slope_floor_z, self.sectors_slope_ceil_z, self.sectors_z_floor, self.sectors_z_ceil, self.texture_select_mode, self.engine_state, self.texture_slot_id, self.axis_lock = input(
                self.keys, self.prev_keys, self.mouse_delta_x, self.mouse_delta_y, self.mouse_position_x, self.mouse_position_y, self.camera_angle, self.camera_angle_sin, self.camera_angle_cos, self.camera_rotation_speed, self.camera_pos_x, self.camera_pos_y, self.camera_pos_z, self.camera_move_speed, self.camera_sector, self.editor_scale, self.editor_offset_x, self.editor_offset_y, self.viewport_width, self.viewport_height, self.viewport_x, self.viewport_y, self.editor_width, self.editor_height, self.editor_x, self.editor_y, self.mouse_r_pressed, self.mouse_l_pressed, self.mouse_m_pressed, self.active_window, self.editor_mode, self.camera_clip, self.actors_id, self.actors_pos_x, self.actors_pos_y, self.actors_pos_z, self.actors_billboard_id, self.possessed_actor_id, self.sectors_walls, self.vertices_x, self.vertices_y, self.walls_a_id, self.walls_b_id, self.walls_portal, self.walls_portal_wall_id, self.sectors_z_floor, self.sectors_z_ceil, self.sectors_slope_floor_z, self.sectors_slope_ceil_z, self.selected_vertices_id_n, self.sectors_slope_floor_end_x, self.sectors_slope_floor_end_y, self.sectors_slope_ceil_end_x, self.sectors_slope_ceil_end_y, self.sectors_slope_floor_wall_id, self.sectors_slope_ceil_wall_id, self.texture_select_mode, self.engine_state, self.texture_slot_id, self.axis_lock
            )

            self.window_surface = pg.surfarray.make_surface(self.screen)
            self.window.blit(self.window_surface, (0, 0))

            pg.display.flip()
            self.clock.tick(60)

            self.prev_mouse_r_pressed = self.mouse_r_pressed
            self.prev_mouse_l_pressed = self.mouse_l_pressed
            self.prev_mouse_m_pressed = self.mouse_m_pressed
            self.prev_keys = copy(self.keys)

        pg.quit()