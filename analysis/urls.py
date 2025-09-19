from django.urls import path
from .views import (
    upload_video, analyze_video_api, ball_speed_api, release_angle_height_api, 
    dtw_similarity_api, skeleton_coords_api, videos_list_api, delete_video_api,
    players_list_api, player_detail_api, player_create_api, player_update_api, player_delete_api,
    player_season_stats_api
)

urlpatterns = [
    # Video APIs
    path('upload', upload_video, name='upload_video'),
    path('videos', videos_list_api, name='videos_list_api'),
    path('videos/delete/<uuid:video_id>', delete_video_api, name='delete_video_api'),
    path('analyze/<uuid:video_id>', analyze_video_api, name='analyze_video_api'),

    # Analysis data APIs
    path('ball_speed/<uuid:video_id>', ball_speed_api, name='ball_speed_api'),
    path('release_angle_height/<uuid:video_id>', release_angle_height_api, name='release_angle_height_api'),
    path('dtw_similarity', dtw_similarity_api, name='dtw_similarity_api'),
    path('skeleton_coords/<uuid:video_id>', skeleton_coords_api, name='skeleton_coords_api'),
    
    # Player APIs
    path('players', players_list_api, name='players_list_api'),
    path('players/create', player_create_api, name='player_create_api'),
    path('players/<uuid:player_id>', player_detail_api, name='player_detail_api'),
    path('players/update/<uuid:player_id>', player_update_api, name='player_update_api'),
    path('players/delete/<uuid:player_id>', player_delete_api, name='player_delete_api'),
    path('players/seasons/<uuid:player_id>', player_season_stats_api, name='player_season_stats_api'),
]