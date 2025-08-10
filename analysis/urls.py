from django.urls import path
from .views import (
    upload_video, analyze_video_api, ball_speed_api, release_angle_height_api, 
    dtw_similarity_api, skeleton_coords_api, videos_list_api,
    players_list_api, player_detail_api, player_create_api, player_update_api, player_delete_api
)

urlpatterns = [
    path('upload/', upload_video, name='upload_video'),
    path('analyze/', analyze_video_api, name='analyze_video_api'),
    path('ball_speed/', ball_speed_api, name='ball_speed_api'),
    path('release_angle_height/', release_angle_height_api, name='release_angle_height_api'),
    path('dtw_similarity/', dtw_similarity_api, name='dtw_similarity_api'),
    path('skeleton_coords/', skeleton_coords_api, name='skeleton_coords_api'),
    path('videos/', videos_list_api, name='videos_list_api'),
    
    # 선수 관리 API
    path('players/', players_list_api, name='players_list_api'),
    path('players/create/', player_create_api, name='player_create_api'),
    path('players/<int:player_id>/', player_detail_api, name='player_detail_api'),
    path('players/<int:player_id>/update/', player_update_api, name='player_update_api'),
    path('players/<int:player_id>/delete/', player_delete_api, name='player_delete_api'),
    
    # 여기에 분석 관련 API 엔드포인트를 추가할 예정
] 