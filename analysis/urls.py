from django.urls import path
from .views import (
    upload_video, analyze_video_api, ball_speed_api, release_angle_height_api,
    dtw_similarity_api, skeleton_coords_api, videos_list_api, delete_video_api,
    players_list_api, player_detail_api, player_create_api, player_update_api, player_delete_api,
    player_season_stats_api, video_detail_api, frame_metrics_api, register_optimum_api, arm_trajectory_api,
    dtw_similarity_score_api
)

urlpatterns = [
    # Video APIs
    path('upload', upload_video, name='upload_video'),  # 비디오 업로드 API
    path('videos', videos_list_api, name='videos_list_api'),  # 비디오 목록 조회 API
    path('videos/delete/<uuid:video_id>', delete_video_api, name='delete_video_api'),  # 비디오 삭제 API
    path('analyze/<uuid:video_id>', analyze_video_api, name='analyze_video_api'),  # 비디오 분석 요청 API

    # Analysis data APIs
    path('ball_speed/<uuid:video_id>', ball_speed_api, name='ball_speed_api'),  # 구속 분석 결과 조회 API
    path('release_angle_height/<uuid:video_id>', release_angle_height_api, name='release_angle_height_api'),  # 릴리즈 각도 및 높이 분석 결과 조회 API
    path('dtw_similarity', dtw_similarity_api, name='dtw_similarity_api'),  # DTW 유사도 분석 요청 API
    path('dtw_similarity/<uuid:player_id>', dtw_similarity_score_api, name='dtw_similarity_score_api'),  # 선수별 DTW 유사도 점수 이력 조회 API
    path('skeleton_coords/<uuid:video_id>', skeleton_coords_api, name='skeleton_coords_api'),  # 스켈레톤 좌표 데이터 조회 API
    path('videos/<uuid:video_id>', video_detail_api, name='video_detail_api'),  # 비디오 상세 정보 조회 API
    path('frame_metrics/<uuid:video_id>', frame_metrics_api, name='frame_metrics_api'),  # 프레임별 분석 지표 조회 API
    path('arm_trajectory/<uuid:video_id>', arm_trajectory_api, name='arm_trajectory_api'),  # 팔 궤적 데이터 조회 API

    # Player APIs
    path('players', players_list_api, name='players_list_api'),  # 선수 목록 조회 API
    path('players/create', player_create_api, name='player_create_api'),  # 선수 등록 API
    path('players/<uuid:player_id>', player_detail_api, name='player_detail_api'),  # 선수 상세 정보 조회 API
    path('players/update/<uuid:player_id>', player_update_api, name='player_update_api'),  # 선수 정보 수정 API
    path('players/delete/<uuid:player_id>', player_delete_api, name='player_delete_api'),  # 선수 삭제 API
    path('players/seasons/<uuid:player_id>', player_season_stats_api, name='player_season_stats_api'),  # 선수 시즌 기록 조회 및 추가 API
    path('players/optimum_form/<uuid:player_id>', register_optimum_api, name='register_optimum_api')  # 선수의 최적 폼 영상 등록 API
]
