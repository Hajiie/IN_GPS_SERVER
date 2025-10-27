from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db import IntegrityError
import os
import json
import uuid
from datetime import datetime
from decimal import Decimal

from .models import Player, PlayerSeason, PlayerStats, VideoAnalysis
from .utils import (
    create_thumbnail_from_video,
    analyze_video, get_ball_trajectory_and_speed,
    get_joint_angles, get_hand_height,
    evaluate_pair_with_dynamic_masks, PhaseSegmentationError
)

# YOLO 모델 전역 로드 (최초 1회)
yolo_model = None
try:
    from ultralytics import YOLO
    yolo_model = YOLO(os.path.join(settings.BASE_DIR, 'model_path', 'best_baseball_ball.pt'))
except Exception as e:
    print(f"[경고] YOLO 모델 로드 실패: {e}")

@csrf_exempt
def upload_video(request):
    if request.method != 'POST' or not request.FILES.get('video'):
        return JsonResponse({'result': 'fail', 'reason': '올바르지 않은 요청입니다.'}, status=400)

    video_file = request.FILES['video']
    video_name = request.POST.get('video_name')
    player_name = request.POST.get('player_name')
    birth_date = request.POST.get('birth_date')

    if not player_name or not birth_date:
        return JsonResponse({'result': 'fail', 'reason': '선수 이름과 생년월일을 모두 입력해주세요'}, status=400)

    try:
        birth_date_obj = datetime.strptime(birth_date, '%Y-%m-%d').date()
        player, _ = Player.objects.get_or_create(
            name=player_name,
            birth_date=birth_date_obj,
            defaults={'name': player_name, 'birth_date': birth_date_obj}
        )
    except ValueError:
        return JsonResponse({'result': 'fail', 'reason': '생년월일 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요'}, status=400)

    try:
        video_obj = VideoAnalysis.objects.create(
            player=player,
            video_name=video_name if video_name else None,
            video_file=video_file
        )
    except IntegrityError:
        return JsonResponse({
            'result': 'fail',
            'reason': f'이미 사용 중인 영상 이름입니다: "{video_name}"'
        }, status=409)

    # --- Thumbnail Generation ---
    thumbnail_content = create_thumbnail_from_video(video_file)
    if thumbnail_content:
        thumbnail_name = f"{os.path.splitext(os.path.basename(video_obj.video_file.name))[0]}.jpg"
        video_obj.thumbnail.save(thumbnail_name, thumbnail_content, save=True)

    return JsonResponse({
        'result': 'success',
        'video_id': str(video_obj.id),
        'video_name': video_obj.video_name or os.path.basename(video_obj.video_file.name),
        'video_url': video_obj.video_file.url,
        'thumbnail_url': video_obj.thumbnail.url if video_obj.thumbnail else None,
        'player_id': str(player.id),
        'player_name': player.name,
    })

@csrf_exempt
def videos_list_api(request):
    if request.method == 'GET':
        videos = VideoAnalysis.objects.all().order_by('-upload_time')
        result = [
            {
                'id': str(v.id),
                'video_name': v.video_name or os.path.basename(v.video_file.name),
                'video_url': v.video_file.url,
                'thumbnail_url': v.thumbnail.url if v.thumbnail else None,
                'upload_time': v.upload_time.isoformat(),
                'player_id': str(v.player.id) if v.player else None,
                'player_name': v.player.name if v.player else None,
            }
            for v in videos
        ]
        return JsonResponse(result, safe=False)
    return JsonResponse({'result': 'fail', 'reason': 'GET only'}, status=405)

@csrf_exempt
def delete_video_api(request, video_id):
    if request.method != 'DELETE':
        return JsonResponse({'result': 'fail', 'reason': 'DELETE method only'}, status=405)
    
    try:
        video_obj = get_object_or_404(VideoAnalysis, id=video_id)
        video_name = video_obj.video_name or os.path.basename(video_obj.video_file.name)
        video_obj.delete()
        return JsonResponse({'result': 'success', 'message': f'Video "{video_name}" deleted successfully.'})
    except VideoAnalysis.DoesNotExist:
        return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
    except Exception as e:
        return JsonResponse({'result': 'fail', 'reason': str(e)}, status=500)

@csrf_exempt
def analyze_video_api(request, video_id):
    if request.method == 'POST':
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No video_id provided'}, status=400)
        
        try:
            video_obj = get_object_or_404(VideoAnalysis, id=video_id)
            video_path = video_obj.video_file.path
        except (VideoAnalysis.DoesNotExist, ValueError):
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)

        if not os.path.exists(video_path):
            return JsonResponse({'result': 'fail', 'reason': f'File not found at path: {video_path}'}, status=404)

        result = analyze_video(video_path, yolo_model=yolo_model)
        release_frame = result['release_frame']
        width = result['width']
        height = result['height']
        knee_xy, ankle_xy, lm = None, None, None
        if release_frame is not None and result['landmarks_list'][release_frame] is not None:
            lm = result['landmarks_list'][release_frame]
            knee_xy = [int(lm[25].x * width), int(lm[25].y * height)]
            ankle_xy = [int(lm[27].x * width), int(lm[27].y * height)]

        video_obj.frame_list = result['frame_list']
        video_obj.fixed_frame = result['fixed_frame']
        video_obj.release_frame = release_frame
        video_obj.width = width
        video_obj.height = height
        video_obj.release_frame_knee = knee_xy
        video_obj.release_frame_ankle = ankle_xy

        ball_speed_result = None
        if release_frame is not None and knee_xy and ankle_xy:
            import numpy as np
            shin_length = float(np.linalg.norm(np.array(knee_xy) - np.array(ankle_xy)))
            ball_speed_result = get_ball_trajectory_and_speed(
                video_path, int(release_frame), yolo_model, shin_length
            )
            video_obj.ball_speed = ball_speed_result

        release_angle_height_result = None
        if lm is not None:
            angles = get_joint_angles(lm, width, height)
            hand_height = get_hand_height(lm, width, height)
            release_angle_height_result = {"angles": angles, "hand_height": hand_height}
            video_obj.release_angle_height = release_angle_height_result

        skeleton_coords_result = []
        for frame_lm in result['landmarks_list']:
            if frame_lm is None:
                skeleton_coords_result.append(None)
            else:
                skeleton_coords_result.append([
                    {'x': float(l.x), 'y': float(l.y), 'visibility': float(l.visibility), 'x_pixel': int(l.x * width), 'y_pixel': int(l.y * height)}
                    for l in frame_lm
                ])
        video_obj.skeleton_coords = skeleton_coords_result
        #video_obj.landmarks_list = result['landmarks_list'] # This might be large, consider if needed
        video_obj.save()

        return JsonResponse({
            'result': 'success',
            'frame_list': result['frame_list'],
            'fixed_frame': result['fixed_frame'],
            'release_frame': release_frame,
            'ball_speed': ball_speed_result,
            'release_angle_height': release_angle_height_result,
        })
    return JsonResponse({'result': 'fail', 'reason': 'POST only'}, status=405)

@csrf_exempt
def ball_speed_api(request, video_id):
    if request.method == 'POST':
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No id provided'}, status=400)
        try:
            video_obj = get_object_or_404(VideoAnalysis, id=video_id)
        except (VideoAnalysis.DoesNotExist, ValueError):
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
        return JsonResponse({'result': 'success', 'ball_speed': video_obj.ball_speed})
    return JsonResponse({'result': 'fail', 'reason': 'POST only'}, status=405)

@csrf_exempt
def release_angle_height_api(request, video_id):
    if request.method == 'POST':
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No id provided'}, status=400)
        try:
            video_obj = get_object_or_404(VideoAnalysis, id=video_id)
        except (VideoAnalysis.DoesNotExist, ValueError):
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
        return JsonResponse({'result': 'success', 'release_angle_height': video_obj.release_angle_height})
    return JsonResponse({'result': 'fail', 'reason': 'POST only'}, status=405)

@csrf_exempt
def dtw_similarity_api(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        reference_id = data.get('reference_id')
        test_id = data.get('test_id')
        used_ids = data.get('used_ids', [11, 12, 14, 16, 23, 24, 25, 27])

        if not (reference_id and test_id):
            return JsonResponse({'result': 'fail', 'reason': 'reference_id와 test_id는 필수 정보입니다.'}, status=400)
        
        try:
            ref_obj = get_object_or_404(VideoAnalysis, id=reference_id)
            test_obj = get_object_or_404(VideoAnalysis, id=test_id)
        except (VideoAnalysis.DoesNotExist, ValueError):
            return JsonResponse({'result': 'fail', 'reason': 'ID가 DB에 없음'}, status=404)

        ref_path = ref_obj.video_file.path
        test_path = test_obj.video_file.path

        if not os.path.exists(ref_path):
            return JsonResponse({'result': 'fail', 'reason': f'File not found: {ref_path}'}, status=404)
        if not os.path.exists(test_path):
            return JsonResponse({'result': 'fail', 'reason': f'File not found: {test_path}'}, status=404)

        try:
            phase_scores, phase_distances, overall_score, worst_idx = evaluate_pair_with_dynamic_masks(
                reference_video=ref_path,
                test_video=test_path,
                used_ids=used_ids,
                yolo_model=yolo_model
            )
        except PhaseSegmentationError as e:
            return JsonResponse({'result': 'fail', 'reason': f'영상 분할 실패: {e}'}, status=500)
        except Exception as e:
            return JsonResponse({'result': 'fail', 'reason': f'분석 중 오류 발생: {e}'}, status=500)
        
        return JsonResponse({
            'result': 'success',
            'phase_scores': [float(s) for s in phase_scores],
            'phase_distances': [float(d) for d in phase_distances],
            'overall_score': float(overall_score),
            'worst_phase': int(worst_idx) if worst_idx is not None else None,
        })
    return JsonResponse({'result': 'fail', 'reason': 'POST only'}, status=405)


# ... (rest of the file remains the same) ...

@csrf_exempt
def skeleton_coords_api(request, video_id):
    if request.method == 'POST':
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No id provided'}, status=400)
        try:
            video_obj = get_object_or_404(VideoAnalysis, id=video_id)
        except (VideoAnalysis.DoesNotExist, ValueError):
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
        return JsonResponse({'result': 'success', 'skeleton_coords': video_obj.skeleton_coords})
    return JsonResponse({'result': 'fail', 'reason': 'POST only'}, status=405)

@csrf_exempt
def players_list_api(request):
    if request.method == 'GET':
        players = Player.objects.all().order_by('name')
        players_data = [
            {
                'id': str(player.id),
                'name': player.name,
                'birth_date': player.birth_date.isoformat() if player.birth_date else None,
                'height': player.height,
                'weight': player.weight,
                'throwing_hand': player.throwing_hand,
                'batting_hand': player.batting_hand,
                'video_count': player.videos.count()
            }
            for player in players
        ]
        return JsonResponse(players_data, safe=False)
    return JsonResponse({'result': 'fail', 'reason': 'GET method only'}, status=405)

@csrf_exempt
def player_detail_api(request, player_id):
    if request.method == 'GET':
        player = get_object_or_404(Player, id=player_id)
        player_data = {
            'id': str(player.id),
            'name': player.name,
            'birth_date': player.birth_date.isoformat() if player.birth_date else None,
            'height': player.height,
            'weight': player.weight,
            'throwing_hand': player.throwing_hand,
            'batting_hand': player.batting_hand,
            'video_count': player.videos.count()
        }
        return JsonResponse(player_data)
    return JsonResponse({'result': 'fail', 'reason': 'GET method only'}, status=405)

@csrf_exempt
def player_create_api(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        name = data.get('name')
        birth_date_str = data.get('birth_date')
        if not name:
            return JsonResponse({'result': 'fail', 'reason': '선수 이름을 입력해주세요'}, status=400)
        
        birth_date = None
        if birth_date_str:
            try:
                birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
            except ValueError:
                return JsonResponse({'result': 'fail', 'reason': '생년월일 형식이 올바르지 않습니다.'}, status=400)
        
        query = Player.objects.filter(name=name, birth_date=birth_date) if birth_date else Player.objects.filter(name=name, birth_date__isnull=True)
        if query.exists():
            return JsonResponse({'result': 'fail', 'reason': '이미 존재하는 선수입니다'}, status=400)
        
        player = Player.objects.create(
            name=name, birth_date=birth_date,
            height=data.get('height'), weight=data.get('weight'),
            throwing_hand=data.get('throwing_hand'), batting_hand=data.get('batting_hand')
        )
        
        return JsonResponse({
            'result': 'success',
            'player': {
                'id': str(player.id),
                'name': player.name,
                'birth_date': player.birth_date.isoformat() if player.birth_date else None,
            }
        })
    return JsonResponse({'result': 'fail', 'reason': 'POST method only'}, status=405)

@csrf_exempt
def player_update_api(request, player_id):
    if request.method == 'PUT':
        data = json.loads(request.body)
        player = get_object_or_404(Player, id=player_id)
        
        player.name = data.get('name', player.name)
        player.height = data.get('height', player.height)
        player.weight = data.get('weight', player.weight)
        player.throwing_hand = data.get('throwing_hand', player.throwing_hand)
        player.batting_hand = data.get('batting_hand', player.batting_hand)
        
        if 'birth_date' in data:
            birth_date_str = data.get('birth_date')
            if birth_date_str:
                try:
                    player.birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
                except ValueError:
                    return JsonResponse({'result': 'fail', 'reason': '생년월일 형식이 올바르지 않습니다.'}, status=400)
            else:
                player.birth_date = None
        
        player.save()
        
        return JsonResponse({
            'result': 'success',
            'player': {
                'id': str(player.id),
                'name': player.name,
                'birth_date': player.birth_date.isoformat() if player.birth_date else None,
                'height': player.height,
                'weight': player.weight,
                'throwing_hand': player.throwing_hand,
                'batting_hand': player.batting_hand
            }
        })
    return JsonResponse({'result': 'fail', 'reason': 'PUT method only'}, status=405)

@csrf_exempt
def player_delete_api(request, player_id):
    if request.method == 'DELETE':
        player = get_object_or_404(Player, id=player_id)
        if player.videos.count() > 0:
            return JsonResponse({
                'result': 'fail', 
                'reason': f'이 선수와 연결된 영상이 {player.videos.count()}개 있습니다. 영상을 먼저 삭제해주세요.'
            }, status=400)
        
        player.delete()
        return JsonResponse({'result': 'success'})
    return JsonResponse({'result': 'fail', 'reason': 'DELETE method only'}, status=405)

@csrf_exempt
def player_season_stats_api(request, player_id):
    player = get_object_or_404(Player, id=player_id)

    if request.method == 'GET':
        seasons = player.seasons.select_related('stats').order_by('-year')
        seasons_data = []
        for season in seasons:
            stats_data = {}
            if hasattr(season, 'stats'):
                stats = season.stats
                stats_data = {
                    'era': str(stats.era) if stats.era is not None else None,
                    'games': stats.games,
                    'wins': stats.wins,
                    'losses': stats.losses,
                    'saves': stats.saves,
                    'holds': stats.holds,
                    'win_rate': str(stats.win_rate) if stats.win_rate is not None else None,
                    'innings_pitched': str(stats.innings_pitched) if stats.innings_pitched is not None else None,
                    'hits_allowed': stats.hits_allowed,
                    'home_runs_allowed': stats.home_runs_allowed,
                    'walks': stats.walks,
                    'hit_by_pitch': stats.hit_by_pitch,
                    'strikeouts': stats.strikeouts,
                    'runs_allowed': stats.runs_allowed,
                    'earned_runs': stats.earned_runs,
                }

            seasons_data.append({
                'season_id': season.id,
                'year': season.year,
                'team': season.team,
                'stats': stats_data
            })
        return JsonResponse(seasons_data, safe=False)

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            year = data.get('year')
            team = data.get('team')

            if not year or not team:
                return JsonResponse({'result': 'fail', 'reason': 'year와 team은 필수 항목입니다.'}, status=400)

            season, season_created = PlayerSeason.objects.update_or_create(
                player=player,
                year=year,
                defaults={'team': team}
            )

            stats_data = {
                'era': data.get('era'), 'games': data.get('games'), 'wins': data.get('wins'),
                'losses': data.get('losses'), 'saves': data.get('saves'), 'holds': data.get('holds'),
                'innings_pitched': data.get('innings_pitched'), 'hits_allowed': data.get('hits_allowed'),
                'home_runs_allowed': data.get('home_runs_allowed'), 'walks': data.get('walks'),
                'hit_by_pitch': data.get('hit_by_pitch'), 'strikeouts': data.get('strikeouts'),
                'runs_allowed': data.get('runs_allowed'), 'earned_runs': data.get('earned_runs'),
            }
            valid_stats_data = {k: v for k, v in stats_data.items() if v is not None}

            stats, stats_created = PlayerStats.objects.update_or_create(
                player_season=season,
                defaults=valid_stats_data
            )

            all_stats = {f.name: getattr(stats, f.name) for f in PlayerStats._meta.get_fields() if not f.is_relation}
            all_stats.pop('id', None)

            return JsonResponse({
                'result': 'success',
                'season_created': season_created,
                'stats_created': stats_created,
                'player_id': str(player.id),
                'season_id': season.id,
                'year': season.year,
                'team': season.team,
                'stats': {k: str(v) if isinstance(v, Decimal) else v for k, v in all_stats.items()},
            })

        except json.JSONDecodeError:
            return JsonResponse({'result': 'fail', 'reason': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'result': 'fail', 'reason': str(e)}, status=500)
    
    return JsonResponse({'result': 'fail', 'reason': 'Method not allowed'}, status=405)
