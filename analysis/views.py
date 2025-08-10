from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
import json
from datetime import datetime

from .models import VideoAnalysis, Player
from .utils import (
    analyze_video, get_ball_trajectory_and_speed,
    get_joint_angles, get_hand_height,
    generate_average_forms, evaluate_against_average_form
)

# YOLO 모델 전역 로드 (최초 1회)
yolo_model = None
try:
    from ultralytics import YOLO
    yolo_model = YOLO(os.path.join(settings.BASE_DIR, 'model_path', 'best_baseball_ball.pt'))
except Exception as e:
    print(f"[경고] YOLO 모델 로드 실패: {e}")

# Create your views here.

@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        
        # 선수 정보 받기
        player_name = request.POST.get('player_name')
        birth_date = request.POST.get('birth_date')
        
        if not player_name or not birth_date:
            return JsonResponse({
                'result': 'fail', 
                'reason': '선수 이름과 생년월일을 모두 입력해주세요'
            }, status=400)
        
        # 선수 찾기 또는 생성
        try:
            from datetime import datetime
            birth_date_obj = datetime.strptime(birth_date, '%Y-%m-%d').date()
            player, created = Player.objects.get_or_create(
                name=player_name,
                birth_date=birth_date_obj,
                defaults={
                    'name': player_name,
                    'birth_date': birth_date_obj
                }
            )
        except ValueError:
            return JsonResponse({
                'result': 'fail', 
                'reason': '생년월일 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요'
            }, status=400)
        
        save_dir = os.path.join(settings.BASE_DIR, 'media', 'videos')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, video_file.name)
        with open(save_path, 'wb+') as f:
            for chunk in video_file.chunks():
                f.write(chunk)
        
        # DB row 생성 (선수 정보 포함)
        video_obj, created = VideoAnalysis.objects.get_or_create(
            filename=video_file.name,
            defaults={'player': player}
        )
        
        # 기존 영상이었다면 선수 정보 업데이트
        if not created:
            video_obj.player = player
            video_obj.save()
        
        return JsonResponse({
            'result': 'success', 
            'filename': video_file.name, 
            'path': save_path,
            'player_id': player.id,
            'player_name': player.name,
            'player_created': created
        })
    return JsonResponse({'result': 'fail', 'reason': 'No file uploaded'}, status=400)

@csrf_exempt
def analyze_video_api(request):
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
            filename = data.get('filename')
        except Exception:
            return JsonResponse({'result': 'fail', 'reason': 'Invalid JSON'}, status=400)
        if not filename:
            return JsonResponse({'result': 'fail', 'reason': 'No filename provided'}, status=400)
        video_path = os.path.join(settings.BASE_DIR, 'media', 'videos', filename)
        if not os.path.exists(video_path):
            return JsonResponse({'result': 'fail', 'reason': 'File not found'}, status=404)

        # 1. 기본 분석
        result = analyze_video(video_path, yolo_model=yolo_model)
        release_frame = result['release_frame']
        width = result['width']
        height = result['height']
        knee_xy = None
        ankle_xy = None
        if release_frame is not None and result['landmarks_list'][release_frame] is not None:
            lm = result['landmarks_list'][release_frame]
            knee_xy = [int(lm[25].x * width), int(lm[25].y * height)]
            ankle_xy = [int(lm[27].x * width), int(lm[27].y * height)]
        else:
            lm = None

        # 2. DB row 조회/생성
        video_obj, _ = VideoAnalysis.objects.get_or_create(filename=filename)
        video_obj.frame_list = result['frame_list']
        video_obj.fixed_frame = result['fixed_frame']
        video_obj.release_frame = release_frame
        video_obj.width = width
        video_obj.height = height
        video_obj.release_frame_knee = knee_xy
        video_obj.release_frame_ankle = ankle_xy

        # 3. ball_speed 분석 및 저장
        ball_speed_result = None
        if release_frame is not None and knee_xy and ankle_xy:
            import numpy as np
            shin_length = float(np.linalg.norm(np.array(knee_xy) - np.array(ankle_xy)))
            ball_speed_result = get_ball_trajectory_and_speed(
                video_path, int(release_frame), yolo_model, int(width), int(height), shin_length
            )
            video_obj.ball_speed = ball_speed_result

        # 4. release_angle_height 분석 및 저장
        release_angle_height_result = None
        if lm is not None:
            angles = get_joint_angles(lm, width, height)
            hand_height = get_hand_height(lm, width, height)
            release_angle_height_result = {
                "angles": angles,
                "hand_height": hand_height
            }
            video_obj.release_angle_height = release_angle_height_result

        # 5. skeleton_coords (전체 프레임별 랜드마크 좌표) 저장
        skeleton_coords_result = []
        for frame_lm in result['landmarks_list']:
            if frame_lm is None:
                skeleton_coords_result.append(None)
            else:
                skeleton_coords_result.append([
                    {
                        'x': float(l.x),
                        'y': float(l.y),
                        'visibility': float(l.visibility),
                        'x_pixel': int(l.x * width),
                        'y_pixel': int(l.y * height)
                    }
                    for l in frame_lm
                ])
        video_obj.skeleton_coords = skeleton_coords_result

        # 6. 저장
        video_obj.save()

        # 7. 응답
        return JsonResponse({
            'result': 'success',
            'frame_list': result['frame_list'],
            'fixed_frame': result['fixed_frame'],
            'release_frame': release_frame,
            'width': width,
            'height': height,
            'release_frame_knee': knee_xy,
            'release_frame_ankle': ankle_xy,
            'ball_speed': ball_speed_result,
            'release_angle_height': release_angle_height_result,
            'skeleton_coords': skeleton_coords_result,
        })
    return JsonResponse({'result': 'fail', 'reason': 'POST only'}, status=405)

@csrf_exempt
def ball_speed_api(request):
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
            video_id = data.get('id')
        except Exception:
            return JsonResponse({'result': 'fail', 'reason': 'Invalid JSON'}, status=400)
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No id provided'}, status=400)
        try:
            video_obj = VideoAnalysis.objects.get(id=video_id)
        except VideoAnalysis.DoesNotExist:
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
        return JsonResponse({'result': 'success', 'ball_speed': video_obj.ball_speed})
    return JsonResponse({'result': 'fail', 'reason': 'POST only'}, status=405)

@csrf_exempt
def release_angle_height_api(request):
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
            video_id = data.get('id')
        except Exception:
            return JsonResponse({'result': 'fail', 'reason': 'Invalid JSON'}, status=400)
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No id provided'}, status=400)
        try:
            video_obj = VideoAnalysis.objects.get(id=video_id)
        except VideoAnalysis.DoesNotExist:
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
        return JsonResponse({'result': 'success', 'release_angle_height': video_obj.release_angle_height})
    return JsonResponse({'result': 'fail', 'reason': 'POST only'}, status=405)

@csrf_exempt
def dtw_similarity_api(request):
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
            average_ids = data.get('average_ids')  # list
            test_id = data.get('test_id')
            used_ids = data.get('used_ids', [11, 12, 14, 16, 23, 24, 25])
        except Exception:
            return JsonResponse({'result': 'fail', 'reason': 'Invalid JSON'}, status=400)
        if not (average_ids and test_id):
            return JsonResponse({'result': 'fail', 'reason': '필수 정보 누락'}, status=400)
        # DB에서 파일명 조회
        from .models import VideoAnalysis
        avg_objs = VideoAnalysis.objects.filter(id__in=average_ids)
        if avg_objs.count() != len(average_ids):
            return JsonResponse({'result': 'fail', 'reason': 'average_ids 중 일부가 DB에 없음'}, status=404)
        try:
            test_obj = VideoAnalysis.objects.get(id=test_id)
        except VideoAnalysis.DoesNotExist:
            return JsonResponse({'result': 'fail', 'reason': 'test_id가 DB에 없음'}, status=404)
        avg_paths = [os.path.join(settings.BASE_DIR, 'media', 'videos', obj.filename) for obj in avg_objs]
        test_path = os.path.join(settings.BASE_DIR, 'media', 'videos', test_obj.filename)
        for p in avg_paths + [test_path]:
            if not os.path.exists(p):
                return JsonResponse({'result': 'fail', 'reason': f'File not found: {p}'}, status=404)
        # 평균폼 생성
        avg_forms = generate_average_forms(avg_paths, used_ids)
        # 유사도 분석
        phase_scores, phase_distances, worst_idx = evaluate_against_average_form(test_path, avg_forms, used_ids)
        # numpy 타입을 파이썬 기본 타입으로 변환
        phase_scores = [float(x) for x in phase_scores]
        phase_distances = [float(x) for x in phase_distances]
        worst_idx = int(worst_idx)
        return JsonResponse({
            'result': 'success',
            'phase_scores': phase_scores,
            'phase_distances': phase_distances,
            'worst_phase': worst_idx,
        })
    return JsonResponse({'result': 'fail', 'reason': 'POST only'}, status=405)

@csrf_exempt
def skeleton_coords_api(request):
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
            video_id = data.get('id')
        except Exception:
            return JsonResponse({'result': 'fail', 'reason': 'Invalid JSON'}, status=400)
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No id provided'}, status=400)
        try:
            video_obj = VideoAnalysis.objects.get(id=video_id)
        except VideoAnalysis.DoesNotExist:
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
        return JsonResponse({'result': 'success', 'skeleton_coords': video_obj.skeleton_coords})
    return JsonResponse({'result': 'fail', 'reason': 'POST only'}, status=405)

@csrf_exempt
def videos_list_api(request):
    if request.method == 'GET':
        videos = VideoAnalysis.objects.all().order_by('-upload_time')
        result = [
            {
                'id': v.id,
                'filename': v.filename,
                'upload_time': v.upload_time.isoformat()
            }
            for v in videos
        ]
        return JsonResponse(result, safe=False)
    return JsonResponse({'result': 'fail', 'reason': 'GET only'}, status=405)

@csrf_exempt
def players_list_api(request):
    """선수 목록 조회"""
    if request.method == 'GET':
        players = Player.objects.all().order_by('name')
        players_data = []
        for player in players:
            players_data.append({
                'id': player.id,
                'name': player.name,
                'birth_date': player.birth_date.isoformat() if player.birth_date else None,
                'height': player.height,
                'weight': player.weight,
                'throwing_hand': player.throwing_hand,
                'batting_hand': player.batting_hand,
                'video_count': player.videos.count()  # 해당 선수의 영상 개수
            })
        return JsonResponse(players_data, safe=False)
    
    return JsonResponse({'result': 'fail', 'reason': 'GET method only'}, status=405)

@csrf_exempt
def player_detail_api(request, player_id):
    """개별 선수 조회"""
    if request.method == 'GET':
        try:
            player = Player.objects.get(id=player_id)
            player_data = {
                'id': player.id,
                'name': player.name,
                'birth_date': player.birth_date.isoformat() if player.birth_date else None,
                'height': player.height,
                'weight': player.weight,
                'throwing_hand': player.throwing_hand,
                'batting_hand': player.batting_hand,
                'video_count': player.videos.count()
            }
            return JsonResponse(player_data)
        except Player.DoesNotExist:
            return JsonResponse({'result': 'fail', 'reason': '선수를 찾을 수 없습니다'}, status=404)
    
    return JsonResponse({'result': 'fail', 'reason': 'GET method only'}, status=405)

@csrf_exempt
def player_create_api(request):
    """선수 등록"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            name = data.get('name')
            birth_date_str = data.get('birth_date')
            
            if not name:
                return JsonResponse({'result': 'fail', 'reason': '선수 이름을 입력해주세요'}, status=400)
            
            # 생년월일 처리
            birth_date = None
            if birth_date_str:
                try:
                    birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
                except ValueError:
                    return JsonResponse({'result': 'fail', 'reason': '생년월일 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요'}, status=400)
            
            # 기존 선수 확인 (이름과 생년월일로)
            if birth_date:
                existing_player = Player.objects.filter(name=name, birth_date=birth_date).first()
            else:
                existing_player = Player.objects.filter(name=name, birth_date__isnull=True).first()
            
            if existing_player:
                return JsonResponse({'result': 'fail', 'reason': '이미 존재하는 선수입니다'}, status=400)
            
            # 새 선수 생성
            player = Player.objects.create(
                name=name,
                birth_date=birth_date,
                height=data.get('height'),
                weight=data.get('weight'),
                throwing_hand=data.get('throwing_hand'),
                batting_hand=data.get('batting_hand')
            )
            
            return JsonResponse({
                'result': 'success',
                'player': {
                    'id': player.id,
                    'name': player.name,
                    'birth_date': player.birth_date.isoformat() if player.birth_date else None,
                    'height': player.height,
                    'weight': player.weight,
                    'throwing_hand': player.throwing_hand,
                    'batting_hand': player.batting_hand
                }
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'result': 'fail', 'reason': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'result': 'fail', 'reason': str(e)}, status=500)
    
    return JsonResponse({'result': 'fail', 'reason': 'POST method only'}, status=405)

@csrf_exempt
def player_update_api(request, player_id):
    """선수 정보 수정"""
    if request.method == 'PUT':
        try:
            data = json.loads(request.body)
            
            try:
                player = Player.objects.get(id=player_id)
            except Player.DoesNotExist:
                return JsonResponse({'result': 'fail', 'reason': '선수를 찾을 수 없습니다'}, status=404)
            
            # 수정할 필드들
            if 'name' in data:
                player.name = data['name']
            
            if 'birth_date' in data:
                if data['birth_date']:
                    try:
                        player.birth_date = datetime.strptime(data['birth_date'], '%Y-%m-%d').date()
                    except ValueError:
                        return JsonResponse({'result': 'fail', 'reason': '생년월일 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요'}, status=400)
                else:
                    player.birth_date = None
            
            if 'height' in data:
                player.height = data['height']
            
            if 'weight' in data:
                player.weight = data['weight']
            
            if 'throwing_hand' in data:
                player.throwing_hand = data['throwing_hand']
            
            if 'batting_hand' in data:
                player.batting_hand = data['batting_hand']
            
            player.save()
            
            return JsonResponse({
                'result': 'success',
                'player': {
                    'id': player.id,
                    'name': player.name,
                    'birth_date': player.birth_date.isoformat() if player.birth_date else None,
                    'height': player.height,
                    'weight': player.weight,
                    'throwing_hand': player.throwing_hand,
                    'batting_hand': player.batting_hand
                }
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'result': 'fail', 'reason': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'result': 'fail', 'reason': str(e)}, status=500)
    
    return JsonResponse({'result': 'fail', 'reason': 'PUT method only'}, status=405)

@csrf_exempt
def player_delete_api(request, player_id):
    """선수 삭제"""
    if request.method == 'DELETE':
        try:
            player = Player.objects.get(id=player_id)
            
            # 해당 선수의 영상이 있는지 확인
            video_count = player.videos.count()
            if video_count > 0:
                return JsonResponse({
                    'result': 'fail', 
                    'reason': f'이 선수와 연결된 영상이 {video_count}개 있습니다. 영상을 먼저 삭제해주세요.'
                }, status=400)
            
            player.delete()
            return JsonResponse({'result': 'success'})
            
        except Player.DoesNotExist:
            return JsonResponse({'result': 'fail', 'reason': '선수를 찾을 수 없습니다'}, status=404)
        except Exception as e:
            return JsonResponse({'result': 'fail', 'reason': str(e)}, status=500)
    
    return JsonResponse({'result': 'fail', 'reason': 'DELETE method only'}, status=405)
