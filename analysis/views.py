import json
import os
from datetime import datetime
from decimal import Decimal

from django.conf import settings
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt

from .models import Player, PlayerSeason, PlayerStats, VideoAnalysis, DTWAnalysis
from .utils import (
    create_thumbnail_from_video,
    analyze_video, get_ball_trajectory_and_speed,
    get_joint_angles, get_hand_height,
    evaluate_pair_with_dynamic_masks, PhaseSegmentationError,
    render_skeleton_video, calculate_frame_by_frame_metrics, render_arm_swing_speed_video,
    render_shoulder_angular_velocity_video, render_ball_trajectory_video
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

    # 파일 업로드 시 영상의 이름이 중복되면 return
    if request.POST.get('video_name') in [v.video_name for v in VideoAnalysis.objects.all()]:
        return JsonResponse({'result': 'fail', 'reason': '이미 사용 중인 영상 이름입니다.'}, status=409)

    video_name = request.POST.get('video_name')
    video_file = request.FILES['video']
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
            video_name=video_name if video_name else os.path.splitext(video_file.name)[0],
            video_file=video_file,
        )
    except e:
        return JsonResponse({'result': 'fail', 'reason': str(e)}, status=500)


    # --- Thumbnail Generation ---
    thumbnail_content = create_thumbnail_from_video(video_file)
    if thumbnail_content:
        # Let the model's upload_to handle the path and name
        video_obj.thumbnail.save(f"{video_obj.id}.jpg", thumbnail_content, save=True)

    return JsonResponse({
        'result': 'success',
        'video_id': str(video_obj.id),
        'video_name': video_obj.video_name,
        'video_url': video_obj.video_file.url,
        'thumbnail_url': video_obj.thumbnail.url if video_obj.thumbnail else None,
        'player_id': str(player.id),
        'player_name': player.name,
    })

@csrf_exempt
def videos_list_api(request):
    if request.method == 'GET':
        videos = VideoAnalysis.objects.all().order_by('-upload_time')
        if videos is None:
            return JsonResponse({'result': 'fail', 'reason': 'No videos found'}, status=404)
        result = [
            {
                'id': str(v.id),
                'video_name': v.video_name or os.path.basename(v.video_file.name),
                'video_url': v.video_file.url if v.video_file else None,
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
def video_detail_api(request, video_id):
    if request.method == 'GET':
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No video_id provided'}, status=400)
        try:
            video_obj = get_object_or_404(VideoAnalysis, id=video_id)
            return JsonResponse({
                'result': 'success',
                'video_name': video_obj.video_name,
                'video_url': video_obj.video_file.url if video_obj.video_file else None,
                'thumbnail_url': video_obj.thumbnail.url if video_obj.thumbnail else None,
                'upload_time': video_obj.upload_time.isoformat(),
                'player_id': str(video_obj.player.id) if video_obj.player else None,
                'player_name': video_obj.player.name if video_obj.player else None,
                'ball_speed': video_obj.ball_speed,
                'release_frame': video_obj.release_frame,
                'width': video_obj.width,
                'height': video_obj.height,
                'release_frame_knee': video_obj.release_frame_knee,
                'release_frame_ankle': video_obj.release_frame_ankle,
                'fixed_frame': video_obj.fixed_frame,
                'frame_list': video_obj.frame_list,
                'skeleton_video_url': video_obj.skeleton_video.url if video_obj.skeleton_video else None,
                'arm_video_url': video_obj.arm_swing_video.url if video_obj.arm_swing_video else None,
                'shoulder_video_url': video_obj.shoulder_swing_video.url if video_obj.shoulder_swing_video else None,
                'release_video_url': video_obj.release_video.url if video_obj.release_video else None,
                'release_angle_height': video_obj.release_angle_height,
                'frame_metrics': video_obj.frame_metrics,
                'arm_trajectory': video_obj.arm_trajectory,
                'arm_swing_speed': video_obj.arm_swing_speed,
                'shoulder_swing_speed': video_obj.shoulder_swing_speed
            })
        except VideoAnalysis.DoesNotExist:
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
    return JsonResponse({'result': 'fail', 'reason': 'GET only'}, status=405)

@csrf_exempt
def frame_metrics_api(request, video_id):
    if request.method == 'GET':
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No video_id provided'}, status=400)
        try:
            video_obj = get_object_or_404(VideoAnalysis, id=video_id)
            return JsonResponse({'result': 'success', 'frame_metrics': video_obj.frame_metrics})
        except VideoAnalysis.DoesNotExist:
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
    return JsonResponse({'result': 'fail', 'reason': 'GET only'}, status=405)

@csrf_exempt
def arm_trajectory_api(request, video_id):
    if request.method == 'GET':
        if not video_id:
            return JsonResponse({'result:': 'fail', 'reason': 'No video_id provided'}, status=400)
        try:
            video_obj = get_object_or_404(VideoAnalysis, id=video_id)
            return JsonResponse({
                'result': 'success',
                'arm_trajectory': video_obj.arm_trajectory,
                'arm_swing_speed': video_obj.arm_swing_speed,
                'shoulder_swing_speed': video_obj.shoulder_swing_speed})
        except VideoAnalysis.DoesNotExist:
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
    return JsonResponse({'result': 'fail', 'reason': 'GET only'}, status=405)


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

        analysis_result = analyze_video(video_path, yolo_model=yolo_model)
        release_frame = analysis_result['release_frame']
        width = analysis_result['width']
        height = analysis_result['height']
        knee_xy, ankle_xy, lm = None, None, None
        if release_frame is not None and analysis_result['landmarks_list'][release_frame] is not None:
            lm = analysis_result['landmarks_list'][release_frame]
            knee_xy = [int(lm[25].x * width), int(lm[25].y * height)]
            ankle_xy = [int(lm[27].x * width), int(lm[27].y * height)]

        video_obj.frame_list = analysis_result['frame_list']
        video_obj.fixed_frame = analysis_result['fixed_frame']
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
        for frame_lm in analysis_result['landmarks_list']:
            if frame_lm is None:
                skeleton_coords_result.append(None)
            else:
                skeleton_coords_result.append([
                    {'x': l.x, 'y': l.y, 'visibility': l.visibility, 'x_pixel': int(l.x * width), 'y_pixel': int(l.y * height)}
                    for l in frame_lm
                ])
        video_obj.skeleton_coords = skeleton_coords_result

        frame_metrics = calculate_frame_by_frame_metrics(analysis_result)
        video_obj.frame_metrics = frame_metrics

        skeleton_video_url = None
        arm_video_url = None
        shoulder_video_url = None
        release_video_url = None
        try:
            # Create a temporary path for the rendered video
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            used_ids = [11, 12, 14, 16, 23, 24, 25, 26, 27, 28]

            temp_save_path = os.path.join(temp_dir, f"{video_obj.id}_skeleton.mp4")
            skeleton_rendered_path = render_skeleton_video(analysis_result, temp_save_path, used_ids)
            temp_save_path = os.path.join(temp_dir, f"{video_obj.id}_arm.mp4")
            arm_rendered_path = render_arm_swing_speed_video(analysis_result, temp_save_path, used_ids)
            temp_save_path = os.path.join(temp_dir, f"{video_obj.id}_shoulder.mp4")
            shoulder_rendered_path = render_shoulder_angular_velocity_video(analysis_result, temp_save_path, used_ids)
            temp_save_path = os.path.join(temp_dir, f"{video_obj.id}_release.mp4")
            release_rendered_path = render_ball_trajectory_video(analysis_result, yolo_model,temp_save_path, used_ids)

            if skeleton_rendered_path:
                if video_obj.skeleton_video:
                    video_obj.skeleton_video.delete(save=False) # Delete old file from storage

                with open(skeleton_rendered_path, 'rb') as f:
                    video_obj.skeleton_video.save(os.path.basename(skeleton_rendered_path), ContentFile(f.read()), save=False)
                
                os.remove(skeleton_rendered_path) # Clean up temp file
                skeleton_video_url = video_obj.skeleton_video.url

            if arm_rendered_path:
                if video_obj.arm_swing_video:
                    video_obj.arm_swing_video.delete(save=False) # Delete old file from storage

                with open(arm_rendered_path, 'rb') as f:
                    video_obj.arm_swing_video.save(os.path.basename(arm_rendered_path), ContentFile(f.read()), save=False)

                os.remove(arm_rendered_path) # Clean up temp file
                arm_video_url = video_obj.arm_swing_video.url

            if shoulder_rendered_path:
                if video_obj.shoulder_swing_video:
                    video_obj.shoulder_swing_video.delete(save=False)

                with open(shoulder_rendered_path, 'rb') as f:
                    video_obj.shoulder_swing_video.save(os.path.basename(shoulder_rendered_path), ContentFile(f.read()), save=False)

                os.remove(shoulder_rendered_path)
                shoulder_video_url = video_obj.shoulder_swing_video.url

            if release_rendered_path:
                if video_obj.release_video:
                    video_obj.release_video.delete(save=False)

                with open(release_rendered_path, 'rb') as f:
                    video_obj.release_video.save(os.path.basename(release_rendered_path), ContentFile(f.read()), save=False)

                os.remove(release_rendered_path)
                release_video_url = video_obj.release_video.url



        except Exception as e:
            print(f"Error rendering skeleton video: {e}")

        # Filter out None values from the trajectory list
        arm_trajectory = [item for item in analysis_result['arm_trajectory'] if item is not None]

        # Convert numpy arrays to lists before saving to JSONField
        # Filter out nan values and convert to a list of floats
        wrist_speeds_mps_list = [float(v) for v in analysis_result['wrist_speeds_mps'] if not np.isnan(v)]
        shoulder_speeds_degps_list = [float(v) for v in analysis_result['shoulder_angular_velocities_degps'] if not np.isnan(v)]

        video_obj.arm_trajectory = arm_trajectory
        video_obj.arm_swing_speed = wrist_speeds_mps_list
        video_obj.shoulder_swing_speed = shoulder_speeds_degps_list
        video_obj.save()

        return JsonResponse({
            'result': 'success',
            'frame_list': analysis_result['frame_list'],
            'fixed_frame': analysis_result['fixed_frame'],
            'release_frame': release_frame,
            'ball_speed': ball_speed_result,
            'release_angle_height': release_angle_height_result,
            'skeleton_video_url': skeleton_video_url,
            'arm_swing_video_url': arm_video_url,
            'shoulder_swing_video_url': shoulder_video_url,
            'release_video_url': release_video_url,
            'frame_metrics': frame_metrics,
            'wrist_speeds_mps': wrist_speeds_mps_list,
            'shoulder_angular_velocities_degps': shoulder_speeds_degps_list,
            'arm_trajectory': arm_trajectory
        })
    return JsonResponse({'result': 'fail', 'reason': 'POST only'}, status=405)

@csrf_exempt
def ball_speed_api(request, video_id):
    if request.method == 'GET':
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No id provided'}, status=400)
        try:
            video_obj = get_object_or_404(VideoAnalysis, id=video_id)
        except (VideoAnalysis.DoesNotExist, ValueError):
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
        return JsonResponse({'result': 'success', 'ball_speed': video_obj.ball_speed})
    return JsonResponse({'result': 'fail', 'reason': 'GET only'}, status=405)

@csrf_exempt
def release_angle_height_api(request, video_id):
    if request.method == 'GET':
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No id provided'}, status=400)
        try:
            video_obj = get_object_or_404(VideoAnalysis, id=video_id)
        except (VideoAnalysis.DoesNotExist, ValueError):
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
        return JsonResponse({'result': 'success', 'release_angle_height': video_obj.release_angle_height})
    return JsonResponse({'result': 'fail', 'reason': 'GET only'}, status=405)

@csrf_exempt
def dtw_similarity_api(request):
    if request.method == 'GET':
        try:
            data = json.loads(request.body)
            if not data.get('reference_id') or not data.get('test_id'):
                return JsonResponse({'result': 'fail', 'reason': 'reference_id와 test_id는 필수 정보입니다.'}, status=400)
            reference_video = data.get('reference_id')
            test_id = data.get('test_id')
            dtw_obj = get_object_or_404(DTWAnalysis, reference_video = reference_video, test_video = test_id)
        except (DTWAnalysis.DoesNotExist, ValueError):
            # DTW 분석 결과를 찾이 못하였을 때 다시 POST 요청으로 바꿀 수 있게 함.
            return JsonResponse({'result': 'POST required', 'reason': 'DTW 분석 결과를 찾을 수 없습니다.'}, status=404)
        return JsonResponse({
            'result': 'success',
            'phase_scores': [float(s) for s in dtw_obj.phase_scores],
            'phase_distances': [float(d) for d in dtw_obj.phase_distances],
            'overall_score': float(dtw_obj.overall_score),
            'worst_phase': int(dtw_obj.worst_phase) if dtw_obj.worst_phase is not None else None,
        })

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
            dtw_obj = DTWAnalysis.objects.create(
                reference_video=ref_obj.id,
                test_video=test_obj.id,
                phase_scores=phase_scores,
                phase_distances=phase_distances,
                overall_score=overall_score,
                worst_phase=worst_idx
            )
            dtw_obj.save()
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
    return JsonResponse({'result': 'fail', 'reason': 'GET or POST only'}, status=405)

@csrf_exempt
def skeleton_coords_api(request, video_id):
    if request.method == 'GET':
        if not video_id:
            return JsonResponse({'result': 'fail', 'reason': 'No id provided'}, status=400)
        try:
            video_obj = get_object_or_404(VideoAnalysis, id=video_id)
            if video_obj.skeleton_coords == None:
                return JsonResponse({'result': 'fail', 'reason': 'Skeleton coordinates not found'}, status=404)
        except (VideoAnalysis.DoesNotExist, ValueError):
            return JsonResponse({'result': 'fail', 'reason': 'Video not found'}, status=404)
        return JsonResponse({
            'result': 'success',
            'skeleton_video_url': video_obj.skeleton_video.url if video_obj.skeleton_video else None,
            'skeleton_coords': video_obj.skeleton_coords
        })
    return JsonResponse({'result': 'fail', 'reason': 'GET only'}, status=405)

@csrf_exempt
def players_list_api(request):
    if request.method == 'GET':
        players = Player.objects.all().order_by('name')
        team_name = PlayerSeason.objects.filter(year=datetime.now().year).values_list('team','player_id')

        players_data = [
            {
                'id': str(player.id),
                'name': player.name,
                'playerImg': player.playerImg.url if player.playerImg else None,
                'birth_date': player.birth_date.isoformat() if player.birth_date else None,
                'height': player.height,
                'weight': player.weight,
                'throwing_hand': player.throwing_hand,
                'batting_hand': player.batting_hand,
                'video_count': player.videos.count(),
                'team_name': team_name.filter(player_id=player.id).first()[0] if team_name.filter(player_id=player.id).first() else None
            }
            for player in players
        ]
        return JsonResponse(players_data, safe=False)
    return JsonResponse({'result': 'fail', 'reason': 'GET method only'}, status=405)

@csrf_exempt
def player_detail_api(request, player_id):
    if request.method == 'GET':
        player = get_object_or_404(Player, id=player_id)
        team_name = PlayerSeason.objects.filter(player_id=player_id, year=datetime.now().year).values_list('team', flat=True).first()
        player_data = {
            'id': str(player.id),
            'name': player.name,
            'playerStandImg': player.playerStandImg.url if player.playerStandImg else None,
            'birth_date': player.birth_date.isoformat() if player.birth_date else None,
            'height': player.height,
            'weight': player.weight,
            'throwing_hand': player.throwing_hand,
            'batting_hand': player.batting_hand,
            'video_count': player.videos.count(),
            'team_name': team_name if team_name else None,
            'join_year': player.join_year,
            'career_stats': player.career_stats,
            'optimum_form': player.optimumForm.id if player.optimumForm else ''
        }
        return JsonResponse(player_data)
    return JsonResponse({'result': 'fail', 'reason': 'GET method only'}, status=405)

@csrf_exempt
def player_create_api(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        playerImg = request.FILES.get('pimage')
        playerStandingImg = request.FILES.get('simage')

        birth_date_str = request.POST.get('birth_date')
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
            height=request.POST.get('height'), weight=request.POST.get('weight'),
            throwing_hand=request.POST.get('throwing_hand'), batting_hand=request.POST.get('batting_hand'),
            playerImg=playerImg, playerStandImg=playerStandingImg, join_year=request.POST.get("join_year"),
            career_stats=request.POST.get("career_stats"),
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

@csrf_exempt
def register_optimum_api(request, player_id):
    if request.method == 'POST':
        data = json.loads(request.body)
        player = get_object_or_404(Player, id=player_id)
        video = get_object_or_404(VideoAnalysis, id=data.get('video_id'))
        player.optimumForm = video
        player.save()
        return JsonResponse({'result': 'success'})
    return JsonResponse({'result': 'fail', 'reason': 'Method not allowed'}, status=405)
