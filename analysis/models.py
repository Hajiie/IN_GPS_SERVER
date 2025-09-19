import os
import uuid
from django.db import models
from django.utils import timezone
from django.db.models import JSONField
from django.utils.text import slugify

def get_video_upload_path(instance, filename):
    """Dynamically sets the upload path and filename."""
    ext = filename.split('.')[-1]
    if instance.video_name and instance.video_name.strip():
        base_name = slugify(instance.video_name, allow_unicode=True)
    else:
        base_name = timezone.now().strftime('%Y-%m-%d_%H-%M-%S')
    new_filename = f"{base_name}.{ext}"
    return os.path.join('media/videos', new_filename)

class Player(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    birth_date = models.DateField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)  # cm
    weight = models.IntegerField(null=True, blank=True)  # kg
    throwing_hand = models.CharField(max_length=10, choices=[
        ('R', '우투'),
        ('L', '좌투'),
    ], null=True, blank=True)
    batting_hand = models.CharField(max_length=10, choices=[
        ('R', '우타'),
        ('L', '좌타'),
        ('S', '양타'),
    ], null=True, blank=True)
    
    def __str__(self):
        return self.name

class PlayerSeason(models.Model):
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name='seasons')
    year = models.IntegerField()
    team = models.CharField(max_length=50)
    
    class Meta:
        unique_together = ['player', 'year', 'team']
        ordering = ['-year']
    
    def __str__(self):
        return f"{self.player.name} - {self.year} {self.team}"

class PlayerStats(models.Model):
    player_season = models.OneToOneField(PlayerSeason, on_delete=models.CASCADE, related_name='stats')
    
    # 기본 스탯
    era = models.DecimalField(max_digits=4, decimal_places=2, null=True, blank=True)
    games = models.IntegerField(default=0)
    wins = models.IntegerField(default=0)
    losses = models.IntegerField(default=0)
    saves = models.IntegerField(default=0)
    holds = models.IntegerField(default=0)
    win_rate = models.DecimalField(max_digits=5, decimal_places=3, null=True, blank=True)
    
    # 투구 관련
    innings_pitched = models.DecimalField(max_digits=5, decimal_places=1, default=0)  # IP
    hits_allowed = models.IntegerField(default=0)  # H
    home_runs_allowed = models.IntegerField(default=0)  # HR
    walks = models.IntegerField(default=0)  # BB
    hit_by_pitch = models.IntegerField(default=0)  # 사구
    strikeouts = models.IntegerField(default=0)  # 삼진
    
    # 실점 관련
    runs_allowed = models.IntegerField(default=0)  # 실점
    earned_runs = models.IntegerField(default=0)  # 자책점
    
    def __str__(self):
        return f"{self.player_season} - ERA: {self.era}"
    
    def save(self, *args, **kwargs):
        # 승률 자동 계산
        if self.wins + self.losses > 0:
            self.win_rate = self.wins / (self.wins + self.losses)
        super().save(*args, **kwargs)

class VideoAnalysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    player = models.ForeignKey(Player, on_delete=models.CASCADE, null=True, blank=True, related_name='videos')
    video_name = models.CharField(max_length=255, null=True, blank=True, unique=True)
    video_file = models.FileField(upload_to=get_video_upload_path, max_length=255)
    upload_time = models.DateTimeField(default=timezone.now)
    frame_list = JSONField(null=True, blank=True)
    fixed_frame = models.IntegerField(null=True, blank=True)
    release_frame = models.IntegerField(null=True, blank=True)
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    release_frame_knee = JSONField(null=True, blank=True)
    release_frame_ankle = JSONField(null=True, blank=True)
    ball_speed = JSONField(null=True, blank=True)  # trajectory, speed_kph
    release_angle_height = JSONField(null=True, blank=True)  # angles, hand_height
    skeleton_coords = JSONField(null=True, blank=True)  # release_frame 기준 랜드마크 좌표

    def __str__(self):
        return self.video_name or os.path.basename(self.video_file.name)

    def delete(self, *args, **kwargs):
        # First, delete the video file from storage
        self.video_file.delete(save=False)
        # Now, call the superclass method to delete the database record
        super().delete(*args, **kwargs)
