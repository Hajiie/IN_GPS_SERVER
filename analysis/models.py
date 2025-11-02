import os
import uuid
from django.db import models
from django.utils import timezone
from django.db.models import JSONField, CharField
from django.utils.text import slugify

def player_image_upload_path(instance, filename):
    ext = filename.split('.')[-1]
    new_filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('player_images', str(instance.id), new_filename)

def player_standing_image_upload_path(instance, filename):
    ext = filename.split('.')[-1]
    new_filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('player_standing_images', str(instance.id), new_filename)

def video_upload_path(instance, filename):
    player_id = str(instance.player.id) if instance.player else 'unknown_player'
    ext = filename.split('.')[-1]
    new_filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('videos', player_id, new_filename)

get_video_upload_path = video_upload_path

def thumbnail_upload_path(instance, filename):
    player_id = str(instance.player.id) if instance.player else 'unknown_player'
    ext = filename.split('.')[-1]
    new_filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('thumbnails', player_id, new_filename)

def skeleton_video_upload_path(instance, filename):
    player_id = str(instance.player.id) if instance.player else 'unknown_player'
    ext = filename.split('.')[-1]
    new_filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('skeleton_videos', player_id, new_filename)

class Player(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    playerImg = models.ImageField(upload_to=player_image_upload_path, null=True, blank=True)
    playerStandImg = models.ImageField(upload_to=player_standing_image_upload_path, null=True, blank=True)
    optimumForm = models.ForeignKey('VideoAnalysis', on_delete=models.SET_NULL, null=True, blank=True, related_name='optimum_form')
    name = models.CharField(max_length=100)
    birth_date = models.DateField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    weight = models.IntegerField(null=True, blank=True)
    join_year = models.IntegerField(null=True, blank=True)
    career_stats = CharField(max_length=100, null=True, blank=True)
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
    
    era = models.DecimalField(max_digits=4, decimal_places=2, null=True, blank=True)
    games = models.IntegerField(default=0)
    wins = models.IntegerField(default=0)
    losses = models.IntegerField(default=0)
    saves = models.IntegerField(default=0)
    holds = models.IntegerField(default=0)
    win_rate = models.DecimalField(max_digits=5, decimal_places=3, null=True, blank=True)
    
    innings_pitched = models.CharField(max_length=10, default=0)
    hits_allowed = models.IntegerField(default=0)
    home_runs_allowed = models.IntegerField(default=0)
    walks = models.IntegerField(default=0)
    hit_by_pitch = models.IntegerField(default=0)
    strikeouts = models.IntegerField(default=0)
    
    runs_allowed = models.IntegerField(default=0)
    earned_runs = models.IntegerField(default=0)
    
    def __str__(self):
        return f"{self.player_season} - ERA: {self.era}"
    
    def save(self, *args, **kwargs):
        if self.wins + self.losses > 0:
            self.win_rate = self.wins / (self.wins + self.losses)
        super().save(*args, **kwargs)

class VideoAnalysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    player = models.ForeignKey(Player, on_delete=models.CASCADE, null=True, blank=True, related_name='videos')
    video_name = models.CharField(max_length=255, null=True, blank=True, unique=True)
    video_file = models.FileField(upload_to=video_upload_path, max_length=255)
    thumbnail = models.ImageField(upload_to=thumbnail_upload_path, null=True, blank=True)
    upload_time = models.DateTimeField(default=timezone.now)
    frame_list = JSONField(null=True, blank=True)
    fixed_frame = models.IntegerField(null=True, blank=True)
    release_frame = models.IntegerField(null=True, blank=True)
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    release_frame_knee = JSONField(null=True, blank=True)
    release_frame_ankle = JSONField(null=True, blank=True)
    ball_speed = JSONField(null=True, blank=True)
    release_angle_height = JSONField(null=True, blank=True)
    skeleton_coords = JSONField(null=True, blank=True)
    skeleton_video = models.FileField(upload_to=skeleton_video_upload_path, null=True, blank=True)
    frame_metrics = JSONField(null=True, blank=True)
    arm_trajectory = JSONField(null=True, blank=True)
    arm_swing_speed = JSONField(null=True, blank=True)
    shoulder_swing_speed = JSONField(null=True, blank=True)


    def __str__(self):
        return self.video_name or os.path.basename(self.video_file.name)

    def delete(self, *args, **kwargs):
        self.video_file.delete(save=False)
        if self.thumbnail:
            self.thumbnail.delete(save=False)
        if self.skeleton_video:
            self.skeleton_video.delete(save=False)
        super().delete(*args, **kwargs)
