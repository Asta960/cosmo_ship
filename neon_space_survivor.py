#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neon Space Survivor - Бесконечный раннер/шутер с ИИ агентом
Стиль: Неоновый киберпанк
Алгоритм: DQN (Deep Q-Network) на PyTorch

Управление:
- В режиме просмотра: нажмите 'F' для переключения Fast Mode, 'S' для сохранения модели, 'L' для загрузки
- ИИ управляет кораблем автоматически
"""

import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import math

# ==================== КОНСТАНТЫ ====================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Цвета (неоновый стиль)
COLOR_BG = (5, 5, 10)  # Почти черный
COLOR_PLAYER = (0, 255, 255)  # Cyan
COLOR_ENEMY = (255, 0, 255)  # Magenta
COLOR_BULLET = (0, 255, 100)  # Lime
COLOR_TEXT = (255, 255, 255)
COLOR_TRAIL_PLAYER = (0, 255, 255, 50)
COLOR_TRAIL_ENEMY = (255, 0, 255, 50)
COLOR_PARTICLE = (255, 200, 100)

# Настройки игры
PLAYER_SPEED = 7
PLAYER_SIZE = 30
ENEMY_BASE_SPEED = 3
ENEMY_SPAWN_RATE = 60  # Кадры между спавном
BULLET_SPEED = 10

# Настройки ИИ
STATE_SIZE = 4  # [X игрока, Расстояние до врага, Скорость врага, Счет]
ACTION_SIZE = 3  # [Влево, Стоять, Вправо]
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 100

# ==================== НЕЙРОСЕТЬ (DQN) ====================
class DQN(nn.Module):
    """Простая нейросеть для DQN агента"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ==================== АГЕНТ ИИ ====================
class AIAgent:
    """DQN агент для обучения с подкреплением"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Вероятность случайного действия (exploration)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.update_target_model()
        
    def update_target_model(self):
        """Обновление целевой сети"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Сохранение опыта в память"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """Выбор действия на основе состояния"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()
    
    def get_action_probabilities(self, state):
        """Получение вероятностей действий для визуализации"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            probs = torch.softmax(q_values, dim=1).cpu().numpy()[0]
        return probs
    
    def replay(self):
        """Обучение на батче из памяти"""
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Текущие Q-значения
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Следующие Q-значения (из целевой сети)
        next_q = self.target_model(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Потеря и оптимизация
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Уменьшение exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, filename="neon_survivor_model.pth"):
        """Сохранение весов модели"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon,
        }, filename)
        print(f"Модель сохранена в {filename}")
        
    def load_model(self, filename="neon_survivor_model.pth"):
        """Загрузка весов модели"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 0.01)
            self.update_target_model()
            print(f"Модель загружена из {filename}")
            return True
        return False

# ==================== ЧАСТИЦЫ ====================
class Particle:
    """Класс частицы для эффектов взрыва"""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-5, 5)
        self.color = color
        self.life = random.randint(20, 40)
        self.max_life = self.life
        self.size = random.randint(2, 5)
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.vy += 0.2  # Гравитация
        
    def draw(self, screen):
        alpha = int(255 * (self.life / self.max_life))
        size = max(1, int(self.size * (self.life / self.max_life)))
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), size)

# ==================== ИГРОВОЕ ОКРУЖЕНИЕ ====================
class GameEnv:
    """Основной класс игры с логикой и отрисовкой"""
    def __init__(self, render=True):
        self.render_mode = render
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        
        # Статистика (инициализируем ДО reset())
        self.generation = 0
        self.total_frames = 0
        
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Neon Space Survivor - AI Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
            
        self.reset()
        
        # Для эффектов
        self.trails = []  # Шлейфы объектов
        self.particles = []  # Частицы взрывов
        self.shake_time = 0  # Время тряски экрана
        
    def reset(self):
        """Сброс игры в начальное состояние"""
        self.player_x = SCREEN_WIDTH // 2
        self.player_y = SCREEN_HEIGHT - 80
        self.enemies = []  # Список врагов: [x, y, speed]
        self.bullets = []  # Список пуль: [x, y]
        self.score = 0
        self.game_over = False
        self.frame_count = 0
        self.spawn_timer = 0
        self.difficulty_multiplier = 1.0
        self.last_enemy_distance = SCREEN_HEIGHT
        
        # Очистка эффектов
        self.trails = []
        self.particles = []
        self.shake_time = 0
        
        self.generation += 1
        
        return self._get_state()
    
    def _get_state(self):
        """Получение текущего состояния для ИИ"""
        # Нормализованные значения
        player_x_norm = self.player_x / SCREEN_WIDTH
        
        # Найти ближайшего врага
        nearest_distance = SCREEN_HEIGHT
        nearest_speed = ENEMY_BASE_SPEED
        
        for enemy in self.enemies:
            dist = enemy[1] - self.player_y
            if 0 < dist < nearest_distance:
                nearest_distance = dist
                nearest_speed = enemy[2]
        
        # Если врагов нет, использовать дефолтные значения
        if nearest_distance == SCREEN_HEIGHT:
            nearest_distance = SCREEN_HEIGHT
            nearest_speed = ENEMY_BASE_SPEED
            
        distance_norm = nearest_distance / SCREEN_HEIGHT
        speed_norm = nearest_speed / (ENEMY_BASE_SPEED * 3)  # Максимальная скорость ~3x базовой
        score_norm = min(self.score / 1000, 1.0)  # Нормализация счета
        
        return np.array([player_x_norm, distance_norm, speed_norm, score_norm], dtype=np.float32)
    
    def step(self, action):
        """
        Выполнение шага игры
        action: 0=Влево, 1=Стоять, 2=Вправо
        Возвращает: (state, reward, done, info)
        """
        reward = 0
        done = False
        info = {}
        
        # Обработка действия игрока
        if action == 0:  # Влево
            self.player_x = max(PLAYER_SIZE, self.player_x - PLAYER_SPEED)
        elif action == 2:  # Вправо
            self.player_x = min(SCREEN_WIDTH - PLAYER_SIZE, self.player_x + PLAYER_SPEED)
        # action == 1: Стоять - ничего не делаем
        
        # Автоматическая стрельба каждые 15 кадров
        if self.frame_count % 15 == 0:
            self.bullets.append([self.player_x, self.player_y - 20])
        
        # Обновление пуль
        for bullet in self.bullets[:]:
            bullet[1] -= BULLET_SPEED
            if bullet[1] < 0:
                self.bullets.remove(bullet)
                
        # Спавн врагов
        self.spawn_timer += 1
        spawn_rate = max(20, int(ENEMY_SPAWN_RATE / self.difficulty_multiplier))
        
        if self.spawn_timer >= spawn_rate:
            self.spawn_timer = 0
            enemy_x = random.randint(50, SCREEN_WIDTH - 50)
            enemy_speed = ENEMY_BASE_SPEED * self.difficulty_multiplier * random.uniform(0.8, 1.2)
            self.enemies.append([enemy_x, -30, enemy_speed])
            
        # Обновление врагов
        for enemy in self.enemies[:]:
            enemy[1] += enemy[2]
            
            # Проверка столкновения с пулей
            for bullet in self.bullets[:]:
                if abs(bullet[0] - enemy[0]) < 25 and abs(bullet[1] - enemy[1]) < 25:
                    if enemy in self.enemies:
                        self.enemies.remove(enemy)
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    self.score += 10
                    reward += 10
                    self._create_explosion(enemy[0], enemy[1], COLOR_ENEMY)
                    break
                    
            # Проверка выхода за экран
            if enemy[1] > SCREEN_HEIGHT + 50:
                self.enemies.remove(enemy)
                
        # Увеличение сложности каждые 10 секунд (600 кадров)
        if self.frame_count % 600 == 0 and self.frame_count > 0:
            self.difficulty_multiplier *= 1.1
            
        # Проверка столкновения с игроком
        player_rect = pygame.Rect(self.player_x - PLAYER_SIZE//2, self.player_y - PLAYER_SIZE//2, 
                                   PLAYER_SIZE, PLAYER_SIZE)
        
        for enemy in self.enemies[:]:
            enemy_rect = pygame.Rect(enemy[0] - 20, enemy[1] - 20, 40, 40)
            if player_rect.colliderect(enemy_rect):
                done = True
                reward = -100
                self._create_explosion(self.player_x, self.player_y, COLOR_PLAYER, count=50)
                self.shake_time = 20
                break
                
        # Награда за выживание
        if not done:
            reward += 1
            
        # Сохранение шлейфов
        if self.render_mode:
            self.trails.append(['player', self.player_x, self.player_y])
            for enemy in self.enemies:
                self.trails.append(['enemy', enemy[0], enemy[1]])
            if len(self.trails) > 10:
                self.trails.pop(0)
        
        # Обновление частиц
        for particle in self.particles[:]:
            particle.update()
            if particle.life <= 0:
                self.particles.remove(particle)
                
        self.frame_count += 1
        self.total_frames += 1
        self.last_enemy_distance = min([e[1] - self.player_y for e in self.enemies], default=SCREEN_HEIGHT)
        
        next_state = self._get_state()
        
        if self.render_mode:
            self._render_frame()
            self._handle_events()
            
        return next_state, reward, done, info
    
    def _create_explosion(self, x, y, color, count=20):
        """Создание взрыва частиц"""
        for _ in range(count):
            self.particles.append(Particle(x, y, color))
            
    def _render_frame(self):
        """Отрисовка кадра"""
        # Тряска экрана
        shake_offset = (0, 0)
        if self.shake_time > 0:
            shake_offset = (random.randint(-5, 5), random.randint(-5, 5))
            self.shake_time -= 1
            
        # Очистка экрана с полупрозрачностью для эффекта шлейфа
        surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        surface.fill(COLOR_BG)
        surface.set_alpha(180)
        self.screen.blit(surface, (0, 0))
        
        # Отрисовка шлейфов
        for trail in self.trails:
            if trail[0] == 'player':
                s = pygame.Surface((PLAYER_SIZE, PLAYER_SIZE), pygame.SRCALPHA)
                pygame.draw.polygon(s, (*COLOR_TRAIL_PLAYER[:3], 30), 
                                   [(PLAYER_SIZE//2, 0), (0, PLAYER_SIZE), (PLAYER_SIZE, PLAYER_SIZE)])
                self.screen.blit(s, (trail[1] - PLAYER_SIZE//2 + shake_offset[0], 
                                     trail[2] - PLAYER_SIZE//2 + shake_offset[1]))
            elif trail[0] == 'enemy':
                s = pygame.Surface((40, 40), pygame.SRCALPHA)
                pygame.draw.circle(s, (*COLOR_TRAIL_ENEMY[:3], 30), (20, 20), 20)
                self.screen.blit(s, (trail[1] - 20 + shake_offset[0], trail[2] - 20 + shake_offset[1]))
        
        # Отрисовка частиц
        for particle in self.particles:
            particle.draw(self.screen)
            
        # Отрисовка пуль
        for bullet in self.bullets:
            pygame.draw.circle(self.screen, COLOR_BULLET, 
                              (bullet[0] + shake_offset[0], bullet[1] + shake_offset[1]), 5)
            
        # Отрисовка врагов
        for enemy in self.enemies:
            pygame.draw.circle(self.screen, COLOR_ENEMY, 
                              (enemy[0] + shake_offset[0], enemy[1] + shake_offset[1]), 20)
            # Неоновое свечение
            pygame.draw.circle(self.screen, (200, 100, 200), 
                              (enemy[0] + shake_offset[0], enemy[1] + shake_offset[1]), 25, 2)
            
        # Отрисовка игрока (треугольник)
        player_points = [
            (self.player_x + shake_offset[0], self.player_y - PLAYER_SIZE//2 + shake_offset[1]),
            (self.player_x - PLAYER_SIZE//2 + shake_offset[0], self.player_y + PLAYER_SIZE//2 + shake_offset[1]),
            (self.player_x + PLAYER_SIZE//2 + shake_offset[0], self.player_y + PLAYER_SIZE//2 + shake_offset[1])
        ]
        pygame.draw.polygon(self.screen, COLOR_PLAYER, player_points)
        # Неоновое свечение
        pygame.draw.polygon(self.screen, (100, 200, 200), player_points, 2)
        
        # Отрисовка интерфейса
        self._draw_ui()
        
        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(FPS)
            
    def _draw_ui(self):
        """Отрисовка пользовательского интерфейса"""
        # Счет
        score_text = self.font.render(f"Счет: {self.score}", True, COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Поколение
        gen_text = self.font.render(f"Поколение: {self.generation}", True, COLOR_TEXT)
        self.screen.blit(gen_text, (10, 50))
        
        # Сложность
        diff_text = self.small_font.render(f"Сложность: {self.difficulty_multiplier:.2f}x", True, COLOR_TEXT)
        self.screen.blit(diff_text, (10, 90))
        
        # FPS
        fps_text = self.small_font.render(f"FPS: {int(self.clock.get_fps())}", True, COLOR_TEXT)
        self.screen.blit(fps_text, (SCREEN_WIDTH - 100, 10))
        
        # Мысли ИИ (вероятности действий)
        state = self._get_state()
        probs = agent.get_action_probabilities(state) if 'agent' in globals() else [0.33, 0.33, 0.34]
        
        actions = ["Влево", "Стоять", "Вправо"]
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
        
        ui_x = SCREEN_WIDTH - 200
        ui_y = SCREEN_HEIGHT - 120
        
        bg_surface = pygame.Surface((190, 110), pygame.SRCALPHA)
        bg_surface.fill((20, 20, 30, 200))
        self.screen.blit(bg_surface, (ui_x - 5, ui_y - 5))
        
        title_text = self.small_font.render("Мысли ИИ:", True, COLOR_TEXT)
        self.screen.blit(title_text, (ui_x, ui_y))
        
        for i, (action, prob, color) in enumerate(zip(actions, probs, colors)):
            bar_width = int(150 * prob)
            bar_height = 20
            bar_y = ui_y + 25 + i * 28
            
            # Фон полоски
            pygame.draw.rect(self.screen, (50, 50, 50), (ui_x, bar_y, 150, bar_height))
            # Заполнение
            pygame.draw.rect(self.screen, color, (ui_x, bar_y, bar_width, bar_height))
            # Текст
            text = self.small_font.render(f"{action}: {prob*100:.1f}%", True, COLOR_TEXT)
            self.screen.blit(text, (ui_x + 5, bar_y + 2))
            
    def _handle_events(self):
        """Обработка событий"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    return 'toggle_fast'
                elif event.key == pygame.K_s:
                    agent.save_model()
                elif event.key == pygame.K_l:
                    agent.load_model()
                    
        return True

# ==================== ГЛАВНЫЙ ЦИКЛ ====================
def main():
    global agent
    
    print("=" * 50)
    print("NEON SPACE SURVIVOR - AI AGENT")
    print("=" * 50)
    print("\nУправление:")
    print("  F - Переключение Fast/View режима")
    print("  S - Сохранить модель")
    print("  L - Загрузить модель")
    print("  ESC - Выход")
    print("\nЗапуск в режиме просмотра...")
    print("=" * 50)
    
    # Создание агента
    agent = AIAgent(STATE_SIZE, ACTION_SIZE)
    
    # Попытка загрузить существующую модель
    if not agent.load_model():
        print("Новая модель создана.")
    
    # Создание окружения (начинаем в режиме просмотра)
    env = GameEnv(render=True)
    
    running = True
    episode = 0
    total_reward = 0
    
    # Статистика для отображения
    avg_reward = 0
    rewards_history = []
    
    while running:
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done and running:
            # Выбор действия
            action = agent.act(state, training=True)
            
            # Шаг игры
            next_state, reward, done, info = env.step(action)
            
            # Сохранение опыта
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_reward += reward
            
            # Обучение
            agent.replay()
            
            # Обновление целевой сети
            if env.total_frames % TARGET_UPDATE == 0:
                agent.update_target_model()
                
        # Конец эпизода
        episode += 1
        rewards_history.append(episode_reward)
        
        # Скользящее среднее
        if len(rewards_history) > 10:
            rewards_history.pop(0)
        avg_reward = sum(rewards_history) / len(rewards_history)
        
        # Вывод статистики каждые 10 эпизодов
        if episode % 10 == 0:
            print(f"Эпизод {episode}, Средняя награда: {avg_reward:.1f}, Epsilon: {agent.epsilon:.3f}")
            
        # Сохранение модели каждые 100 эпизодов
        if episode % 100 == 0:
            agent.save_model()
            
    pygame.quit()
    print(f"\nИгра завершена. Всего эпизодов: {episode}")
    print(f"Средняя награда: {avg_reward:.1f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nИгра прервана пользователем.")
        pygame.quit()
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
