import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
import os

# ==========================================
# КОНФИГУРАЦИЯ ИГРЫ
# ==========================================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Цвета (Neon Cyberpunk)
COLOR_BG = (5, 5, 10)  # Почти черный
COLOR_PLAYER = (0, 255, 255)  # Cyan
COLOR_ENEMY = (255, 0, 255)   # Magenta
COLOR_BULLET = (50, 255, 50)  # Lime
COLOR_TEXT = (255, 255, 255)
COLOR_UI_BAR = (50, 50, 50)

# Параметры игры
PLAYER_SPEED = 7
BASE_ENEMY_SPEED = 3
SPAWN_RATE = 60  # Кадры между спавном
MAX_LIVES = 3

# ==========================================
# НЕЙРОСЕТЬ (DQN AGENT)
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)

class AIAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename="model.pth"):
        torch.save(self.model.state_dict(), filename)
        print(f"Модель сохранена в {filename}")

    def load(self, filename="model.pth"):
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename, map_location=self.device))
            self.update_target_model()
            print(f"Модель загружена из {filename}")
            return True
        return False

# ==========================================
# ИГРОВАЯ СРЕДА (ENV)
# ==========================================
class GameEnv:
    def __init__(self, render=True):
        self.render_mode = render
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        
        # Состояние игры
        self.player_x = SCREEN_WIDTH // 2
        self.lives = MAX_LIVES
        self.score = 0
        self.generation = 0
        self.frame_count = 0
        self.difficulty_multiplier = 1.0
        
        # Объекты
        self.enemies = []
        self.particles = []
        self.trails = []  # Шлейфы
        
        # Таймеры
        self.spawn_timer = 0
        self.difficulty_timer = 0
        
        # Для отрисовки мыслей ИИ
        self.last_q_values = [0, 0, 0]
        
        # Флаг тряски экрана
        self.shake_offset = (0, 0)
        self.shake_duration = 0

        if self.render_mode:
            self._init_pygame()

    def _init_pygame(self):
        if not pygame.get_init():
            pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Neon Space Survivor - AI Agent")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 18)

    def reset(self):
        """Перезапуск игры (смерть или новый эпизод)"""
        self.player_x = SCREEN_WIDTH // 2
        self.lives = MAX_LIVES
        self.score = 0
        self.enemies = []
        self.particles = []
        self.trails = []
        self.spawn_timer = 0
        self.difficulty_multiplier = 1.0
        self.difficulty_timer = 0
        self.frame_count = 0
        self.generation += 1
        self.shake_duration = 0
        self.shake_offset = (0, 0)
        
        # Очистка экрана при рестарте в режиме рендера
        if self.render_mode and self.screen:
            self.screen.fill(COLOR_BG)
            pygame.display.flip()
            
        return self._get_state()

    def _get_state(self):
        """Формирование вектора состояния для ИИ"""
        # Нормализация данных
        norm_player_x = self.player_x / SCREEN_WIDTH
        
        # Находим 3 ближайших врагов (сортируем по расстоянию до игрока)
        player_y = SCREEN_HEIGHT - 50
        nearby_enemies = []
        
        for e in self.enemies:
            dist = math.hypot(e['x'] - self.player_x, e['y'] - player_y)
            nearby_enemies.append({
                'dist': dist,
                'x': e['x'],
                'y': e['y'],
                'speed': e['speed']
            })
        
        # Сортируем и берем топ-3
        nearby_enemies.sort(key=lambda k: k['dist'])
        top_3 = nearby_enemies[:3]
        
        state_data = [norm_player_x, self.lives / MAX_LIVES]
        
        # Добавляем данные о 3 врагах (если меньше 3, заполняем нулями/максимумом)
        for i in range(3):
            if i < len(top_3):
                e = top_3[i]
                state_data.append(e['x'] / SCREEN_WIDTH)
                state_data.append(e['y'] / SCREEN_HEIGHT)
                state_data.append(e['speed'] / (BASE_ENEMY_SPEED * 2))
                state_data.append((e['x'] - self.player_x) / SCREEN_WIDTH)
            else:
                state_data.extend([0, 0, 0, 0])
        
        return np.array(state_data, dtype=np.float32)

    def step(self, action):
        """
        Выполнение шага игры.
        Action: 0 - Влево, 1 - Стоять, 2 - Вправо
        """
        reward = 0
        done = False
        info = {}

        # 1. Движение игрока
        if action == 0: # Влево
            self.player_x -= PLAYER_SPEED
        elif action == 2: # Вправо
            self.player_x += PLAYER_SPEED
        # Ограничение границами
        self.player_x = max(20, min(SCREEN_WIDTH - 20, self.player_x))

        # 2. Спавн врагов
        current_spawn_rate = int(SPAWN_RATE / self.difficulty_multiplier)
        if self.spawn_timer >= current_spawn_rate:
            self._spawn_enemy()
            self.spawn_timer = 0
        else:
            self.spawn_timer += 1

        # 3. Обновление врагов и коллизии
        enemies_to_remove = []
        
        # Хитбокс игрока (треугольник примерно)
        player_rect = pygame.Rect(self.player_x - 15, SCREEN_HEIGHT - 60, 30, 30)

        for i, enemy in enumerate(self.enemies):
            enemy['y'] += enemy['speed']
            
            # Проверка столкновения
            enemy_rect = pygame.Rect(enemy['x'] - 15, enemy['y'] - 15, 30, 30)
            if player_rect.colliderect(enemy_rect):
                # Столкновение!
                self.lives -= 1
                reward -= 50  # Большой штраф за удар
                self._create_explosion(enemy['x'], enemy['y'], COLOR_ENEMY)
                self._create_explosion(self.player_x, SCREEN_HEIGHT - 50, COLOR_PLAYER)
                self._trigger_screen_shake()
                enemies_to_remove.append(i)
                continue

            # Проверка выхода за экран (враг пролетел мимо)
            if enemy['y'] > SCREEN_HEIGHT + 50:
                self.lives -= 1
                reward -= 10  # Штраф за пропуск врага ("ему будет плохо")
                enemies_to_remove.append(i)
                continue

        # Удаляем обработанных врагов (с конца списка, чтобы не сбить индексы)
        for i in reversed(enemies_to_remove):
            if i < len(self.enemies):
                del self.enemies[i]

        # 4. Награды
        reward += 0.1  # Маленькая награда за выживание каждый кадр
        
        # Проверка конца игры
        if self.lives <= 0:
            done = True
            reward -= 100 # Финальный штраф за смерть
            # generation увеличивается только при полном геймовере (в reset)

        # 5. Прогрессия сложности (каждые 10 секунд ~ 600 кадров)
        self.frame_count += 1
        if self.frame_count % 600 == 0:
            self.difficulty_multiplier *= 1.1
            # Увеличиваем скорость существующих врагов
            for e in self.enemies:
                e['speed'] *= 1.1

        # 6. Отрисовка (если нужно)
        if self.render_mode:
            self._render(action)
            # Обработка событий внутри шага для быстрого выхода или переключения
            event_signal = self._handle_events()
            if event_signal == 'quit':
                done = True
            elif event_signal == 'toggle_fast':
                info['toggle_fast'] = True

        next_state = self._get_state()
        return next_state, reward, done, info

    def _spawn_enemy(self):
        x = random.randint(40, SCREEN_WIDTH - 40)
        speed = BASE_ENEMY_SPEED * self.difficulty_multiplier * random.uniform(0.8, 1.2)
        self.enemies.append({
            'x': x,
            'y': -30,
            'speed': speed,
            'color': COLOR_ENEMY
        })

    def _create_explosion(self, x, y, color):
        for _ in range(20):
            self.particles.append({
                'x': x,
                'y': y,
                'vx': random.uniform(-5, 5),
                'vy': random.uniform(-5, 5),
                'life': 1.0,
                'color': color
            })

    def _trigger_screen_shake(self):
        self.shake_duration = 15  # 15 кадров тряски

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'quit'
                if event.key == pygame.K_f:
                    return 'toggle_fast'
                if event.key == pygame.K_s:
                    agent.save()
                if event.key == pygame.K_l:
                    agent.load()
        return None

    def _render(self, action):
        # Обработка тряски экрана
        shake_x, shake_y = 0, 0
        if self.shake_duration > 0:
            shake_x = random.randint(-5, 5)
            shake_y = random.randint(-5, 5)
            self.shake_duration -= 1

        # Очистка с эффектом шлейфа (полупрозрачный прямоугольник)
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        s.set_alpha(40)
        s.fill(COLOR_BG)
        self.screen.blit(s, (0, 0))

        # Отрисовка игрока (Треугольник) с учетом тряски
        pts = [
            (self.player_x + shake_x, SCREEN_HEIGHT - 80 + shake_y),
            (self.player_x - 20 + shake_x, SCREEN_HEIGHT - 30 + shake_y),
            (self.player_x + 20 + shake_x, SCREEN_HEIGHT - 30 + shake_y)
        ]
        pygame.draw.polygon(self.screen, COLOR_PLAYER, pts)
        pygame.draw.polygon(self.screen, (100, 255, 255), pts, 2)

        # Отрисовка врагов
        for e in self.enemies:
            ex, ey = int(e['x'] + shake_x), int(e['y'] + shake_y)
            pygame.draw.circle(self.screen, e['color'], (ex, ey), 15)
            pygame.draw.circle(self.screen, (255, 100, 255), (ex, ey), 15, 2)

        # Отрисовка частиц
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 0.05
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                radius = max(1, int(5 * p['life']))
                c = tuple(int(val * p['life']) for val in p['color'])
                px, py = int(p['x'] + shake_x), int(p['y'] + shake_y)
                pygame.draw.circle(self.screen, c, (px, py), radius)

        # Отрисовка интерфейса (без тряски)
        score_text = self.font.render(f"Счет: {int(self.score)}", True, COLOR_TEXT)
        lives_color = (255, 50, 50) if self.lives == 1 else COLOR_TEXT
        lives_text = self.font.render(f"Жизни: {self.lives}", True, lives_color)
        gen_text = self.small_font.render(f"Поколение: {self.generation}", True, COLOR_TEXT)
        diff_text = self.small_font.render(f"Сложность: {self.difficulty_multiplier:.1f}x", True, COLOR_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (10, 45))
        self.screen.blit(gen_text, (10, SCREEN_HEIGHT - 30))
        self.screen.blit(diff_text, (10, SCREEN_HEIGHT - 55))

        # Визуализация мыслей ИИ (Q-values)
        self._draw_ai_thoughts(action)

        pygame.display.flip()
        if self.clock:
            self.clock.tick(FPS)

    def _draw_ai_thoughts(self, action):
        labels = ["Влево", "Стоять", "Вправо"]
        colors = [(0, 100, 255), (200, 200, 200), (255, 100, 0)]
        
        start_x = SCREEN_WIDTH - 150
        start_y = 10
        
        for i, label in enumerate(labels):
            y = start_y + i * 25
            pygame.draw.rect(self.screen, COLOR_UI_BAR, (start_x, y, 140, 20))
            
            width = 0
            if i == action:
                width = 140
                color = (0, 255, 0)
            else:
                width = 20
                color = (100, 100, 100)
                
            pygame.draw.rect(self.screen, color, (start_x, y, width, 20))
            
            text = self.small_font.render(f"{label}", True, COLOR_TEXT)
            self.screen.blit(text, (start_x + 5, y + 2))

# ==========================================
# ГЛАВНЫЙ ЦИКЛ
# ==========================================
agent = None  # Глобальная переменная для доступа из обработчиков событий

def main():
    global agent
    
    print("==================================================")
    print("NEON SPACE SURVIVOR - AI AGENT")
    print("==================================================")
    print("\nУправление:")
    print("  F - Переключение Fast/View режима")
    print("  S - Сохранить модель")
    print("  L - Загрузить модель")
    print("  ESC - Выход")
    print("==================================================\n")

    # Инициализация среды (начинаем в режиме просмотра)
    env = GameEnv(render=True)
    
    # Параметры состояния: 1 (player_x) + 1 (lives) + 3 врага * 4 параметра = 14
    state_size = 14 
    action_size = 3 # Влево, Стоять, Вправо
    
    agent = AIAgent(state_size, action_size)
    
    # Попытка загрузить последнюю модель
    if not agent.load():
        print("Новая модель создана.")
    
    batch_size = 64
    total_episodes = 0
    
    state = env.reset()
    running = True
    
    while running:
        # Выбор действия
        action = agent.act(state, training=True)
        
        # Шаг игры
        next_state, reward, done, info = env.step(action)
        
        # Запоминание опыта
        agent.remember(state, action, reward, next_state, done)
        
        # Обучение
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        state = next_state
        
        # Обработка переключения режима
        if info.get('toggle_fast'):
            env.render_mode = not env.render_mode
            if env.render_mode:
                print("Переключено в режим просмотра (View Mode)")
                env._init_pygame()
                env._render(action)
            else:
                print("Переключено в быстрый режим (Fast Mode) - графика отключена")
                if env.screen:
                    pygame.display.quit()
                    env.screen = None
                    env.clock = None
                    env.font = None
                    env.small_font = None
        
        if done:
            total_episodes += 1
            if total_episodes % 10 == 0:
                print(f"Эпизодов: {total_episodes}, Epsilon: {agent.epsilon:.2f}, Score: {env.score}")
            
            # Авто-рестарт
            state = env.reset()
            
            # Если окно закрыто (Fast Mode), проверяем возможность выхода через консоль
            # Но для простоты оставим только переключение через F в View Mode
            
    if env.screen:
        pygame.quit()
    print("Игра завершена.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        if pygame.get_init():
            pygame.quit()
