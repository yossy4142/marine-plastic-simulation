import asyncio
import json
import random
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

class Settings(BaseModel):
    w_trash: float
    w_avoidfish: float
    w_avoidrobot: float

class ResetConfig(BaseModel):
    num_scouts: int
    num_collectors: int
    max_steps: int
    target_score: int
    max_battery: int
    charge_time: int

# 初期状態のデフォルト値を更新 (ターゲットスコア2000, バッテリー200, 重みすべて1.0)
state = {
    "status": "standby",
    "step": 0,
    "max_steps": 120,
    "target_score": 2000,
    "max_battery": 200,
    "charge_time": 10,
    "grid_width": 20,
    "grid_height": 20,
    "robots": [],
    "trash": [],
    "fishes": [],
    "settings": {"w_trash": 1.0, "w_avoidfish": 1.0, "w_avoidrobot": 1.0},
    "shared_trash_memory": [],
    "accumulated_stress": 0.0,
    "stats": {"n_trash": 0, "n_collision": 0, "energy": 0, "total_stress": 0.0, "score": 0}
}

@app.post("/update_settings")
async def update_settings(settings: Settings):
    state["settings"]["w_trash"] = settings.w_trash
    state["settings"]["w_avoidfish"] = settings.w_avoidfish
    state["settings"]["w_avoidrobot"] = settings.w_avoidrobot
    return {"status": "success"}

@app.post("/reset")
async def reset_simulation(config: ResetConfig):
    state["status"] = "running"
    state["step"] = 0
    state["max_steps"] = config.max_steps
    state["target_score"] = config.target_score
    state["max_battery"] = config.max_battery
    state["charge_time"] = config.charge_time
    state["accumulated_stress"] = 0.0
    state["stats"] = {"n_trash": 0, "n_collision": 0, "energy": 0, "total_stress": 0.0, "score": 0}
    
    state["trash"] = [{"x": random.randint(0, 19), "y": random.randint(0, 19)} for _ in range(10)]
    state["shared_trash_memory"] = []
    
    state["fishes"] = [{"id": i, "x": random.randint(0, 19), "y": random.randint(0, 19), "stress": 0.0} for i in range(10)]
    
    robots = []
    r_id = 1
    for _ in range(config.num_scouts):
        robots.append({
            "id": r_id, "type": "scout", "x": random.randint(0, 19), "y": random.randint(0, 19),
            "energy": config.max_battery, "is_charging": False, "charge_timer": 0
        })
        r_id += 1
    for _ in range(config.num_collectors):
        robots.append({
            "id": r_id, "type": "collector", "x": random.randint(0, 19), "y": random.randint(0, 19),
            "energy": config.max_battery, "is_charging": False, "charge_timer": 0
        })
        r_id += 1
    state["robots"] = robots
    
    return {"status": "success"}

def calculate_v_next(robot, assigned_target_list, all_robots):
    v_trash = np.array([0.0, 0.0])
    v_avoidfish = np.array([0.0, 0.0])
    v_avoidrobot = np.array([0.0, 0.0])
    
    if robot["type"] == "collector" and assigned_target_list:
        target = assigned_target_list[0]
        dx = target["x"] - robot["x"]
        dy = target["y"] - robot["y"]
        norm = np.linalg.norm([dx, dy])
        if norm > 0:
            v_trash = np.array([dx, dy]) / norm

    elif robot["type"] == "scout":
        v_scout_repel = np.array([0.0, 0.0])
        for other in all_robots:
            if other["type"] == "scout" and other["id"] != robot["id"]:
                dist = max(abs(other["x"] - robot["x"]), abs(other["y"] - robot["y"]))
                if dist <= 8:
                    dx = robot["x"] - other["x"]
                    dy = robot["y"] - other["y"]
                    if dx == 0 and dy == 0:
                        dx, dy = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
                    norm = np.linalg.norm([dx, dy])
                    if norm > 0:
                        v_scout_repel += np.array([dx, dy]) / norm

        v_random = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        v_trash = v_random + v_scout_repel * 2.0
        norm = np.linalg.norm(v_trash)
        if norm > 0:
            v_trash = v_trash / norm

    if state["fishes"]:
        min_dist_fish = float('inf')
        nearest_fish = None
        for f in state["fishes"]:
            dist = max(abs(f["x"] - robot["x"]), abs(f["y"] - robot["y"]))
            if dist < min_dist_fish:
                min_dist_fish = dist
                nearest_fish = f
        if nearest_fish and min_dist_fish <= 3:
            dx = robot["x"] - nearest_fish["x"]
            dy = robot["y"] - nearest_fish["y"]
            norm = np.linalg.norm([dx, dy])
            if norm > 0:
                v_avoidfish = np.array([dx, dy]) / norm

    min_dist_robot = float('inf')
    nearest_other_robot = None
    for other in all_robots:
        if other["id"] != robot["id"]:
            dist = max(abs(other["x"] - robot["x"]), abs(other["y"] - robot["y"]))
            if dist < min_dist_robot:
                min_dist_robot = dist
                nearest_other_robot = other
                
    if nearest_other_robot and min_dist_robot <= 2:
        dx = robot["x"] - nearest_other_robot["x"]
        dy = robot["y"] - nearest_other_robot["y"]
        if dx == 0 and dy == 0:
            dx, dy = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        norm = np.linalg.norm([dx, dy])
        if norm > 0:
            v_avoidrobot = np.array([dx, dy]) / norm

    v_noise = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])

    w_trash = state["settings"]["w_trash"] if robot["type"] == "collector" else 1.0
    w_avoidfish = state["settings"]["w_avoidfish"]
    w_avoidrobot = state["settings"]["w_avoidrobot"]
    
    v_next = w_trash * v_trash + w_avoidfish * v_avoidfish + w_avoidrobot * v_avoidrobot + v_noise
    
    if np.linalg.norm(v_next) > 0:
        v_dir = np.round(v_next / np.linalg.norm(v_next)).astype(int)
        return int(v_dir[0]), int(v_dir[1])
    return 0, 0

async def simulation_loop():
    while True:
        if state["status"] == "running":
            state["step"] += 1
            
            num_new_trash = random.randint(0, 2)
            for _ in range(num_new_trash):
                state["trash"].append({
                    "x": random.randint(0, state["grid_width"] - 1),
                    "y": random.randint(0, state["grid_height"] - 1)
                })
                
            total_stress_current = 0.0
            surviving_fishes = []
            
            for fish in state["fishes"]:
                min_dist = float('inf')
                nearest_robot = None
                for robot in state["robots"]:
                    dist = max(abs(robot["x"] - fish["x"]), abs(robot["y"] - fish["y"]))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_robot = robot

                if nearest_robot and min_dist <= 3:
                    dx = fish["x"] - nearest_robot["x"]
                    dy = fish["y"] - nearest_robot["y"]
                    step_x = 1 if dx > 0 else (-1 if dx < 0 else random.choice([-1, 1]))
                    step_y = 1 if dy > 0 else (-1 if dy < 0 else random.choice([-1, 1]))
                    fish["stress"] += 2.0
                else:
                    step_x = random.choice([-1, 0, 1])
                    step_y = random.choice([-1, 0, 1])
                    fish["stress"] = max(0.0, fish["stress"] - 0.5)

                fish["x"] = max(0, min(state["grid_width"] - 1, fish["x"] + step_x))
                fish["y"] = max(0, min(state["grid_height"] - 1, fish["y"] + step_y))
                
                if fish["stress"] >= 10.0:
                    state["accumulated_stress"] += fish["stress"]
                else:
                    surviving_fishes.append(fish)
                    total_stress_current += fish["stress"]
                
            state["fishes"] = surviving_fishes
            state["stats"]["total_stress"] = total_stress_current + state["accumulated_stress"]

            for robot in state["robots"]:
                if robot["type"] == "scout" and not robot.get("is_charging", False):
                    new_discoveries = 0
                    memory_coords = {(tm["x"], tm["y"]) for tm in state["shared_trash_memory"]}
                    
                    for t in state["trash"]:
                        dist = max(abs(t["x"] - robot["x"]), abs(t["y"] - robot["y"]))
                        if dist <= 7:
                            if (t["x"], t["y"]) not in memory_coords:
                                state["shared_trash_memory"].append(t.copy())
                                memory_coords.add((t["x"], t["y"]))
                                new_discoveries += 1
                    
                    if new_discoveries > 0:
                        consumed = 5 * new_discoveries
                        robot["energy"] -= consumed
                        state["stats"]["energy"] += consumed
                        if robot["energy"] <= 0:
                            robot["is_charging"] = True
                            robot["charge_timer"] = 0

            claimed_trash_coords = set()
            collector_targets = {}
            active_collectors = [r for r in state["robots"] if r["type"] == "collector" and not r.get("is_charging", False)]

            for robot in active_collectors:
                visible_trash = list(state["shared_trash_memory"])
                memory_coords = {(t["x"], t["y"]) for t in visible_trash}
                
                for t in state["trash"]:
                    dist = max(abs(t["x"] - robot["x"]), abs(t["y"] - robot["y"]))
                    if dist <= 2 and (t["x"], t["y"]) not in memory_coords:
                        visible_trash.append(t)

                min_dist = float('inf')
                best_trash = None
                for t in visible_trash:
                    t_coord = (t["x"], t["y"])
                    if t_coord in claimed_trash_coords:
                        continue
                    
                    dist = abs(t["x"] - robot["x"]) + abs(t["y"] - robot["y"])
                    if dist < min_dist:
                        min_dist = dist
                        best_trash = t

                if best_trash:
                    claimed_trash_coords.add((best_trash["x"], best_trash["y"]))
                    collector_targets[robot["id"]] = best_trash

            positions = {}
            for robot in state["robots"]:
                if robot.get("is_charging", False):
                    robot["charge_timer"] += 1
                    if robot["charge_timer"] >= state["charge_time"]:
                        robot["energy"] = state["max_battery"]
                        robot["is_charging"] = False
                        robot["charge_timer"] = 0
                    
                    pos_key = (robot["x"], robot["y"])
                    if pos_key in positions:
                        state["stats"]["n_collision"] += 1
                    positions[pos_key] = True
                    continue

                assigned_target_list = []
                if robot["type"] == "collector":
                    target = collector_targets.get(robot["id"])
                    if target:
                        assigned_target_list.append(target)

                dx, dy = calculate_v_next(robot, assigned_target_list, state["robots"])
                next_x = max(0, min(state["grid_width"] - 1, robot["x"] + dx))
                next_y = max(0, min(state["grid_height"] - 1, robot["y"] + dy))
                
                if dx != 0 or dy != 0:
                    robot["energy"] -= 1
                    state["stats"]["energy"] += 1
                    
                robot["x"] = next_x
                robot["y"] = next_y

                pos_key = (robot["x"], robot["y"])
                if pos_key in positions:
                    state["stats"]["n_collision"] += 1
                positions[pos_key] = True

                if robot["type"] == "collector":
                    recovered = [t for t in state["trash"] if t["x"] == robot["x"] and t["y"] == robot["y"]]
                    if recovered:
                        num_recovered = len(recovered)
                        state["stats"]["n_trash"] += num_recovered
                        state["trash"] = [t for t in state["trash"] if t not in recovered]
                        state["shared_trash_memory"] = [t for t in state["shared_trash_memory"] if (t["x"], t["y"]) not in [(r["x"], r["y"]) for r in recovered]]
                        
                        consumed = 10 * num_recovered
                        robot["energy"] -= consumed
                        state["stats"]["energy"] += consumed

                if robot["energy"] <= 0:
                    robot["is_charging"] = True
                    robot["charge_timer"] = 0

            while len(state["trash"]) < 5:
                state["trash"].append({
                    "x": random.randint(0, state["grid_width"] - 1),
                    "y": random.randint(0, state["grid_height"] - 1)
                })
            
            s_trash = 30 * state["stats"]["n_trash"]
            s_col = 10 * state["stats"]["n_collision"]
            s_ene = int(0.2 * state["stats"]["energy"])
            s_str = int(0.5 * state["stats"]["total_stress"])
            state["stats"]["score"] = s_trash - s_col - s_ene - s_str

            if state["step"] >= state["max_steps"]:
                state["status"] = "finished"

        await asyncio.sleep(0.5)

@app.on_event("startup")
async def startup_event():
    # 起動時の初期機体数も各4機、スコア2000に変更
    await reset_simulation(ResetConfig(num_scouts=4, num_collectors=4, max_steps=120, target_score=2000, max_battery=200, charge_time=10))
    state["status"] = "standby"
    asyncio.create_task(simulation_loop())

@app.get("/")
async def get_html():
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(json.dumps(state))
            await asyncio.sleep(0.5)
    except Exception:
        pass