"""Patch to update Neo4j password in required files"""

import fileinput
import sys
import os

def update_password(filename, old_password="password", new_password="00000000"):
    with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace(f'password="{old_password}"', f'password="{new_password}"'), end='')

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    # Update automated_runner.py
    runner_path = os.path.join(base_dir, "automated_runner.py")
    if os.path.exists(runner_path):
        update_password(runner_path)
        print("Updated automated_runner.py")
        
    # Update quantum_monitor.py
    monitor_path = os.path.join(parent_dir, "ai_ml_lab", "quantum_monitor.py")
    if os.path.exists(monitor_path):
        update_password(monitor_path)
        print("Updated quantum_monitor.py")
