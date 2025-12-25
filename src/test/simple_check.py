#!/usr/bin/env python
print("Script started")

import sys
print(f"Python version: {sys.version}")

import os
print(f"Current directory: {os.getcwd()}")

current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {current_dir}")

src_dir = os.path.dirname(current_dir)
print(f"Src directory: {src_dir}")

# Check if judge folder exists
judge_path = os.path.join(src_dir, "judge")
print(f"Judge path: {judge_path}")
print(f"Judge exists: {os.path.exists(judge_path)}")

if os.path.exists(judge_path):
    files = os.listdir(judge_path)
    print(f"Files in judge: {files}")

print("Script completed")