---
title: AgriRL Environment
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
---
# AgriRL Environment

A reinforcement learning environment where an agent manages soil moisture and crop growth.

## Actions

* wait
* irrigate
* fertilize
* harvest

## Goal

Maximize crop yield while maintaining optimal soil conditions.

## Scoring

Final score is based on:

* crop growth
* efficiency (growth vs time)

## Explainability

The baseline agent provides reasoning for each action:
- Irrigate when soil moisture is low
- Fertilize during early growth phase
- Harvest at peak efficiency

## Baseline Agent

A rule-based agent is provided in server/inference.py.

## Highlights

* Real-world inspired agricultural simulation
* Dynamic weather affecting crop growth
* Resource optimization via RL
* Deterministic scoring system

## API
- POST /reset
- POST /step

## Deployment
HuggingFace Space:
https://huggingface.co/spaces/ayush712145/agrirl-env


## Run locally

uv run server

