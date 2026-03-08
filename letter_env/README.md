---
title: OpenEnv Letter Env
emoji: 🔤
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Letter Guessing Environment

An OpenEnv 0.2.1 reinforcement learning environment where an agent must guess the letters 'a' then 'z' in order to win.

## API

- `POST /reset` — start a new episode
- `POST /step` — submit an action `{"action": {"letter": "x"}}`
- `GET /state` — get current episode state
- `GET /health` — health check
