def evaluate_easy(env):
    return min(1.0, env.crop_growth / 100)


def evaluate_medium(env):
    cost_penalty = env.day * 0.5
    score = (env.crop_growth - cost_penalty) / 100
    return max(0.0, min(1.0, score))


def evaluate_hard(env):
    if env.crop_growth < 50:
        return 0.0

    efficiency = env.crop_growth / (env.day + 1)
    score = efficiency / 5
    return max(0.0, min(1.0, score))