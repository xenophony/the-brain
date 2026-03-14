"""
Shared Pong oracle for trajectory prediction and motion planning probes.

Simulates ball trajectory with wall reflections and determines optimal paddle action.
"""


def pong_oracle(ball_x, ball_y, ball_dx, ball_dy,
                paddle_x, paddle_cy, paddle_h,
                paddle_speed, court_h):
    """Simulate ball trajectory and determine optimal paddle action.

    Returns (action, steps, ball_final_y) where action is 'up', 'down', or 'stay'.
    """
    x, y, dy = float(ball_x), float(ball_y), float(ball_dy)
    dx = float(ball_dx)
    steps = 0
    while x < paddle_x:
        x += dx
        y += dy
        steps += 1
        if y <= 0:
            y = abs(y)
            dy = abs(dy)
        elif y >= court_h:
            y = 2 * court_h - y
            dy = -abs(dy)

    ball_final_y = y
    distance = abs(ball_final_y - paddle_cy)

    if distance <= paddle_h / 2:
        return "stay", steps, ball_final_y

    max_reach = paddle_speed * steps
    if max_reach < distance - paddle_h / 2:
        return "stay", steps, ball_final_y  # unreachable

    if ball_final_y > paddle_cy:
        return "up", steps, ball_final_y
    else:
        return "down", steps, ball_final_y
