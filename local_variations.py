import math as m
import random
import numpy as np

import scipy.integrate as integrate
import matplotlib.pyplot as plt

alpha = 0.1
beta = 0.5
initial_value = 0.0
final_value = 1.0
length = 1


def interpolate_y(
        x: float, x_prev: float, x_next: float, y_prev: float, y_next: float
) -> float:
    return y_prev + ((y_next - y_prev) / (x_next - x_prev)) * (x - x_prev)


def compute_y_derivative(x_prev: float, x_next: float, y_prev: float, y_next: float) -> float:
    return (y_next - y_prev) / (x_next - x_prev)


def z_derivative_stupid(x_prev: float, x_next: float, z_prev: float, z_next: float) -> float:
    return (z_next - z_prev) / (x_next - x_prev)


def compute_z_derivative(
        x_prev: float, x_next: float, y_prev: float, y_next: float
) -> float:
    return (m.sin(5 * x_next) * m.sin(y_next) - m.sin(5 * x_prev) * m.sin(y_prev)) / (x_next - x_prev)


def compute_f(x_prev: float, x_next: float, y_prev: float, y_next: float) -> float:
    y_derivative = compute_y_derivative(x_prev, x_next, y_prev, y_next)
    z_derivative = compute_z_derivative(x_prev, x_next, y_prev, y_next)
    return m.sqrt(1 + y_derivative ** 2 + z_derivative ** 2)


def calculate_f_sum(route: list[float], delta: float) -> float:
    result = 0
    for i in range(len(route) - 1):
        result += compute_f(i * delta, (i + 1) * delta, route[i], route[i + 1])
    return result


def compute_j_integral(
        x_prev: float, x_next: float, y_prev: float, y_next: float, prev_route: list[float]
) -> float:
    f = compute_f(x_prev, x_next, y_prev, y_next)
    sum_f = calculate_f_sum(prev_route, x_next - x_prev)
    delta_x = x_next - x_prev

    first_integrand = lambda x: alpha * (x - x_prev)
    second_integrand = lambda x: beta
    third_integrand = lambda x: alpha

    first_integral = (f ** 2) * integrate.quad(first_integrand, x_prev, x_next)[0]
    second_integral = f * integrate.quad(second_integrand, x_prev, x_next)[0]
    third_integral = f * sum_f * delta_x * integrate.quad(third_integrand, x_prev, x_next)[0]

    return first_integral + second_integral + third_integral


def generate_broom_approximation(approx_points: int, local_points: int) -> list[float]:
    step_x = length / (approx_points - 1)
    step_y = 1 / (approx_points - 1)

    road_map: list[list[tuple[float, int | None]]] = [[(float("inf"), None)] * approx_points]
    road_map[0][0] = (0.0, None)  # Starting point has a cost of 0

    def extract_route(x: int, y: int) -> list[float]:
        route = [step_y * y]
        while road_map[x][y][1] is not None:
            y = road_map[x][y][1]
            x -= 1
            route.insert(0, step_y * y)
        return route

    for i in range(1, approx_points):
        prev_x = step_x * (i - 1)
        curr_x = step_x * i
        previous_routes = [extract_route(i - 1, j) for j in range(approx_points)]
        new_column = []

        for j in range(approx_points):
            curr_y = step_y * j
            paths = [
                (
                    compute_j_integral(prev_x, curr_x, step_y * k, curr_y, previous_routes[k])
                    + road_map[i - 1][k][0],
                    k,
                )
                for k in range(approx_points)
            ]
            new_column.append(min(paths, key=lambda path: path[0]))

        road_map.append(new_column)

    local_result = [1.0]
    current_index = approx_points - 1
    for i in range(approx_points - 1, -1, -1):
        next_index = road_map[i][current_index][1]
        if next_index is None:
            break
        local_result.insert(0, step_y * next_index)
        current_index = next_index

    result = [0.0]
    for i in range(1, local_points):
        fractional_index = i * (approx_points - 1) / (local_points - 1)
        lower_index = int(fractional_index)
        if lower_index == approx_points - 1:
            result.append(local_result[approx_points - 1])
        else:
            weight = fractional_index - lower_index
            interpolated_value = (
                    (1 - weight) * local_result[lower_index]
                    + weight * local_result[lower_index + 1]
            )
            result.append(interpolated_value)

    return result


def generate_random_route(local_points: int) -> list[float]:
    return [initial_value] + [random.random() for _ in range(local_points - 2)] + [final_value]


def generate_x2_route(local_points: int) -> list[float]:
    return [((i * length / (local_points - 1)) ** 2) for i in range(local_points)]


def generate_xm_route(m: int, local_points: int) -> list[float]:
    return [((i * length / (local_points - 1)) ** m) for i in range(local_points)]


def optimize_point(
        step_x: float, step_y: float, x_curr: float, y_prev: float, y_curr: float, y_next: float,
        prev_route: list[float]
) -> float:
    x_prev = x_curr - step_x
    x_next = x_curr + step_x
    alpha_0 = compute_j_integral(
        x_prev, x_curr, y_prev, y_curr, prev_route
    ) + compute_j_integral(x_curr, x_next, y_curr, y_next, prev_route + [y_curr])
    alpha_1 = compute_j_integral(
        x_prev, x_curr, y_prev, y_curr + step_y, prev_route
    ) + compute_j_integral(
        x_curr, x_next, y_curr + step_y, y_next, prev_route + [y_curr + step_y]
    )
    alpha_2 = compute_j_integral(
        x_prev, x_curr, y_prev, y_curr - step_y, prev_route
    ) + compute_j_integral(
        x_curr, x_next, y_curr - step_y, y_next, prev_route + [y_curr - step_y]
    )
    if alpha_0 <= alpha_1 and alpha_0 <= alpha_2:
        return y_curr
    elif alpha_1 <= alpha_0 and alpha_1 <= alpha_2:
        return y_curr + step_y
    else:
        return y_curr - step_y


def optimize_route(route: list[float], iterations: int, local_points: int, step_x: float, step_y: float) -> list[float]:
    for _ in range(iterations):
        prev_route = route.copy()
        for i in range(1, local_points - 1):
            route[i] = optimize_point(
                step_x, step_y, i * step_x, route[i - 1], route[i], route[i + 1], route[:i]
            )
    return route


def calculate_route_cost(route: list[float], local_points: int, step_x: float) -> float:
    return sum(
        compute_j_integral(i * step_x, (i + 1) * step_x, route[i], route[i + 1], route[: (i + 1)])
        for i in range(local_points - 1)
    )


def plot_route(route: list[float]) -> None:
    x_graph = [i * (length / (len(route) - 1)) for i in range(len(route))]
    plt.plot(x_graph, route)
    plt.show()


def plot_route_3d_line(route: list[float]) -> None:
    x = [i * (length / (len(route) - 1)) for i in range(len(route))]
    y = route
    z = [m.sin(5 * xi) * m.sin(yi) for xi, yi in zip(x, y)]  # z = sin(5x) * sin(y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def plot_combined_3d(route: list[float]) -> None:
    x_line = [i * (length / (len(route) - 1)) for i in range(len(route))]
    y_line = route
    z_line = [m.sin(5 * xi) * m.sin(yi) for xi, yi in zip(x_line, y_line)]

    x_surface = np.linspace(0, length, len(route))
    y_surface = np.array(route)
    X, Y = np.meshgrid(x_surface, y_surface)
    Z = np.sin(5 * X) * np.sin(Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    ax.plot(x_line, y_line, z_line, color='red', linewidth=2, label='3D Line')

    ax.set_xlabel('X')
    ax.set_ylabel('Y (Route)')
    ax.set_zlabel('Z')
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    ax.legend()

    plt.show()


def main() -> None:
    # Local variations net
    local_variations_points = 32
    # First approximation (Kiev broom) net
    approximation_points = 16
    # Local variations iteration number
    iterations = 16

    step_x = length / (local_variations_points - 1)
    step_y = length / (2 * local_variations_points - 1)

    route = generate_broom_approximation(approximation_points, local_variations_points)
    # route = generate_random_route(local_variations_points)
    # route = generate_x2_route(local_variations_points)
    # route = generate_xm_route(1000, local_variations_points)
    cost = calculate_route_cost(route, local_variations_points, step_x)
    print("Route cost: \n", cost)
    plot_combined_3d(route)

    route = optimize_route(route, iterations, local_variations_points, step_x, step_y)
    cost = calculate_route_cost(route, local_variations_points, step_x)
    print("Route cost: \n", cost)
    plot_combined_3d(route)


if __name__ == "__main__":
    main()
