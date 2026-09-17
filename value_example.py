"""A worked Value example: predict one number, backpropagate, then update.

Run with: python value_example.py
The model is prediction = weight * x + bias, with squared error as its loss.
"""

from backpropagation import Value


def run_value_example():
    # Values represent individual numbers. Labels help us inspect the graph.
    x = Value(3.0, label="x")
    weight = Value(2.0, label="weight")
    bias = Value(1.0, label="bias")
    target = 10.0

    # 1. Forward pass: Python operators create Value nodes and save derivatives.
    weighted_input = weight * x       # 2 * 3 = 6
    prediction = weighted_input + bias  # 6 + 1 = 7
    error = prediction - target       # 7 - 10 = -3
    loss = error ** 2                 # (-3)**2 = 9
    print("Forward pass: prediction = weight * x + bias")
    print(f"prediction={prediction.data:.2f}, target={target:.2f}, "
          f"loss={loss.data:.2f}")

    # 2. Backward pass: start at loss, then apply the chain rule toward inputs.
    # This fills .grad; it does not change any .data values.
    loss.backward()
    print("\nAfter loss.backward(): each .grad is d(loss)/d(node)")
    print(f"{'node':<16} {'.data':>8} {'.grad':>8}")
    for name, node in (
        ("x", x), ("weight", weight), ("bias", bias),
        ("weighted_input", weighted_input), ("prediction", prediction),
        ("error", error), ("loss", loss),
    ):
        print(f"{name:<16} {node.data:>8.2f} {node.grad:>8.2f}")

    print("\nWhere the gradients come from:")
    print(f"dL/dprediction = 2 * (prediction - target) = {prediction.grad:.2f}")
    print(f"dL/dweight = dL/dprediction * x = {weight.grad:.2f}")
    print(f"dL/dbias = dL/dprediction * 1 = {bias.grad:.2f}")
    print(f"dL/dx = dL/dprediction * weight = {x.grad:.2f}")
    print("The input has a gradient too; here we train only weight and bias.")

    # 3. Optimization: subtract the gradient, scaled by a learning rate.
    learning_rate = 0.01
    weight.data -= learning_rate * weight.grad  # 2 - 0.01*(-18) = 2.18
    bias.data -= learning_rate * bias.grad     # 1 - 0.01*(-6) = 1.06

    # Rebuild the forward graph: old nodes still describe the old computation.
    new_prediction = weight * x + bias
    new_loss = (new_prediction - target) ** 2
    print(f"\nAfter one gradient-descent step (learning rate={learning_rate}):")
    print(f"weight={weight.data:.2f}, bias={bias.data:.2f}, x={x.data:.2f}")
    print(f"prediction={new_prediction.data:.2f}")
    print(f"loss: {loss.data:.2f} -> {new_loss.data:.2f}")


if __name__ == "__main__":
    run_value_example()
