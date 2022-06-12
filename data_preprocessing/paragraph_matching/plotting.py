import matplotlib.pyplot as plt
import numpy as np

def similarities_plot(matching,metric_name):
    results = []
    for y, row in enumerate(matching):
        if np.min(row) != np.max(row):
            results.append((y, np.argmax(row)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(matching)
    zeros = np.zeros_like(matching)
    zeros[tuple(np.array(results).T)] = 1
    ax2.imshow(zeros)
    plt.suptitle(f"Paragraph similarity - {metric_name}")
    ax1.set_title('Continuous')
    ax2.set_title('Discrete')
    ax1.set_xlabel('Number of a paragraph in second translation')
    ax1.set_ylabel('Number of a paragraph in first translation')
    ax2.set_xlabel('Number of a paragraph in second translation')
    plt.show()