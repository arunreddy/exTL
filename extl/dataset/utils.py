import matplotlib.pyplot as plt


def plot_image_from_data(data, label=None):
  plt.figure()
  plt.imshow(data)
  if label:
    plt.title(label)
  # plt.show()
