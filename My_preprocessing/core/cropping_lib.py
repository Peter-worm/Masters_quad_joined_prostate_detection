def center_crop_shift(figure, center, crop_size):
    shape = figure.shape
    start = [int(c - s//2) for c, s in zip(center, crop_size)]
    end = [start[i] + crop_size[i] for i in range(3)]

    for i in range(3):
        if start[i] < 0:
            end[i] -= start[i]
            start[i] = 0
        if end[i] > shape[i]:
            start[i] -= end[i] - shape[i]
            end[i] = shape[i]

    return figure[start[0]:end[0], start[1]:end[1], start[2]:end[2]]