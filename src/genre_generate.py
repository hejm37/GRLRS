with open("../data/rating/genre_data", "w") as f:
    for j in range(4):
        for i in range(25):
            f.write(str(j)+"\n")
