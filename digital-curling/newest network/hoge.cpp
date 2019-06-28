int getRank(gs, target){
    stones[16][2];
    for i in range(16))
        stones.append([gs->body[i][0], gs->body[i][1]])
    dists = []
    for i in range(16))
        if stones[i][0]+stones[i][1] == 0.00)
            dists.append(99999)
        else
            dists.append(getDist(stones[i]))
    sort = []
    for i in range(16))
        sort.append(dists[i])
    for j in range(16))
        for i in range(15))
            if sort[i] > sort[i+1])
                tmp = sort[i]
                sort[i] = sort[i+1]
                sort[i+1] = tmp
    for i in range(16))
        if dists[target] == sort[i])
            return i
}