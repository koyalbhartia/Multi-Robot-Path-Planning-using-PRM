#
#  Copyright 2019 Harsh Kakashaniya,Koyal Bhartia,Aalap Rana
#  @file    final_prm.py
#  @date    14/05/2019
#  @version 1.0
#
#  @brief This is the code for Project 5 - ENPM661
#
import random
import math
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

class Node:

    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)


class KDTree:
    def __init__(self, data):
        self.tree = scipy.spatial.cKDTree(data)
    def search(self, inp, k=1):
        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []
            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)
            return index, dist
        dist, index = self.tree.query(inp, k=k)
        return index, dist
    def search_in_distance(self, inp, r):
        index = self.tree.query_ball_point(inp, r)
        return index


def PRM_planning(start,goal,obs_x, obs_y, robot_size):
    obkdtree = KDTree(np.vstack((obs_x, obs_y)).T)
    sample_x, sample_y = sample_points(start,goal, robot_size, obs_x, obs_y, obkdtree)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")
    road_map = generate_roadmap(sample_x, sample_y, robot_size, obkdtree)
    rx1,ry1 =dijkstra_planning(start[0,0], start[0,1], goal[0,0],goal[0,1], obs_x, obs_y, robot_size, road_map, sample_x, sample_y,6)
    rx2,ry2 =dijkstra_planning(start[1,0], start[1,1], goal[1,0],goal[1,1], obs_x, obs_y, robot_size, road_map, sample_x, sample_y,4)
    rx3,ry3 =dijkstra_planning(start[2,0], start[2,1], goal[2,0],goal[2,1], obs_x, obs_y, robot_size, road_map, sample_x, sample_y,2)
    return rx1,ry1,rx2,ry2,rx3,ry3

def is_collision(startx, starty, goalx, goaly, rr, okdtree):
    x = startx
    y = starty
    dx = goalx - startx
    dy = goaly - starty
    yaw = math.atan2(goaly - starty, goalx - startx)
    d = math.sqrt(dx**2 + dy**2)
    if d >= MAX_EDGE_LEN:
        return True
    D = rr
    nstep = round(d / D)
    for i in range(nstep):
        idxs, dist = okdtree.search(np.array([x, y]).reshape(2, 1))
        if dist[0] <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)
    idxs, dist = okdtree.search(np.array([goalx, goaly]).reshape(2, 1))
    if dist[0] <= rr:
        return True  # collision
    return False  # OK


def generate_roadmap(sample_x, sample_y, rr, obkdtree):
    road_map = []
    nsample = len(sample_x)
    skdtree = KDTree(np.vstack((sample_x, sample_y)).T)
    for (i, ix, iy) in zip(range(nsample), sample_x, sample_y):
        index, dists = skdtree.search(
            np.array([ix, iy]).reshape(2, 1), k=nsample)
        inds = index[0]
        edge_id = []
        for ii in range(1, len(inds)):
            nx = sample_x[inds[ii]]
            ny = sample_y[inds[ii]]
            if not is_collision(ix, iy, nx, ny, rr, obkdtree):
                edge_id.append(inds[ii])
            if len(edge_id) >= N_KNN:
                break
        road_map.append(edge_id)
    return road_map

def dijkstra_planning(startx, starty, goalx, goaly, obs_x, obs_y, rr, road_map, sample_x, sample_y,ob_no):
    nstart = Node(startx, starty, 0.0, -1)
    ngoal = Node(goalx, goaly, 0.0, -1)
    openset, closedset = dict(), dict()
    openset[len(road_map) - ob_no] = nstart
    while True:
        if not openset:
            print("Cannot find path")
            break
        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]
        if show_animation and len(closedset.keys()) % 2 == 0:
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)
        if c_id == (len(road_map) - (ob_no-1)):
            print("goal is found!")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break
        del openset[c_id]
        closedset[c_id] = current
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.sqrt(dx**2 + dy**2)
            dxg=sample_x[n_id] - ngoal.x
            dyg=sample_y[n_id] - ngoal.y
            dg=math.sqrt(dxg**2 + dyg**2)
            node = Node(sample_x[n_id], sample_y[n_id],current.cost + d+1.5*dg, c_id)
            if n_id in closedset:
                continue
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node
    rx, ry = [ngoal.x], [ngoal.y]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        ry.append(n.y)
        pind = n.pind
    rx.reverse()
    ry.reverse()
    return rx, ry


def plot_road_map(road_map, sample_x, sample_y):
    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]
            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")

def heuristic(startx,starty,endx,endy):
    dist=np.sqrt(math.pow((startx-endx),2)+math.pow((starty-endy),2))
    return dist

def sample_points(start,goal, rr, obs_x, obs_y, obkdtree):
    maxx = max(obs_x)
    maxy = max(obs_y)
    minx = min(obs_x)
    miny = min(obs_y)
    sample_x, sample_y = [], []
    while len(sample_x) <= N_SAMPLE:
        tx = (random.random() - minx) * (maxx - minx)
        ty = (random.random() - miny) * (maxy - miny)
        index, dist = obkdtree.search(np.array([tx, ty]).reshape(2, 1))
        if dist[0] >= rr:
            sample_x.append(tx)
            sample_y.append(ty)
    for i in range(len(start)):
        sample_x.append(start[i,0])
        sample_y.append(start[i,1])
        sample_x.append(goal[i,0])
        sample_y.append(goal[i,1])

    return sample_x, sample_y

def check_radius(x1,x2,radius):
    if abs(x1-x2)>radius:return False
    else : return True

def compare_robots(indexA,indexB,rxA,rxB,ryA,ryB,radius):
    lengthA=len(rxA)
    lengthB=len(rxB)
    heuA=heuristic(rxA[indexA],ryA[indexA],rxA[lengthA-1],ryA[lengthA-1])
    heuB=heuristic(rxB[indexB],ryB[indexB],rxB[lengthB-1],ryB[lengthB-1])
    prio=max(heuA,heuB)
    if prio==heuA:
        if indexA==lengthA-1: indexA=indexA
        else: indexA+=1
    else:
        if indexB==lengthB-1: indexB=indexB
        else: indexB+=1
    return indexA,indexB

def increment(rxA,rxB,rxC,ryA,ryB,ryC,indexA,indexB,indexC,radius):
    lengthA=len(rxA)
    lengthB=len(rxB)
    lengthC=len(rxC)
    che=0
    pi=0
    if not (check_radius(rxA[indexA+1],rxB[indexB],radius) and check_radius(ryA[indexA+1],ryB[indexB],radius)):
        che=1
    else:
        heuA=heuristic(rxA[indexA],ryA[indexA],rxA[0],ryA[0])
        heuB=heuristic(rxB[indexB],ryB[indexB],rxB[0],ryB[0])
        prio=min(heuA,heuB)
        if prio==heuA:
            if indexA==0: indexA=indexA
            else: indexA=indexA-1
    if not (check_radius(rxA[indexA+1],rxC[indexC],radius) and check_radius(ryA[indexA+1],ryC[indexC],radius)):
        pi=1
    else:
        heuA=heuristic(rxA[indexA],ryA[indexA],rxA[0],ryA[0])
        heuC=heuristic(rxC[indexC],ryC[indexC],rxC[0],ryC[0])
        prio=min(heuA,heuC)
        if prio==heuA:
            if indexA==0: indexA=indexA
            else: indexA=indexA-1

    if che==1 and pi==1:
        if indexA==lengthA-1: indexA=indexA
        else: indexA+=1

    return indexA

def Global_path(rx1,ry1,rx2,ry2,rx3,ry3,radius):
    print(len(rx1))
    print(len(rx2))
    print(len(rx3))

    length1=len(rx1)
    length2=len(rx2)
    length3=len(rx3)
    nodes=np.mat([0,rx1[0],ry1[0],rx2[0],ry2[0],rx3[0],ry3[0]]).astype(float)
    t=0
    index1=0
    index2=0
    index3=0
    count=0
    while (index1<length1-1 or index2<length2-1 or index3<length3-1) and count<2000 :
        count+=1
        print("index1:",index1)
        print("index2:",index2)
        print("index3:",index3)
        print("t:",t)
        if check_radius(rx1[index1],rx2[index2],radius) and check_radius(ry1[index1],ry2[index2],radius):
            if check_radius(rx2[index2],rx3[index3],radius) and check_radius(ry2[index2],ry3[index3],radius):
                heu1=heuristic(rx1[index1],ry1[index1],rx1[length1-1],ry1[length1-1])
                heu2=heuristic(rx2[index2],ry2[index2],rx2[length2-1],ry2[length2-1])
                heu3=heuristic(rx3[index3],ry3[index3],rx3[length3-1],ry3[length3-1])
                prio=max(heu1,heu2,heu3)
                if prio==heu1:
                    if index1==length1-1: index1=index1
                    else: index1+=1
                elif prio==heu2:
                    if index2==length2-1: index2=index2
                    else: index2+=1
                else:
                    if index3==length3-1: index3=index3
                    else: index3+=1
                t+=1
                new_node=[t,rx1[index1],ry1[index1],rx2[index2],ry2[index2],rx3[index3],ry3[index3]]
                nodes=np.vstack((nodes,new_node))
            else:
                index1,index2=compare_robots(index1,index2,rx1,rx2,ry1,ry2,radius)
                if index3==length3-1: index3=index3
                else: index3+=1
                new_node=[t,rx1[index1],ry1[index1],rx2[index2],ry2[index2],rx3[index3],ry3[index3]]
                nodes=np.vstack((nodes,new_node))

        elif check_radius(rx1[index1],rx3[index3],radius) and check_radius(ry1[index1],ry3[index3],radius):
            index1,index3=compare_robots(index1,index3,rx1,rx3,ry1,ry3,radius)
            if index2==length2-1: index2=index2
            else: index2+=1
            t+=1
            new_node=[t,rx1[index1],ry1[index1],rx2[index2],ry2[index2],rx3[index3],ry3[index3]]
            nodes=np.vstack((nodes,new_node))
        elif check_radius(rx2[index2],rx3[index3],radius) and check_radius(ry2[index2],ry3[index3],radius):
            index2,index3=compare_robots(index2,index3,rx2,rx3,ry2,ry3,radius)
            if index1==length1-1: index1=index1
            else: index1+=1
            t+=1
            new_node=[t,rx1[index1],ry1[index1],rx2[index2],ry2[index2],rx3[index3],ry3[index3]]
            nodes=np.vstack((nodes,new_node))
        else:
            if index1==length1-1: index1=index1
            else: index1=increment(rx1,rx2,rx3,ry1,ry2,ry3,index1,index2,index3,radius)
            if index2==length2-1: index2=index2
            else: index2=increment(rx2,rx1,rx3,ry2,ry1,ry3,index2,index1,index3,radius)
            if index3==length3-1: index3=index3
            else: index3=increment(rx3,rx1,rx2,ry3,ry1,ry2,index3,index1,index2,radius)
            t+=1
            new_node=[t,rx1[index1],ry1[index1],rx2[index2],ry2[index2],rx3[index3],ry3[index3]]
            nodes=np.vstack((nodes,new_node))
    return nodes

def plot_global(nodes):
    print(nodes)
    for i in range(len(nodes)-1):
        plt.plot(nodes[i:i+2,1],nodes[i:i+2,2], "-r")
        plt.plot(nodes[i:i+2,3],nodes[i:i+2,4],"-g")
        plt.plot(nodes[i:i+2,5],nodes[i:i+2,6],"-b")
        plt.plot(nodes[i:i+2,1],nodes[i:i+2,2], ".r",markeredgewidth=10,markersize=10)
        plt.plot(nodes[i:i+2,3],nodes[i:i+2,4], ".g",markeredgewidth=10,markersize=10)
        plt.plot(nodes[i:i+2,5],nodes[i:i+2,6], ".b",markeredgewidth=10,markersize=10)

        plt.axis([0,250,0,200])

        plt.pause(1)
    plt.show()

def obstacle_space(start,goal):
    obs_x = []
    obs_y = []

    #bottom wall
    for i in range(250):
        obs_x.append(i)
        obs_y.append(0)
    #left wall
    for j in range(200):
        obs_x.append(0)
        obs_y.append(j)
    #top wall
    for i in range(250):
        obs_x.append(i)
        obs_y.append(200)

    #right wall
    for j in range(200):
        obs_x.append(250)
        obs_y.append(j)

    #obstacle 1
    #lower
    for i in range(50,75):
        obs_x.append(i)
        obs_y.append(50)

    for j in range(0,75):
        obs_x.append(50)
        obs_y.append(j)

    #upper
    for i in range(50,75):
        obs_x.append(i)
        obs_y.append(150)

    for j in range(125,200):
        obs_x.append(50)
        obs_y.append(j)

    #obstacle2
    for j in range(10):
        obs_x.append(125)
        obs_y.append(j)


    for j in range(40,50):
        obs_x.append(125)
        obs_y.append(j)

    for j in range(150,160):
        obs_x.append(125)
        obs_y.append(j)


    for j in range(190,200):
        obs_x.append(125)
        obs_y.append(j)

    for i in range(100,150):
        obs_x.append(i)
        obs_y.append(50)


    for i in range(100,150):
        obs_x.append(i)
        obs_y.append(150)


    #obstacle3

    for i in range(175,225):
        obs_x.append(i)
        obs_y.append(75)

    for j in range(0,75):
        obs_x.append(200)
        obs_y.append(j)

    for i in range(175,225):
        obs_x.append(i)
        obs_y.append(125)

    for j in range(125,200):
        obs_x.append(200)
        obs_y.append(j)

    #obstacle4

    for i in range(240,250):
        obs_x.append(i)
        obs_y.append(110)

    for i in range(240,250):
        obs_x.append(i)
        obs_y.append(90)

    if show_animation:
        plt.plot(obs_x, obs_y, ".k")
        plt.plot(start[0,0], start[0,1], "^r")
        plt.plot(goal[0,0], goal[0,1], "^c")
        plt.plot(start[1,0], start[1,1], "^r")
        plt.plot(goal[1,0], goal[1,1], "^c")
        plt.plot(start[2,0], start[2,1], "^r")
        plt.plot(goal[2,0], goal[2,1], "^c")
        plt.grid(True)
        plt.axis("equal")
    return obs_x,obs_y

def main():
    print(__file__ + " start!!")
    start=np.mat([[25,25],[225,25],[60,20]])#case1
    goal=np.mat([[225,175],[25,175],[150,175]])

#    start=np.mat([[10,100],[100,25],[240,100]])#case2
#    goal=np.mat([[240,100],[125,175],[10,100]])

    #start=np.mat([[25,100],[225,100],[240,25]])#case3
    #goal=np.mat([[225,100],[25,100],[60,180]])

    #start=np.mat([[25,25],[225,180],[240,25]])#case4
    #goal=np.mat([[125,100],[125,100],[125,100]])

    # start and goal position
    robot_size = 5.0  # [m]

    obs_x,obs_y=obstacle_space(start,goal)

    rx1,ry1,rx2,ry2,rx3,ry3 = PRM_planning(start,goal,obs_x, obs_y, robot_size)
    nodes=Global_path(rx1,ry1,rx2,ry2,rx3,ry3,robot_size)
    plot_global(nodes)


    assert rx1, 'Cannot found path'
    if show_animation:
        plt.plot(obs_x, obs_y, ".k")
        plt.plot(rx1, ry1, "-r")
        plt.plot(rx2, ry2, "-b")
        plt.plot(rx3, ry3, "-g")

if __name__ == '__main__':
    # parameter
    N_SAMPLE = 1000 # number of sample_points
    N_KNN = 50 # number of edge from one sampled point
    MAX_EDGE_LEN = 50  # [m] Maximum edge length
    show_animation = True
    main()
