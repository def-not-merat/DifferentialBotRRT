import json
import numpy as np
import cv2
from rclpy.node import Node
import rclpy



class RRT:
    def __init__(self, number_nodes, jsonfile, given_r):
        #scaling parameters
        self.t_x = 500
        self.t_y = 500
        self.scale = 100
        self.logger = None

        map = np.ones((1000, 1000, 3), dtype=np.uint8)
        map = map*255
        obs = jsonfile

        for ob in obs:
            x,y = self.scale_point([obs[ob]["x"],obs[ob]["y"]])
            r = int((obs[ob]["r"]+given_r) * self.scale)
            map = cv2.circle(map, (y, x), radius=r, color=(0, 0, 0), thickness=-1)


        curr_node = 0
        num_nodes = number_nodes

        rng = np.random.default_rng(5)
        nodes = rng.integers(low=0, high=1000, size=(num_nodes,2), dtype=int)
        nodes[0] = [500,500]

        #find "num_nodes" nodes that are not in the obsticles
        while curr_node<num_nodes:
            x,y = nodes[curr_node]
            if map[x,y,0] != 255:
                nodes[curr_node] = np.random.randint(0, high=1000, size=2, dtype=int)
                continue
            map[x,y] = [0,0,255]
            curr_node += 1

        #calculate pairwise distance of all nodes (used to find the best tree)
        dists = np.zeros((num_nodes,num_nodes))
        for i in range(num_nodes):
            dif_squared = np.square(nodes-nodes[i])
            dist = dif_squared[:,0] + dif_squared[:,1]
            dists[i] = dist
        dists = np.sqrt(dists)        

        m = np.copy(map)
        RRT = {0:[]}
        bests = np.copy(dists[0])
        bests_ind = np.zeros((num_nodes))
        bests[0] = np.inf

        #build the tree
        for coun in range(num_nodes-1):
            
            p_new = int(np.argmin(bests)) #new candidate
            p_parent = int(bests_ind[p_new]) #parent of candidate
            
            intersects,x_inter,y_inter = self.intersects_2(nodes[p_parent], nodes[p_new], map)
            
            if intersects: #continue if intersection
                bests[p_new] = np.inf
                continue

            #get ready for next candidate selection
            RRT[p_parent].append(p_new)
            RRT[p_new] = [p_parent]
            bests[p_new] = np.inf
            
            m = cv2.line(m, np.flip(nodes[p_new]), np.flip(nodes[p_parent]), (0,0,0), 2)
            
            dif_squared = np.square(nodes-nodes[p_new])
            new_dist = dif_squared[:,0] + dif_squared[:,1]
            new_dist = np.sqrt(new_dist)

            for i in range(num_nodes):
                if bests[i] == np.inf:
                    continue
                if new_dist[i]<bests[i]:
                    bests[i] = new_dist[i]
                    bests_ind[i] = p_new 

        self.m = m #used for visualization
        self.RRT = RRT #tree structure
        self.map = map #obsticles in the map
        self.nodes = nodes #all nodes
        self.num_nodes = num_nodes #num of nodes

    def scale_point(self,point):
        return [int(point[0]*self.scale+self.t_x), int(point[1]*self.scale+self.t_y)]

    def inv_scale(self,point):
        return [(point[0]-self.t_x)/self.scale, (point[1]-self.t_y)/self.scale]

    def get_nodes(self):
        return self.nodes

    def get_map(self):
        return self.map

    def _set_logger(self, logger):
        self.logger = logger 


    def intersects_2(self, a, b, map):
        #used to see if a line (passing a and b) intersects our obsticles
        x,y = a[0],a[1]
        dx = float(b[0]-a[0])
        dy = float(b[1]-a[1])
        steps = max(abs(dx),abs(dy))
        if steps == 0:
            return False,None,None
        dx = dx/steps
        dy = dy/steps
        for i in range(int(steps)):
            nx = dx*i + x
            ny = dy*i + y
            if map[int(nx),int(ny),2] != 255:
                return True,int(nx),int(ny)
        return False,None,None    

    def find_best_path(self, orig_start, orig_goal):
        start = self.scale_point(orig_start)
        goal = self.scale_point(orig_goal)


        nodes = self.nodes
        RRT=self.RRT
        map=self.map
        m=self.m
        dif_squared = np.square(nodes-start)
        new_dist = dif_squared[:,0] + dif_squared[:,1]
        new_dist = np.sqrt(new_dist)
        argdist = np.argsort(new_dist)

        #keep looking for nodes closest to start and goal that
        #dont intersect
        start_ind=-1
        for i in argdist:
            intersects,x_inter,y_inter = self.intersects_2(start, nodes[i], map)
            if intersects or i not in RRT:   
                continue
            start_ind=i
            break

        dif_squared = np.square(nodes-goal)
        new_dist = dif_squared[:,0] + dif_squared[:,1]
        new_dist = np.sqrt(new_dist)
        argdist = np.argsort(new_dist)

        end_ind=-1
        for i in argdist:
            intersects,x_inter,y_inter = self.intersects_2(goal, nodes[i], map)
            if intersects or i not in RRT:
                continue
            end_ind = i
            break
        if end_ind==-1 or start_ind==-1:
            self.logger.error(f"given point is inside obsticle")
            return None

        #find path from start to goal in RTT
        def dfs (RRT, curr, end_ind, path, visited):
            if curr == end_ind:
                return True
                
            if visited[curr]==1:
                return False
            visited[curr]=1
            path.append(curr)
            for i in RRT[curr]:
                if visited[i] == 0:
                    a = dfs (RRT, i, end_ind, path, visited)
                    if a:
                        return True
            path.pop()
            return False

        visited = np.zeros(self.num_nodes)
        queue=[]
        found = dfs (RRT, start_ind, end_ind, queue, visited)

        #show the path on image
        if found:
            path = []
            for i in range(1,len(queue)):
                path.append(nodes[queue[i]])
                show_path = cv2.line(m, np.flip(nodes[queue[i-1]]), np.flip(nodes[queue[i]]), (255,0,0), 2)
            show_path = cv2.line(show_path, np.flip(nodes[queue[0]]), np.flip(start), (255,0,0), 2)
            show_path = cv2.line(show_path, np.flip(nodes[queue[-1]]), np.flip(goal), (255,0,0), 2)

            path = [orig_start] + [self.inv_scale(point) for point in path] + [orig_goal]
            return show_path, path
        else:
            show_path = cv2.circle(m, (goal[1], goal[0]), radius=10, color=(255, 0, 0), thickness=-1)
            self.logger.error(f'nothing found')
            cv2.imshow('map', show_path)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()
            return show_path, None

def main(args=None):
    pass
    
if __name__ == '__main__':
    main()